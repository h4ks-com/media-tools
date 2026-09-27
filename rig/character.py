"""Texture a wrapped character from its front and back pictures and rig it from its drawn pose.

Blender runs this in its own process:
python character.py <model.glb> <front.png> <back.png> <points.json> <out.glb> <rig|texture>
The points are the pose skeleton the pictures were drawn from, as OpenPose COCO-18 points in a
square picture of POINTS_SIZE pixels. With texture, it only paints the model for UniRig to rig.
"""

import json
import os
import sys
import time
from dataclasses import dataclass

import bpy

# isort: split
# Blender's bundled modules only become importable once bpy itself is loaded.
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

POINTS_SIZE = 512
TEXTURE_SIZE = 2048
FRONT = np.array([0.0, -1.0, 0.0])  # glTF characters face +Z, which Blender imports as -Y
HALF = 0.5
SEEN = 0.02  # the smallest summed picture weight that counts as seen
FACING_POWER = 1.5
OUTLINE_PIXELS = 6
SEAM_TEXELS = 8
LIMB_RADIUS = 0.035  # of the model's height, the reach we center limb joints within
Box = tuple[int, int, int, int]
Fit = tuple[float, float, float, float]
started = time.time()


def lap(label: str) -> None:
    print(f"{label}: {time.time() - started:.1f}s", flush=True)


def shrink(mask: np.ndarray, times: int) -> np.ndarray:
    for _ in range(times):
        mask = mask & np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1)
        mask &= np.roll(mask, -1, 1)
    return mask


def grow(mask: np.ndarray) -> np.ndarray:
    return (
        mask
        | np.roll(mask, 1, 0)
        | np.roll(mask, -1, 0)
        | np.roll(mask, 1, 1)
        | np.roll(mask, -1, 1)
    )


@dataclass
class Picture:
    """A cutout: RGB rows top-down, its outline, the part well inside it, and its box."""

    pixels: np.ndarray
    alpha: np.ndarray
    solid: np.ndarray
    box: Box

    @classmethod
    def load(cls, path: str) -> "Picture":
        image = bpy.data.images.load(path)
        image.colorspace_settings.name = (
            "Non-Color"  # we copy the file's own values into the texture
        )
        width, height = image.size
        pixels = np.empty(width * height * 4, np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(height, width, 4)[::-1]
        alpha = pixels[:, :, 3] > HALF
        rows, cols = np.nonzero(alpha)
        box = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
        # The outline blends into the background, so only pixels well inside the character count.
        return cls(pixels[:, :, :3], alpha, shrink(alpha, OUTLINE_PIXELS), box)

    def sample(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the colors and the inside-the-character flags at picture points."""
        height, width = self.solid.shape
        pixel = np.clip(np.round(xy).astype(int), 0, [width - 1, height - 1])
        return self.pixels[pixel[..., 1], pixel[..., 0]], self.solid[pixel[..., 1], pixel[..., 0]]


class Projection:
    """Where a model point lands in a picture, fitted so both outlines cover each other."""

    def __init__(self, coords: np.ndarray, picture: Picture, mirrored: bool) -> None:
        self.low = coords.min(0)
        self.high = coords.max(0)
        self.box = picture.box
        self.mirrored = mirrored
        self.fit: Fit = (1.0, 1.0, 0.0, 0.0)
        self.fit = self.search(coords, picture.alpha)

    def points(self, xyz: np.ndarray, fit: Fit | None = None) -> np.ndarray:
        sx, sy, dx, dy = fit or self.fit
        x0, y0, x1, y1 = self.box
        across = (xyz[..., 0] - self.low[0]) / (self.high[0] - self.low[0])
        if self.mirrored:
            across = 1 - across  # seen from behind, the character's left is on the picture's left
        px = x0 + across * (x1 - x0)
        py = y0 + (self.high[2] - xyz[..., 2]) / (self.high[2] - self.low[2]) * (y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return np.stack([cx + (px - cx) * sx + dx, cy + (py - cy) * sy + dy], axis=-1)

    def model_point(self, px: float, py: float) -> np.ndarray:
        """Return the model point, on the Y=0 plane, that lands on a picture pixel."""
        sx, sy, dx, dy = self.fit
        x0, y0, x1, y1 = self.box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        px, py = cx + (px - dx - cx) / sx, cy + (py - dy - cy) / sy
        across = (px - x0) / (x1 - x0)
        height = self.high[2] - self.low[2]
        return np.array(
            [
                self.low[0] + across * (self.high[0] - self.low[0]),
                0.0,
                self.high[2] - (py - y0) / (y1 - y0) * height,
            ]
        )

    def search(self, coords: np.ndarray, alpha: np.ndarray) -> Fit:
        height, width = alpha.shape

        def overlap(fit: Fit) -> float:
            mask = np.zeros_like(alpha)
            points = np.round(self.points(coords, fit)).astype(int)
            inside = (
                (points[:, 0] >= 0)
                & (points[:, 0] < width)
                & (points[:, 1] >= 0)
                & (points[:, 1] < height)
            )
            mask[points[inside, 1], points[inside, 0]] = True
            mask = grow(mask)
            return float((mask & alpha).sum() / max(1, (mask | alpha).sum()))

        # A coarse grid finds the neighbourhood, then two finer rounds settle the fit around it.
        coarse = [
            (sx, sy, dx, dy)
            for sx in np.linspace(0.92, 1.08, 5)
            for sy in np.linspace(0.92, 1.08, 5)
            for dx in np.linspace(-32, 32, 5)
            for dy in np.linspace(-32, 32, 5)
        ]
        best = max(coarse, key=overlap)
        steps = (0.02, 0.02, 8.0, 8.0)
        for _ in range(2):
            around = [
                (
                    best[0] + a * steps[0],
                    best[1] + b * steps[1],
                    best[2] + c * steps[2],
                    best[3] + d * steps[3],
                )
                for a in (-1, 0, 1)
                for b in (-1, 0, 1)
                for c in (-1, 0, 1)
                for d in (-1, 0, 1)
            ]
            best = max(around, key=overlap)
            steps = (steps[0] / 2, steps[1] / 2, steps[2] / 2, steps[3] / 2)
        return best


@dataclass
class View:
    picture: Picture
    projection: Projection
    toward: np.ndarray
    visible: np.ndarray  # per vertex: whether the camera sees it


class Baker:
    """Bakes shader values into square float textures through the mesh's own UVs."""

    def __init__(self, mesh: bpy.types.Object) -> None:
        self.mesh = mesh
        material = bpy.data.materials.new("character")
        material.use_nodes = True
        mesh.data.materials.clear()
        mesh.data.materials.append(material)
        self.nodes, self.links = material.node_tree.nodes, material.node_tree.links
        self.output = self.nodes["Material Output"]
        self.principled = self.nodes["Principled BSDF"]
        self.geometry = self.nodes.new("ShaderNodeNewGeometry")
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1

    def bake(self, socket: bpy.types.NodeSocket) -> np.ndarray:
        image = bpy.data.images.new("bake", TEXTURE_SIZE, TEXTURE_SIZE, float_buffer=True)
        emission = self.nodes.new("ShaderNodeEmission")
        holder = self.nodes.new("ShaderNodeTexImage")
        holder.image = image
        self.links.new(socket, emission.inputs["Color"])
        self.links.new(emission.outputs["Emission"], self.output.inputs["Surface"])
        self.nodes.active = holder
        bpy.ops.object.bake(type="EMIT", margin=0)
        values = np.empty(TEXTURE_SIZE * TEXTURE_SIZE * 4, np.float32)
        image.pixels.foreach_get(values)
        self.nodes.remove(emission)
        self.nodes.remove(holder)
        bpy.data.images.remove(image)
        return values.reshape(TEXTURE_SIZE, TEXTURE_SIZE, 4)

    def bake_vertex_values(self, name: str, values: np.ndarray) -> np.ndarray:
        """Bake per-vertex values, which Blender blends across each triangle, into a texture."""
        attribute = self.mesh.data.color_attributes.new(name, "FLOAT_COLOR", "POINT")
        rgba = np.ones((len(values), 4), np.float32)
        rgba[:, : values.shape[1]] = values
        attribute.data.foreach_set("color", rgba.ravel())
        node = self.nodes.new("ShaderNodeVertexColor")
        node.layer_name = name
        baked = self.bake(node.outputs["Color"])
        self.nodes.remove(node)
        self.mesh.data.color_attributes.remove(self.mesh.data.color_attributes[name])
        return baked[:, :, : values.shape[1]]

    def finish(self, colors: np.ndarray) -> None:
        """Use the colors as the model's base color texture."""
        for node in list(self.nodes):
            if node not in (self.output, self.principled):
                self.nodes.remove(node)
        image = bpy.data.images.new("texture", TEXTURE_SIZE, TEXTURE_SIZE)
        rgba = np.concatenate(
            [np.clip(colors, 0, 1), np.ones((TEXTURE_SIZE, TEXTURE_SIZE, 1), np.float32)], axis=2
        )
        image.pixels.foreach_set(rgba.ravel())
        # Pixels set on a generated image only survive packing once the image is written to a file.
        image.filepath_raw = f"/tmp/character-{started}.png"  # nosec B108: the pod's own private /tmp
        image.file_format = "PNG"
        image.save()
        image.pack()
        texture = self.nodes.new("ShaderNodeTexImage")
        texture.image = image
        self.links.new(texture.outputs["Color"], self.principled.inputs["Base Color"])
        self.links.new(self.principled.outputs["BSDF"], self.output.inputs["Surface"])
        self.principled.inputs["Roughness"].default_value = 0.85
        self.principled.inputs["Metallic"].default_value = 0.0


def vertex_arrays(mesh: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    coords = np.empty(len(mesh.data.vertices) * 3, np.float32)
    mesh.data.vertices.foreach_get("co", coords)
    normals = np.empty(len(mesh.data.vertices) * 3, np.float32)
    mesh.data.vertices.foreach_get("normal", normals)
    return coords.reshape(-1, 3), normals.reshape(-1, 3)


def views(
    mesh: bpy.types.Object, coords: np.ndarray, front_path: str, back_path: str
) -> list[View]:
    """Fit both pictures and find which vertices each camera really sees.

    A ray from each vertex toward the camera must leave the model, so a ponytail behind the head
    never gets the face painted on it.
    """
    bvh = BVHTree.FromObject(mesh, bpy.context.evaluated_depsgraph_get())
    result = []
    for path, toward, mirrored in ((front_path, FRONT, False), (back_path, -FRONT, True)):
        picture = Picture.load(path)
        direction = Vector(toward)
        visible = np.fromiter(
            (bvh.ray_cast(Vector(co) + direction * 1e-3, direction)[0] is None for co in coords),
            bool,
            len(coords),
        )
        result.append(View(picture, Projection(coords, picture, mirrored), toward, visible))
    return result


def vertex_fill(coords: np.ndarray, normals: np.ndarray, seen_by: list[View]) -> np.ndarray:
    """Blend the pictures per vertex; unseen vertices take the nearest seen vertex's color."""
    weight_sum = np.zeros(len(coords), np.float32)
    color_sum = np.zeros((len(coords), 3), np.float32)
    for view in seen_by:
        colors, inside = view.picture.sample(view.projection.points(coords))
        weight = np.clip(normals @ view.toward, 0, 1) ** FACING_POWER * inside * view.visible
        color_sum += colors * weight[:, None]
        weight_sum += weight
    seen = weight_sum > SEEN
    fill = np.zeros_like(color_sum)
    fill[seen] = color_sum[seen] / weight_sum[seen, None]
    seen_index = np.flatnonzero(seen)
    tree = KDTree(len(seen_index))
    for i, index in enumerate(seen_index):
        tree.insert(coords[index], i)
    tree.balance()
    for index in np.flatnonzero(~seen):
        fill[index] = fill[seen_index[tree.find(coords[index])[1]]]
    return fill


def spread_into_seams(colors: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Grow each UV island's colors outward so filtering at seams never samples the empty gaps."""
    filled = covered.copy()
    for _ in range(SEAM_TEXELS):
        grown = np.zeros_like(colors)
        count = np.zeros(filled.shape, np.float32)
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grown += np.roll(colors * filled[..., None], shift, (0, 1))
            count += np.roll(filled, shift, (0, 1))
        new = ~filled & (count > 0)
        colors[new] = grown[new] / count[new, None]
        filled |= new
    return colors


def texture(
    mesh: bpy.types.Object, front_path: str, back_path: str
) -> tuple[np.ndarray, Projection]:
    """Paint the model from both pictures by how much each surface faces them; return the front fit.

    Each texel takes the pictures in proportion to how much it faces them and whether that camera
    sees it; texels no picture sees keep the per-vertex fill.
    """
    coords, normals = vertex_arrays(mesh)
    seen_by = views(mesh, coords, front_path, back_path)
    lap("fitted and ray cast both pictures")
    baker = Baker(mesh)
    position = baker.bake(baker.geometry.outputs["Position"])[:, :, :3]
    normal = baker.bake(baker.geometry.outputs["Normal"])[:, :, :3]
    covered = np.linalg.norm(normal, axis=2) > HALF
    colors = baker.bake_vertex_values("fill", vertex_fill(coords, normals, seen_by))
    color_sum = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), np.float32)
    weight_sum = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE), np.float32)
    for index, view in enumerate(seen_by):
        visible = (
            baker.bake_vertex_values(f"visible{index}", view.visible[:, None].astype(np.float32))[
                :, :, 0
            ]
            > HALF
        )
        sampled, inside = view.picture.sample(view.projection.points(position))
        weight = np.clip(normal @ view.toward, 0, 1) ** FACING_POWER * inside * visible * covered
        color_sum += sampled * weight[..., None]
        weight_sum += weight
    seen = weight_sum > SEEN
    colors[seen] = color_sum[seen] / weight_sum[seen, None]
    baker.finish(spread_into_seams(colors, covered))
    lap("textured")
    return coords, seen_by[0].projection


def joints(
    coords: np.ndarray, front: Projection, points: dict[int, tuple[float, float]], scale: float
) -> dict[int, np.ndarray]:
    """Place the drawn pose's joints on the model, centered inside the mesh around each one."""
    limb = LIMB_RADIUS * (front.high[2] - front.low[2])

    def snap(point: np.ndarray, radius: float, move: bool) -> np.ndarray:
        near = coords[np.hypot(coords[:, 0] - point[0], coords[:, 2] - point[2]) < radius]
        if len(near) == 0:
            return point
        snapped = point.copy()
        snapped[1] = (near[:, 1].min() + near[:, 1].max()) / 2
        if move:
            snapped[0], snapped[2] = near[:, 0].mean(), near[:, 2].mean()
        return snapped

    placed = {k: front.model_point(x * scale, y * scale) for k, (x, y) in points.items()}
    for k in (3, 4, 6, 7, 9, 10, 12, 13):  # elbows, wrists, knees and ankles sit inside a limb
        placed[k] = snap(placed[k], limb, move=True)
    for k in (0, 1, 2, 5, 8, 11):  # the others sit on the torso, so we only center their depth
        placed[k] = snap(placed[k], limb * 1.5, move=False)
    return placed


def skeleton(coords: np.ndarray, front: Projection, p: dict[int, np.ndarray]) -> dict[str, tuple]:
    """Lay the 22 VRoid bones between the joints, from the hips up the spine and out the limbs."""
    height = front.high[2] - front.low[2]
    limb = LIMB_RADIUS * height
    hips = (p[8] + p[11]) / 2 + np.array([0, 0, 0.02 * height])
    neck = p[1]
    top = np.array([neck[0], neck[1], front.high[2]])

    def spine(fraction: float) -> np.ndarray:
        return hips + (neck - hips) * fraction

    def toe(ankle: np.ndarray) -> np.ndarray:
        near = coords[(np.abs(coords[:, 0] - ankle[0]) < limb * 1.5) & (coords[:, 2] < ankle[2])]
        front_y = near[:, 1].min() if len(near) else ankle[1] - 0.05 * height
        return np.array(
            [ankle[0], ankle[1] + (front_y - ankle[1]) * 0.6, front.low[2] + 0.02 * height]
        )

    bones = {
        "J_Bip_C_Hips": (hips, spine(0.25), None),
        "J_Bip_C_Spine": (spine(0.25), spine(0.5), "J_Bip_C_Hips"),
        "J_Bip_C_Chest": (spine(0.5), spine(0.75), "J_Bip_C_Spine"),
        "J_Bip_C_UpperChest": (spine(0.75), neck, "J_Bip_C_Chest"),
        "J_Bip_C_Neck": (neck, neck + (top - neck) * 0.25, "J_Bip_C_UpperChest"),
        "J_Bip_C_Head": (neck + (top - neck) * 0.25, top, "J_Bip_C_Neck"),
    }
    for side, shoulder, elbow, wrist, hip, knee, ankle in (
        ("R", 2, 3, 4, 8, 9, 10),
        ("L", 5, 6, 7, 11, 12, 13),
    ):
        tip = toe(p[ankle])
        bone = f"J_Bip_{side}_"
        bones |= {
            bone + "Shoulder": (
                spine(0.85) + (p[shoulder] - neck) * 0.25,
                p[shoulder],
                "J_Bip_C_UpperChest",
            ),
            bone + "UpperArm": (p[shoulder], p[elbow], bone + "Shoulder"),
            bone + "LowerArm": (p[elbow], p[wrist], bone + "UpperArm"),
            bone + "Hand": (p[wrist], p[wrist] + (p[wrist] - p[elbow]) * 0.45, bone + "LowerArm"),
            bone + "UpperLeg": (p[hip], p[knee], "J_Bip_C_Hips"),
            bone + "LowerLeg": (p[knee], p[ankle], bone + "UpperLeg"),
            bone + "Foot": (p[ankle], tip, bone + "LowerLeg"),
            bone + "ToeBase": (tip, tip + np.array([0, -0.04 * height, 0]), bone + "Foot"),
        }
    return bones


def build_armature(bones: dict[str, tuple]) -> bpy.types.Object:
    data = bpy.data.armatures.new("Armature")
    armature = bpy.data.objects.new("Armature", data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    for name, (head, tail, parent) in bones.items():
        bone = data.edit_bones.new(name)
        bone.head, bone.tail = Vector(head), Vector(tail)
        if parent:
            bone.parent = data.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def parent_to(armature: bpy.types.Object, mesh: bpy.types.Object, kind: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type=kind)


def welded_copy(mesh: bpy.types.Object) -> bpy.types.Object:
    welded = mesh.copy()
    welded.data = mesh.data.copy()
    bpy.context.collection.objects.link(welded)
    geometry = bmesh.new()
    geometry.from_mesh(welded.data)
    bmesh.ops.remove_doubles(geometry, verts=geometry.verts, dist=1e-4)
    geometry.to_mesh(welded.data)
    geometry.free()
    return welded


def rig(mesh: bpy.types.Object, bones: dict[str, tuple]) -> None:
    """Build the armature and skin the mesh with Blender's heat weights.

    The glTF import splits vertices at every UV seam, which leaves heat weighting with islands, so
    we weight a welded copy and transfer its weights back by nearest surface.
    """
    armature = build_armature(bones)
    welded = welded_copy(mesh)
    parent_to(armature, welded, "ARMATURE_AUTO")
    weighted = {g.group for v in welded.data.vertices for g in v.groups if g.weight > 0}
    empty = [g.name for g in welded.vertex_groups if g.index not in weighted]
    if empty:
        raise SystemExit(f"heat weighting left these bones without vertices: {', '.join(empty)}")
    for group in welded.vertex_groups:
        mesh.vertex_groups.new(name=group.name)
    transfer = mesh.modifiers.new("weights", "DATA_TRANSFER")
    transfer.object = welded
    transfer.use_vert_data = True
    transfer.data_types_verts = {"VGROUP_WEIGHTS"}
    transfer.vert_mapping = "POLYINTERP_NEAREST"
    transfer.layers_vgroup_select_src = "ALL"
    transfer.layers_vgroup_select_dst = "NAME"
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.modifier_apply(modifier=transfer.name)
    bpy.data.objects.remove(welded)
    parent_to(armature, mesh, "ARMATURE_NAME")
    for extra in [m for m in mesh.modifiers if m.type == "ARMATURE"][1:]:
        mesh.modifiers.remove(extra)


def import_mesh(model_path: str) -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=model_path)
    mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    for other in [o for o in bpy.data.objects if o is not mesh]:
        bpy.data.objects.remove(other)
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    mesh.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    lap(f"imported {len(mesh.data.vertices)} vertices")
    return mesh


def main() -> None:
    model_path, front_path, back_path, points_path, target, mode = sys.argv[1:7]
    with open(points_path) as points_file:
        points = {int(k): (float(v[0]), float(v[1])) for k, v in json.load(points_file).items()}
    mesh = import_mesh(model_path)
    coords, front = texture(mesh, front_path, back_path)
    if mode == "rig":
        scale = bpy.data.images.load(front_path).size[0] / POINTS_SIZE
        rig(mesh, skeleton(coords, front, joints(coords, front, points, scale)))
        lap("rigged")
    bpy.ops.export_scene.gltf(filepath=target, export_format="GLB", export_animations=False)
    lap("exported")


main()
sys.stdout.flush()
# Blender crashes while shutting down after a glTF export, so we leave before it tears down.
os._exit(0)
