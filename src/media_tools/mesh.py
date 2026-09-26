"""Shrink a GLB for browser viewers with gltfpack and make its materials opaque."""

import subprocess  # nosec B404: we run only the gltfpack binary, with arguments we build
import tempfile
from pathlib import Path

from media_tools.glb import GlbError
from media_tools.glb import entries
from media_tools.glb import index_into
from media_tools.glb import natural
from media_tools.glb import read_glb
from media_tools.glb import write_glb

MIN_RATIO = 1e-6


class MeshError(ValueError):
    pass


def triangle_count(data: bytes) -> int:
    """Count the triangles a GLB declares.

    :raises GlbError: when the data is not a GLB.
    """
    document, _ = read_glb(data)
    accessors = entries(document, "accessors")
    total = 0
    for mesh in entries(document, "meshes"):
        for primitive in entries(mesh, "primitives"):
            attributes = primitive.get("attributes", {})
            if not isinstance(attributes, dict):
                raise GlbError("a mesh primitive's attributes are not an object")
            source = primitive.get("indices", attributes.get("POSITION"))
            if source is None:
                raise GlbError("a mesh primitive has no positions")
            accessor = accessors[index_into(source, accessors, "a mesh primitive")]
            total += natural(accessor.get("count"), "an accessor count") // 3
    return total


def make_opaque(data: bytes) -> bytes:
    """Mark every material opaque, since TRELLIS writes BLEND and its vertex alpha shows through."""
    document, binary = read_glb(data)
    for material in entries(document, "materials"):
        material["alphaMode"] = "OPAQUE"
        material.pop("alphaCutoff", None)
    return write_glb(document, binary)


def simplify(data: bytes, triangles: int, gltfpack: str, timeout: float) -> bytes:
    """Return the GLB simplified toward that many triangles, meshopt-compressed and opaque.

    gltfpack aims at the ratio we give and may stop above it when simplifying further would tear
    the mesh.

    :raises GlbError: when the data is not a GLB.
    :raises MeshError: when gltfpack fails or takes too long.
    """
    ratio = max(MIN_RATIO, min(1.0, triangles / max(1, triangle_count(data))))
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir, "input.glb")
        target = Path(workdir, "output.glb")
        source.write_bytes(data)
        command = [gltfpack, "-i", str(source), "-o", str(target), "-si", f"{ratio:.8f}", "-cc"]
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)  # nosec B603
        except subprocess.TimeoutExpired as error:
            raise MeshError("gltfpack timed out") from error
        if result.returncode != 0:
            raise MeshError(result.stderr.decode(errors="replace")[-2000:] or "gltfpack failed")
        return make_opaque(target.read_bytes())
