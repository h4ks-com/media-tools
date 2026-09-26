"""Shrink a GLB for browser viewers with gltfpack and make its materials opaque."""

import subprocess  # nosec B404: we run only the gltfpack binary, with arguments we build
import tempfile
from pathlib import Path

from media_tools.glb import GlbError
from media_tools.glb import read_glb
from media_tools.glb import write_glb


class MeshError(ValueError):
    pass


def triangle_count(data: bytes) -> int:
    """Count the triangles a GLB declares.

    :raises GlbError: when the data is not a GLB.
    """
    document, _ = read_glb(data)
    accessors = document.get("accessors", [])
    total = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            source = primitive.get("indices", primitive.get("attributes", {}).get("POSITION"))
            if source is None:
                raise GlbError("a mesh primitive has no positions")
            total += int(accessors[source]["count"]) // 3
    return total


def make_opaque(data: bytes) -> bytes:
    """Mark every material opaque, since TRELLIS writes BLEND and its vertex alpha shows through."""
    document, binary = read_glb(data)
    for material in document.get("materials", []):
        material["alphaMode"] = "OPAQUE"
        material.pop("alphaCutoff", None)
    return write_glb(document, binary)


def simplify(data: bytes, triangles: int, gltfpack: str, timeout: float) -> bytes:
    """Return the GLB simplified to at most that many triangles, meshopt-compressed and opaque.

    :raises GlbError: when the data is not a GLB.
    :raises MeshError: when gltfpack fails or takes too long.
    """
    ratio = max(0.001, min(1.0, triangles / max(1, triangle_count(data))))
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir, "input.glb")
        target = Path(workdir, "output.glb")
        source.write_bytes(data)
        command = [gltfpack, "-i", str(source), "-o", str(target), "-si", str(ratio), "-cc"]
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)  # nosec B603
        except subprocess.TimeoutExpired as error:
            raise MeshError("gltfpack timed out") from error
        if result.returncode != 0:
            raise MeshError(result.stderr.decode(errors="replace")[-2000:] or "gltfpack failed")
        return make_opaque(target.read_bytes())
