import stat
from pathlib import Path

import pytest

from media_tools import mesh
from media_tools.glb import GlbError
from media_tools.glb import read_glb
from media_tools.glb import write_glb

TRIANGLES = {
    "accessors": [{"count": 30}, {"count": 12}],
    "meshes": [{"primitives": [{"indices": 0}, {"attributes": {"POSITION": 1}}]}],
    "materials": [{"alphaMode": "BLEND", "alphaCutoff": 0.5}],
}


def fake_gltfpack(tmp_path: Path, script: str) -> str:
    path = tmp_path / "gltfpack"
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_triangles_count_indices_or_positions() -> None:
    assert mesh.triangle_count(write_glb(TRIANGLES, b"")) == 14


def test_a_primitive_without_positions_is_refused() -> None:
    with pytest.raises(GlbError, match="no positions"):
        mesh.triangle_count(write_glb({"meshes": [{"primitives": [{"attributes": {}}]}]}, b""))


def test_materials_become_opaque() -> None:
    document, _ = read_glb(mesh.make_opaque(write_glb(TRIANGLES, b"\1\2")))

    assert document["materials"] == [{"alphaMode": "OPAQUE"}]


def test_simplify_runs_gltfpack_and_makes_the_result_opaque(tmp_path: Path) -> None:
    # The fake gltfpack copies its input, so we see the arguments reach it and the output come back.
    gltfpack = fake_gltfpack(tmp_path, 'cp "$2" "$4"')

    document, _ = read_glb(mesh.simplify(write_glb(TRIANGLES, b""), 7, gltfpack, 10))

    assert document["materials"][0]["alphaMode"] == "OPAQUE"


def test_a_failing_gltfpack_reports_its_error(tmp_path: Path) -> None:
    gltfpack = fake_gltfpack(tmp_path, "echo broken mesh >&2; exit 1")

    with pytest.raises(mesh.MeshError, match="broken mesh"):
        mesh.simplify(write_glb(TRIANGLES, b""), 7, gltfpack, 10)


def test_a_slow_gltfpack_times_out(tmp_path: Path) -> None:
    gltfpack = fake_gltfpack(tmp_path, "sleep 5")

    with pytest.raises(mesh.MeshError, match="timed out"):
        mesh.simplify(write_glb(TRIANGLES, b""), 7, gltfpack, 0.2)


@pytest.mark.parametrize("data", [b"", b"notaglb-file", write_glb({}, b"")[:12] + b"\0" * 8])
def test_non_glb_data_is_refused(data: bytes) -> None:
    with pytest.raises(GlbError):
        read_glb(data)
