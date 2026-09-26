import io
from collections.abc import Callable

import numpy
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import edit
from conftest import glb_with_json
from conftest import png
from conftest import vroid_rig
from media_tools import app as app_module
from media_tools import mesh
from media_tools.glb import Document
from media_tools.glb import read_glb

client = TestClient(app_module.app)
lenient = TestClient(app_module.app, raise_server_exceptions=False)


def post(path: str, body: bytes) -> tuple[int, bytes]:
    response = client.post(path, content=body, headers={"Content-Type": "application/octet-stream"})
    return response.status_code, response.content


def test_healthz() -> None:
    assert client.get("/healthz").text == "ok"


def test_pose_returns_a_png() -> None:
    response = client.get("/pose", params={"move": "walk", "frames": 6, "frame": 2})

    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == (512, 512)


def test_a_bad_pose_is_a_400() -> None:
    response = client.get("/pose", params={"move": "dance"})

    assert (response.status_code, response.text) == (
        400,
        "move is one of idle, walk, run, jump, attack, hurt",
    )


def test_key_cutout_returns_a_transparent_png() -> None:
    status, body = post("/cutout?method=key", png((64, 64), (20, 20, 44, 44)))

    assert status == 200
    assert numpy.asarray(Image.open(io.BytesIO(body)))[1, 1, 3] == 0


def test_isnet_cutout_uses_the_model_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module.cutter, "mask", lambda picture: Image.new("L", picture.size, 200)
    )

    status, body = post("/cutout", png((32, 32)))

    assert status == 200
    assert numpy.asarray(Image.open(io.BytesIO(body)))[5, 5, 3] == 200


@pytest.mark.parametrize("path", ["/cutout?method=key", "/pixelate"])
def test_a_non_picture_is_a_400(path: str) -> None:
    assert post(path, b"not a picture")[0] == 400


def test_an_empty_body_is_a_400() -> None:
    assert post("/pixelate", b"")[0] == 400


def test_pixelate_returns_pixel_art() -> None:
    status, body = post("/pixelate?size=16&colors=4&scale=2", png((64, 64), (10, 10, 50, 50)))

    assert status == 200
    assert Image.open(io.BytesIO(body)).size == (32, 32)


@pytest.mark.parametrize(("fmt", "mime"), [("png", "image/png"), ("gif", "image/gif")])
def test_sprite_frames_join_the_frames(fmt: str, mime: str) -> None:
    frames = [png((100, 100), (30, 20, 50 + shift, 80)) for shift in (0, 10)]
    lengths = ",".join(str(len(frame)) for frame in frames)

    response = client.post(
        f"/sprite-frames?lengths={lengths}&format={fmt}&size=32", content=b"".join(frames)
    )

    assert (response.status_code, response.headers["content-type"]) == (200, mime)


@pytest.mark.parametrize(
    ("lengths", "body", "status"),
    [
        ("5", b"12345", 400),
        ("2,x", b"123", 400),
        ("2,2", b"123", 400),
    ],
)
def test_sprite_frames_check_the_lengths(lengths: str, body: bytes, status: int) -> None:
    assert post(f"/sprite-frames?lengths={lengths}", body)[0] == status


def test_sprite_frames_without_a_character_is_a_422() -> None:
    frames = [png((50, 50)), png((50, 50))]
    lengths = ",".join(str(len(frame)) for frame in frames)

    assert post(f"/sprite-frames?lengths={lengths}", b"".join(frames))[0] == 422


def test_sprite_frames_of_garbage_is_a_400() -> None:
    assert post("/sprite-frames?lengths=3,3", b"abcdef")[0] == 400


def test_mesh_runs_gltfpack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mesh, "simplify", lambda data, triangles, gltfpack, timeout: data)

    status, body = post("/mesh?triangles=5000", b"glb bytes")

    assert (status, body) == (200, b"glb bytes")


def test_mesh_of_garbage_is_a_400() -> None:
    assert post("/mesh", b"not a glb at all")[0] == 400


def test_mesh_failure_is_a_422(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object) -> bytes:
        raise mesh.MeshError("gltfpack failed")

    monkeypatch.setattr(mesh, "simplify", fail)

    assert post("/mesh", b"glb") == (422, b"gltfpack failed")


def test_retarget_adds_named_clips(walk: bytes, jump: bytes) -> None:
    character = vroid_rig(walk)
    lengths = f"{len(character)},{len(walk)},{len(jump)}"

    status, body = post(f"/retarget?lengths={lengths}&names=Walk,Jump", character + walk + jump)

    assert status == 200
    assert [clip["name"] for clip in read_glb(body)[0]["animations"]] == ["Walk", "Jump"]


def test_retarget_needs_one_name_per_motion(walk: bytes) -> None:
    character = vroid_rig(walk)

    status, _ = post(
        f"/retarget?lengths={len(character)},{len(walk)}&names=Walk,Jump", character + walk
    )

    assert status == 400


def test_retarget_refuses_large_motions(walk: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "MAX_MOTION_BYTES", 10)
    character = vroid_rig(walk)

    status, _ = post(f"/retarget?lengths={len(character)},{len(walk)}&names=Walk", character + walk)

    assert status == 400


def test_requests_beyond_the_in_flight_cap_are_told_to_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "in_flight", app_module.MAX_IN_FLIGHT)

    assert post("/pixelate", png((32, 32))) == (429, b"busy, try again")


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/pixelate?size=16", [png((64, 64), (10, 10, 50, 50))]),
        ("/sprite-frames?size=16", [png((64, 64), (10, 10, 50, 50))] * 2),
    ],
)
def test_picture_work_runs_in_the_heavy_slot(
    monkeypatch: pytest.MonkeyPatch, path: str, body: list[bytes]
) -> None:
    slotted: list[bool] = []

    def in_slot(work: Callable[[], bytes]) -> bytes:
        slotted.append(True)
        return work()

    monkeypatch.setattr(app_module, "in_slot", in_slot)
    lengths = ",".join(str(len(part)) for part in body)

    status, _ = post(f"{path}&lengths={lengths}", b"".join(body))

    assert (status, slotted) == (200, [True])


def bad_json(_: bytes) -> bytes:
    return glb_with_json(b"{not json")


def far_accessor(walk: bytes) -> bytes:
    def spoil(document: Document, _: bytearray) -> None:
        document["accessors"][1]["byteOffset"] = 10**9

    return edit(walk, spoil)


def short_sampler(walk: bytes) -> bytes:
    def spoil(document: Document, _: bytearray) -> None:
        document["accessors"][1]["count"] = 100

    return edit(walk, spoil)


def numbered_node(walk: bytes) -> bytes:
    def spoil(document: Document, _: bytearray) -> None:
        document["nodes"][3]["name"] = 5

    return edit(walk, spoil)


@pytest.mark.parametrize("spoil", [bad_json, far_accessor, short_sampler, numbered_node])
def test_broken_motions_are_a_422_never_a_500(walk: bytes, spoil: Callable[[bytes], bytes]) -> None:
    character, motion = vroid_rig(walk), spoil(walk)

    response = lenient.post(
        f"/retarget?lengths={len(character)},{len(motion)}&names=Walk", content=character + motion
    )

    assert response.status_code == 422


def test_a_broken_mesh_is_a_400_never_a_500() -> None:
    assert lenient.post("/mesh", content=glb_with_json(b"{not json")).status_code == 400


@pytest.mark.parametrize("path", ["/pixelate", "/cutout?method=key", "/sprite-frames"])
def test_a_truncated_picture_is_a_400_never_a_500(path: str) -> None:
    whole = png((64, 64), (10, 10, 50, 50))
    frames = [whole[:-40], whole[:-40]] if path == "/sprite-frames" else [whole[:-40]]
    joiner = "&" if "?" in path else "?"
    lengths = ",".join(str(len(frame)) for frame in frames)

    response = lenient.post(f"{path}{joiner}lengths={lengths}", content=b"".join(frames))

    assert response.status_code == 400


def test_retarget_onto_an_unrigged_model_is_a_422(walk: bytes) -> None:
    status, body = post(f"/retarget?lengths={len(walk)},{len(walk)}&names=Walk", walk + walk)

    assert status == 422
    assert b"rig it first" in body
