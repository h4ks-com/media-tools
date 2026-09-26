import json
import struct
import subprocess  # nosec B404: we list processes with pgrep
import time
import uuid
from pathlib import Path

import app as rig_app
import pytest
from fastapi.testclient import TestClient

client = TestClient(rig_app.app)


def glb(text: bytes) -> bytes:
    text += b" " * (-len(text) % 4)
    chunks = struct.pack("<II", len(text), 0x4E4F534A) + text
    return struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks


def document_glb(document: object) -> bytes:
    return glb(json.dumps(document).encode())


def script(tmp_path: Path, body: str) -> list[str]:
    path = tmp_path / "stage.sh"
    path.write_text(body)
    return ["bash", str(path)]


def running(command: str) -> list[str]:
    found = subprocess.run(  # nosec B603 B607
        ["pgrep", "-fx", command], capture_output=True, text=True, check=False
    )
    return found.stdout.split()


def test_an_embedded_glb_passes() -> None:
    rig_app.check_embedded(
        document_glb({"buffers": [{"byteLength": 4}], "images": [{"uri": "data:image/png,x"}]})
    )


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ({"buffers": [{"uri": "../secret.bin"}]}, "embed"),
        ({"images": [{"uri": 5}]}, "embed"),
        ({"buffers": ["../secret.bin"]}, "list of objects"),
        ({"images": [None]}, "list of objects"),
        ({"buffers": {}}, "not a list"),
        ([], "not an object"),
    ],
)
def test_linked_or_odd_entries_are_refused(document: object, match: str) -> None:
    with pytest.raises(rig_app.RigError, match=match):
        rig_app.check_embedded(document_glb(document))


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_json_constants_are_refused(constant: bytes) -> None:
    with pytest.raises(rig_app.RigError, match="holds"):
        rig_app.check_embedded(glb(b'{"scale": [' + constant + b"]}"))


@pytest.mark.parametrize(
    ("data", "match"),
    [(b"glTF", "too short"), (glb(b"{not json"), "cannot be read")],
)
def test_broken_glbs_are_refused(data: bytes, match: str) -> None:
    with pytest.raises(rig_app.RigError, match=match):
        rig_app.check_embedded(data)


def test_a_stage_that_times_out_leaves_no_process_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rig_app, "STAGE_TIMEOUT_SECONDS", 0.5)
    sleeper = f"sleep 30.{uuid.uuid4().int % 10**9}"
    command = script(tmp_path, f'cmd="{sleeper}"; eval $cmd; wait\n')

    with pytest.raises(rig_app.RigError, match="timed out"):
        rig_app.run_stage(command, tmp_path / "out", cwd=tmp_path)

    deadline = time.monotonic() + 5
    while running(sleeper) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert running(sleeper) == []


def test_a_stage_that_makes_its_file_passes(tmp_path: Path) -> None:
    rig_app.run_stage(script(tmp_path, "touch out\n"), tmp_path / "out", cwd=tmp_path)


def test_a_stage_that_exits_zero_without_its_file_fails(tmp_path: Path) -> None:
    with pytest.raises(rig_app.RigError, match="python crashed"):
        rig_app.run_stage(script(tmp_path, "echo python crashed\n"), tmp_path / "out")


def test_a_stage_without_output_names_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(rig_app.RigError, match="made no out"):
        rig_app.run_stage(script(tmp_path, "true\n"), tmp_path / "out")


def test_a_failing_stage_reports_its_error(tmp_path: Path) -> None:
    with pytest.raises(rig_app.RigError, match="bad input"):
        rig_app.run_stage(script(tmp_path, "echo bad input >&2; exit 3\n"), tmp_path / "out")


def test_requests_beyond_the_in_flight_cap_are_told_to_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rig_app, "in_flight", rig_app.MAX_IN_FLIGHT)

    response = client.post("/rig", content=b"x")

    assert (response.status_code, response.text) == (429, "busy, try again")


def test_rig_returns_the_rigged_glb(monkeypatch: pytest.MonkeyPatch) -> None:
    model = document_glb({"buffers": [{"byteLength": 4}]})
    monkeypatch.setattr(rig_app, "rig", lambda data: data[::-1])

    response = client.post("/rig", content=model)

    assert (response.status_code, response.content) == (200, model[::-1])
    assert rig_app.in_flight == 0


def test_rig_refuses_a_linked_glb() -> None:
    response = client.post("/rig", content=document_glb({"buffers": [{"uri": "/etc/passwd"}]}))

    assert (response.status_code, response.text) == (400, "the GLB buffers must embed their data")


@pytest.mark.parametrize("body", [b"", b"not a glb"])
def test_rig_refuses_what_is_no_glb(body: bytes) -> None:
    assert client.post("/rig", content=body).status_code == 400


def test_a_failed_rig_is_a_422(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: bytes) -> bytes:
        raise rig_app.RigError("no skeleton")

    monkeypatch.setattr(rig_app, "rig", fail)

    response = client.post("/rig", content=document_glb({}))

    assert (response.status_code, response.text) == (422, "no skeleton")


def test_healthz() -> None:
    assert client.get("/healthz").text == "ok"
