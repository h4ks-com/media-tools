import pytest

from conftest import glb_with_json
from media_tools.glb import GlbError
from media_tools.glb import read_glb
from media_tools.glb import write_glb


@pytest.mark.parametrize(
    "document",
    [
        {"images": [{"uri": "../../etc/passwd"}]},
        {"buffers": [{"byteLength": 4, "uri": "/etc/passwd"}]},
        {"buffers": [{"byteLength": 4, "uri": 5}]},
    ],
)
def test_outside_files_are_refused(document: dict[str, object]) -> None:
    with pytest.raises(GlbError, match="embed"):
        read_glb(write_glb(document, b""))


def test_embedded_data_is_accepted() -> None:
    document = {"images": [{"uri": "data:image/png;base64,AAAA"}], "buffers": [{"byteLength": 4}]}

    assert read_glb(write_glb(document, b"\0\0\0\0"))[0] == document


@pytest.mark.parametrize("text", [b"{not json", b'{"a": NaN}', b"\xff\xfe", b"[" * 100_000])
def test_a_broken_json_chunk_is_refused(text: bytes) -> None:
    with pytest.raises(GlbError, match="does not parse"):
        read_glb(glb_with_json(text))


def test_writing_a_number_json_cannot_hold_is_refused() -> None:
    with pytest.raises(GlbError, match="cannot write"):
        write_glb({"value": float("nan")}, b"")
