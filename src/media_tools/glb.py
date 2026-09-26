"""Read and write binary glTF (GLB) files: a JSON document plus one binary buffer."""

import json
import struct
from collections.abc import Sized
from typing import Any

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
HEADER = struct.Struct("<III")
CHUNK = struct.Struct("<II")

# A glTF document is free-form JSON, so we keep it as the parsed tree the format defines.
type Document = dict[str, Any]


class GlbError(ValueError):
    pass


def refuse_constant(name: str) -> float:
    raise GlbError(f"the JSON holds {name}")


def parse_json(chunk: bytes) -> object:
    try:
        return json.loads(chunk, parse_constant=refuse_constant)
    except (ValueError, RecursionError) as error:
        raise GlbError(f"the JSON chunk does not parse: {error}") from error


def read_glb(data: bytes) -> tuple[Document, bytearray]:
    """Split a GLB into its JSON document and its binary buffer.

    :raises GlbError: when the data is not a GLB with a JSON chunk, or it points at outside files.
    """
    if len(data) < HEADER.size:
        raise GlbError("not a GLB file")
    magic, _, length = HEADER.unpack_from(data, 0)
    if magic != GLB_MAGIC:
        raise GlbError("not a GLB file")
    offset, document, binary = HEADER.size, None, b""
    while offset + CHUNK.size <= min(length, len(data)):
        chunk_length, chunk_type = CHUNK.unpack_from(data, offset)
        chunk = data[offset + CHUNK.size : offset + CHUNK.size + chunk_length]
        if chunk_type == JSON_CHUNK:
            document = parse_json(chunk)
        elif chunk_type == BIN_CHUNK:
            binary = bytes(chunk)
        offset += CHUNK.size + chunk_length
    if not isinstance(document, dict):
        raise GlbError("the GLB has no JSON chunk")
    refuse_outside_files(document)
    return document, bytearray(binary)


def refuse_outside_files(document: Document) -> None:
    """Allow only embedded data, since gltfpack would read a relative uri from our disk."""
    for key in ("buffers", "images"):
        for item in entries(document, key):
            uri = item.get("uri")
            if uri is not None and not (isinstance(uri, str) and uri.startswith("data:")):
                raise GlbError(f"{key} may only embed their data, not point at {uri!r}")


def entries(document: Document, key: str) -> list[Document]:
    """Return the list of objects under `key`, empty when absent.

    :raises GlbError: when it is not a list of objects.
    """
    value = document.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise GlbError(f"{key} is not a list of objects")
    return value


def natural(value: object, what: str) -> int:
    """Return `value` as a whole number of zero or more.

    :raises GlbError: when it is not one.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GlbError(f"{what} is not a whole number")
    return value


def index_into(value: object, items: Sized, what: str) -> int:
    """Return `value` as an index into `items`.

    :raises GlbError: when it points outside them.
    """
    index = natural(value, what)
    if index >= len(items):
        raise GlbError(f"{what} points past the end")
    return index


def write_glb(document: Document, binary: bytes | bytearray) -> bytes:
    """Join a JSON document and a binary buffer into a GLB, padding both chunks to four bytes.

    :raises GlbError: when the document holds a number JSON cannot hold.
    """
    try:
        text = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    except ValueError as error:
        raise GlbError(f"cannot write the GLB: {error}") from error
    text += b" " * (-len(text) % 4)
    body = bytes(binary) + b"\0" * (-len(binary) % 4)
    chunks = CHUNK.pack(len(text), JSON_CHUNK) + text
    if body:
        chunks += CHUNK.pack(len(body), BIN_CHUNK) + body
    return HEADER.pack(GLB_MAGIC, 2, HEADER.size + len(chunks)) + chunks
