"""Read and write binary glTF (GLB) files: a JSON document plus one binary buffer."""

import json
import struct
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


def read_glb(data: bytes) -> tuple[Document, bytearray]:
    """Split a GLB into its JSON document and its binary buffer.

    :raises GlbError: when the data is not a GLB with a JSON chunk.
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
            document = json.loads(chunk)
        elif chunk_type == BIN_CHUNK:
            binary = bytes(chunk)
        offset += CHUNK.size + chunk_length
    if not isinstance(document, dict):
        raise GlbError("the GLB has no JSON chunk")
    return document, bytearray(binary)


def write_glb(document: Document, binary: bytes | bytearray) -> bytes:
    """Join a JSON document and a binary buffer into a GLB, padding both chunks to four bytes."""
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    body = bytes(binary) + b"\0" * (-len(binary) % 4)
    chunks = CHUNK.pack(len(text), JSON_CHUNK) + text
    if body:
        chunks += CHUNK.pack(len(body), BIN_CHUNK) + body
    return HEADER.pack(GLB_MAGIC, 2, HEADER.size + len(chunks)) + chunks
