FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --no-editable

COPY README.md pyproject.toml uv.lock ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# The tools come in at pinned checksums, so the running pod needs no network at all.
ADD --checksum=sha256:ebc236f5f6c08c7e5c5750476a187d24805d44d8c680449c4b7369c333f817b1 \
    https://github.com/zeux/meshoptimizer/releases/download/v1.2/gltfpack-ubuntu.zip /tools/gltfpack.zip
ADD --checksum=sha256:60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a \
    https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx /tools/isnet-general-use.onnx
RUN python -m zipfile -e /tools/gltfpack.zip /tools && chmod 0755 /tools/gltfpack && rm /tools/gltfpack.zip


FROM python:3.14-slim

ENV PATH="/app/.venv/bin:$PATH" \
    GLTFPACK=/opt/tools/gltfpack \
    CUTOUT_MODEL=/opt/tools/isnet-general-use.onnx

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /tools /opt/tools

RUN groupadd --system --gid 1000 media && useradd --system --uid 1000 --gid media media

USER 1000:1000

EXPOSE 8080

CMD ["uvicorn", "media_tools.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
