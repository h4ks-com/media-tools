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
ADD --checksum=sha256:60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a --chmod=644 \
    https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx /tools/isnet-general-use.onnx
RUN python -m zipfile -e /tools/gltfpack.zip /tools && chmod 0755 /tools/gltfpack && rm /tools/gltfpack.zip

# Karaoke lyrics and cover titles use Lilita One; libass falls back to Noto for other scripts.
ADD --checksum=sha256:f5b641c45c69d772ee4eda687bc9fda411d5cad6b0b45371491da4580cbc8d59 --chmod=644 \
    https://raw.githubusercontent.com/google/fonts/23e54b51ddff/ofl/lilitaone/LilitaOne-Regular.ttf /tools/fonts/LilitaOne-Regular.ttf
ADD --checksum=sha256:bfb7bb691513f12e734dc346c03a03f784912432d7e3fa8e56efcf906fe86b3d --chmod=644 \
    "https://raw.githubusercontent.com/google/fonts/23e54b51ddff/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf" /tools/fonts/NotoSans.ttf
ADD --checksum=sha256:c2f3b4d463500a2ddcd3849cded1fceeb9fd6d1c32e6cbecd568453ba50fc68f --chmod=644 \
    "https://raw.githubusercontent.com/google/fonts/23e54b51ddff/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf" /tools/fonts/NotoSansJP.ttf
ADD --checksum=sha256:194018e6b2b293a7964f037b25c0249ce1418bc9ab3c971060a03aa57861e252 --chmod=644 \
    "https://raw.githubusercontent.com/google/fonts/23e54b51ddff/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf" /tools/fonts/NotoSansKR.ttf
ADD --checksum=sha256:a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da --chmod=644 \
    "https://raw.githubusercontent.com/google/fonts/23e54b51ddff/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf" /tools/fonts/NotoSansSC.ttf
COPY --chmod=644 fonts.conf /tools/fonts.conf
# ADD creates the folders it fills as root-only, and the pod reads them as an unprivileged user.
RUN chmod -R a+rX /tools


FROM python:3.14-slim

ENV PATH="/app/.venv/bin:$PATH" \
    GLTFPACK=/opt/tools/gltfpack \
    CUTOUT_MODEL=/opt/tools/isnet-general-use.onnx \
    FFMPEG=/opt/tools/ffmpeg \
    FFPROBE=/opt/tools/ffprobe \
    FONTS_DIR=/opt/tools/fonts \
    FONTCONFIG_FILE=/opt/tools/fonts.conf

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /tools /opt/tools
COPY --from=mwader/static-ffmpeg:9.0.2@sha256:7d9bdaaf887f7e6ce6151f67325c344074b5ff1fb75316011c3376503e449a7b \
    /ffmpeg /ffprobe /opt/tools/

# We remove the base image's pip, since uv installs our packages and pip vendors a vulnerable msgpack.
RUN /usr/local/bin/python -m pip uninstall --yes pip \
    && groupadd --system --gid 1000 media && useradd --system --uid 1000 --gid media media

USER 1000:1000

EXPOSE 8080

CMD ["uvicorn", "media_tools.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
