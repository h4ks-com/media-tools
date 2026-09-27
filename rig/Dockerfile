FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# The pod mounts the UniRig and opt-350m Hugging Face cache read-only at HF_HOME and has no network.
ENV DEBIAN_FRONTEND=noninteractive \
    PYOPENGL_PLATFORM=osmesa \
    HF_HOME=/opt/unirig/hf-cache \
    HOME=/tmp \
    PIP_NO_CACHE_DIR=1

# libxi6/libxrender1/libxkbcommon0/libsm6/libgl1 are bpy's headless runtime deps;
# libegl1/libgles2 and libosmesa6-dev (OSMesa software rendering) are pyrender's;
# libgomp1 is the OpenMP runtime UniRig's compiled packages load.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgomp1 \
        libxi6 libxrender1 libxkbcommon0 libsm6 libgl1 \
        libegl1 libgles2 \
        libusb-1.0-0 \
        libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/unirig
RUN git clone https://github.com/VAST-AI-Research/UniRig repo \
    && git -C repo checkout 6793c6640ff01c8fb389f3993434124bb43d2933

WORKDIR /opt/unirig/repo

COPY patches/vertex_group.patch /tmp/vertex_group.patch
RUN git apply /tmp/vertex_group.patch && rm /tmp/vertex_group.patch

# Upstream ar_inference_articulationxl.yaml hardcodes assign_cls: articulationxl and
# ignores --cls vroid, so we copy in two configs that set the vroid class and make
# generate_skeleton.sh output exactly the 22 VRoid body bones.
COPY configs/ar_inference_vroid_forced.yaml configs/system/ar_inference_vroid_forced.yaml
COPY configs/quick_inference_skeleton_vroid_forced.yaml configs/task/quick_inference_skeleton_vroid_forced.yaml

# We install UniRig's packages from a hash-locked file without resolving, because the base
# image already holds torch and its CUDA wheels and pyrender pins an older PyOpenGL.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-deps --require-hashes -r /tmp/requirements.txt && rm /tmp/requirements.txt

# pip reads a wheel's platform tags from its filename, so we keep the upstream name.
ADD --checksum=sha256:1c455db7a1e5d58e9d43370ecfc75a3a2eb1e37ec12353db66d12c23dbbc3ac6 \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl /tmp/
RUN pip install /tmp/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    && rm /tmp/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# We bake the same pinned gltfpack as the main image and use it to strip
# EXT_meshopt_compression, since bpy reads only uncompressed meshes.
ADD --checksum=sha256:ebc236f5f6c08c7e5c5750476a187d24805d44d8c680449c4b7369c333f817b1 \
    https://github.com/zeux/meshoptimizer/releases/download/v1.2/gltfpack-ubuntu.zip /tmp/gltfpack.zip
RUN python -m zipfile -e /tmp/gltfpack.zip /opt/tools \
    && chmod 0755 /opt/tools/gltfpack \
    && rm /tmp/gltfpack.zip

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app.py character.py /opt/unirig/

RUN groupadd --system --gid 1000 rig && useradd --system --uid 1000 --gid rig rig

ENV PATH="/opt/tools:$PATH" \
    GLTFPACK=/opt/tools/gltfpack \
    UNIRIG_DIR=/opt/unirig/repo

USER 1000:1000

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "--app-dir", "/opt/unirig", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
