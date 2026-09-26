FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE come later: baking the weights below needs
# network, and the pod itself never gets any.
ENV DEBIAN_FRONTEND=noninteractive \
    PYOPENGL_PLATFORM=osmesa \
    HF_HOME=/opt/unirig/hf-cache \
    HOME=/tmp \
    PIP_NO_CACHE_DIR=1

# libxi6/libxrender1/libxkbcommon0/libsm6/libgl1 are bpy's headless runtime deps;
# libegl1/libgles2 and libosmesa6-dev (OSMesa software rendering) are pyrender's.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
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

# --cls vroid is a no-op upstream: ar_inference_articulationxl.yaml hardcodes
# assign_cls: articulationxl. These two configs force the vroid class instead, so
# generate_skeleton.sh outputs exactly the 22 VRoid body bones.
COPY configs/ar_inference_vroid_forced.yaml configs/system/ar_inference_vroid_forced.yaml
COPY configs/quick_inference_skeleton_vroid_forced.yaml configs/task/quick_inference_skeleton_vroid_forced.yaml

# requirements.txt pulls in flash_attn too, which tries to build from source; install
# its other packages directly instead and drop in the prebuilt flash_attn wheel below.
RUN pip install transformers==4.51.3 python-box einops omegaconf pytorch_lightning \
        lightning addict timm fast-simplification trimesh open3d pyrender \
        huggingface_hub wandb \
    && pip install numpy==1.26.4 \
    && pip install spconv-cu124 \
    && pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html \
    && pip install bpy==4.2.0 \
    && pip install PyOpenGL==3.1.7 \
    && pip install fastapi==0.141.1 "uvicorn[standard]==0.53.0"

ADD --checksum=sha256:1c455db7a1e5d58e9d43370ecfc75a3a2eb1e37ec12353db66d12c23dbbc3ac6 \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl /tmp/flash_attn.whl
RUN pip install /tmp/flash_attn.whl && rm /tmp/flash_attn.whl

# Same pinned gltfpack as the main image, used here to strip EXT_meshopt_compression
# before handing a mesh to bpy, which can't read it.
ADD --checksum=sha256:ebc236f5f6c08c7e5c5750476a187d24805d44d8c680449c4b7369c333f817b1 \
    https://github.com/zeux/meshoptimizer/releases/download/v1.2/gltfpack-ubuntu.zip /tmp/gltfpack.zip
RUN python -m zipfile -e /tmp/gltfpack.zip /opt/tools \
    && chmod 0755 /opt/tools/gltfpack \
    && rm /tmp/gltfpack.zip

# Bakes the two checkpoints generate_skeleton.sh/generate_skin.sh load, plus the
# tiny facebook/opt-350m config the skeleton model's architecture is built from, so
# the pod needs no network at request time.
# download() (src/inference/download.py) and AutoConfig.from_pretrained() both resolve
# the unpinned "main" ref at request time; the mkdir/printf below point it at the same
# pinned commit so that offline lookup finds it with no network call.
RUN python -c "\
from huggingface_hub import hf_hub_download; \
SKEL_REV = '36842e2b5947e9e60f89275b83208c8e74071c63'; \
OPT_REV = '08ab08cc4b72ff5593870b5d527cf4230323703c'; \
hf_hub_download(repo_id='VAST-AI/UniRig', filename='skeleton/articulation-xl_quantization_256/model.ckpt', revision=SKEL_REV); \
hf_hub_download(repo_id='VAST-AI/UniRig', filename='skin/articulation-xl/model.ckpt', revision=SKEL_REV); \
hf_hub_download(repo_id='facebook/opt-350m', filename='config.json', revision=OPT_REV)" \
    && mkdir -p $HF_HOME/hub/models--VAST-AI--UniRig/refs $HF_HOME/hub/models--facebook--opt-350m/refs \
    && printf '%s' 36842e2b5947e9e60f89275b83208c8e74071c63 > $HF_HOME/hub/models--VAST-AI--UniRig/refs/main \
    && printf '%s' 08ab08cc4b72ff5593870b5d527cf4230323703c > $HF_HOME/hub/models--facebook--opt-350m/refs/main

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app.py /opt/unirig/app.py

RUN groupadd --system --gid 1000 rig && useradd --system --uid 1000 --gid rig rig

ENV PATH="/opt/tools:$PATH" \
    GLTFPACK=/opt/tools/gltfpack \
    UNIRIG_DIR=/opt/unirig/repo

USER 1000:1000

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "--app-dir", "/opt/unirig", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
