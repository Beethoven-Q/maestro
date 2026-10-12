# GroundedSAM segmentation service: POST /segment on port 8005.
# Built by compose.yaml at the repository root.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

# Python environment validated with this service.
RUN uv venv --python 3.9 /opt/venv
COPY servers/groundedsam_server/requirements.txt /tmp/requirements.txt
RUN uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 -r /tmp/requirements.txt

# Grounded-Segment-Anything at the tested commit; its GroundingDINO CUDA op needs a fix to build
# with current PyTorch. There is no GPU during the build, so the CUDA architectures are named.
ARG GSA_COMMIT=126abe633ffe333e16e4a0a4e946bc1003caf757
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
COPY servers/groundedsam_server/groundingdino-scalar-type.patch /tmp/
RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git /opt/gsa \
    && git -C /opt/gsa checkout $GSA_COMMIT \
    && git -C /opt/gsa apply /tmp/groundingdino-scalar-type.patch \
    && AM_I_DOCKER=true BUILD_WITH_CUDA=true uv pip install --no-deps --no-build-isolation -e /opt/gsa/GroundingDINO \
    && uv pip install --no-deps -e /opt/gsa/segment_anything \
    && python -c "import torch; from groundingdino import _C"

# Weights at pinned versions: GroundingDINO, SAM ViT-H and the BERT text encoder.
ENV HF_HOME=/opt/cache/huggingface
ARG BERT_REVISION=86b5e0934494bd15c9632b12f734a8a67f723594
RUN mkdir -p /models && cd /models \
    && curl -fLO https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    && curl -fLO https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    && printf '%s\n' \
       "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799  groundingdino_swint_ogc.pth" \
       "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e  sam_vit_h_4b8939.pth" \
       | sha256sum -c -
# transformers asks the offline Hugging Face cache for "main", so main names the pinned revision.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('bert-base-uncased', revision='$BERT_REVISION', allow_patterns=[ \
'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt'])" \
    && mkdir -p $HF_HOME/hub/models--bert-base-uncased/refs \
    && printf %s $BERT_REVISION > $HF_HOME/hub/models--bert-base-uncased/refs/main

COPY servers/groundedsam_server/server.py /app/server.py
ENV HF_HUB_OFFLINE=1 \
    GROUNDINGDINO_CKPT=/models/groundingdino_swint_ogc.pth SAM_CKPT=/models/sam_vit_h_4b8939.pth \
    GROUNDEDSAM_PORT=8005
EXPOSE 8005
CMD ["python", "/app/server.py"]
