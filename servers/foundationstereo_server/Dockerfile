# FoundationStereo stereo-depth service: POST /depth on port 5005.
# Built by compose.yaml at the repository root.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

# Python environment validated with this service.
RUN uv venv --python 3.11 /opt/venv
COPY servers/foundationstereo_server/requirements.txt /tmp/requirements.txt
RUN uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 -r /tmp/requirements.txt

# FoundationStereo at the tested commit. Its bundled DINOv2 is the torch.hub code it loads.
ARG FOUNDATIONSTEREO_COMMIT=5b8527e90babace45d6acdb5acc2840526906302
ENV TORCH_HOME=/opt/cache/torch HF_HOME=/opt/cache/huggingface
RUN git clone https://github.com/NVlabs/FoundationStereo.git /opt/FoundationStereo \
    && git -C /opt/FoundationStereo checkout $FOUNDATIONSTEREO_COMMIT \
    && mkdir -p $TORCH_HOME/hub \
    && cp -r /opt/FoundationStereo/dinov2 $TORCH_HOME/hub/facebookresearch_dinov2_main

# Weights at pinned versions: the stereo checkpoint and the EdgeNeXt backbone timm loads.
# timm asks the offline Hugging Face cache for "main", so main names the pinned revision.
ARG EDGENEXT_REVISION=81bd8f2ee56330bdf6a7a2e47e1b2b82eb34f978
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('timm/edgenext_small.usi_in1k', revision='$EDGENEXT_REVISION', \
allow_patterns=['model.safetensors'])" \
    && mkdir -p $HF_HOME/hub/models--timm--edgenext_small.usi_in1k/refs \
    && printf %s $EDGENEXT_REVISION > $HF_HOME/hub/models--timm--edgenext_small.usi_in1k/refs/main
RUN mkdir -p /models && curl -fL -o /models/model_best_bp2.pth \
    https://huggingface.co/Felix-Zhenghao/FoundationStereo/resolve/204d16d379a6165e01705b5f056a7688789edfd3/model_best_bp2.pth \
    && echo "60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1  /models/model_best_bp2.pth" | sha256sum -c -
COPY servers/foundationstereo_server/cfg.yaml /models/cfg.yaml

COPY servers/foundationstereo_server/server.py /app/server.py
ENV HF_HUB_OFFLINE=1 \
    FOUNDATIONSTEREO_ROOT=/opt/FoundationStereo \
    FOUNDATIONSTEREO_CKPT=/models/model_best_bp2.pth \
    FOUNDATIONSTEREO_HOST=0.0.0.0 FOUNDATIONSTEREO_PORT=5005
EXPOSE 5005
CMD ["python", "/app/server.py"]
