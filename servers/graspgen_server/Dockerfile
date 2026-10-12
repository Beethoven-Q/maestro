# GraspGen grasp-sampling service: POST /grasps on port 8006, meshcat viewer on port 7000.
# Built by compose.yaml at the repository root.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
# GraspGen imports pyrender, which loads pyglet's X11 bindings even without a display.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git libgl1 libglib2.0-0 libgomp1 \
    libx11-6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

# Python environment validated with this service.
RUN uv venv --python 3.9 /opt/venv
COPY servers/graspgen_server/requirements.txt /tmp/requirements.txt
RUN uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0+cu128 torchvision==0.23.0+cu128 -r /tmp/requirements.txt

# GraspGen at the tested commit, with Python 3.9 annotations and float64 meshcat transforms.
# There is no GPU during the build, so the CUDA architectures of pointnet2_ops are named.
ARG GRASPGEN_COMMIT=1c443d96f3109a497cb75ab214eac5c9a1a92b01
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
COPY servers/graspgen_server/*.patch /tmp/patches/
RUN git clone https://github.com/NVlabs/GraspGen.git /opt/GraspGen \
    && git -C /opt/GraspGen checkout $GRASPGEN_COMMIT \
    && git -C /opt/GraspGen apply /tmp/patches/*.patch \
    && uv pip install --no-deps --no-build-isolation -e /opt/GraspGen \
    && MAX_JOBS=8 uv pip install --no-build-isolation /opt/GraspGen/pointnet2_ops

# Franka Panda gripper checkpoint at a pinned revision.
ARG MODELS=https://huggingface.co/adithyamurali/GraspGenModels/resolve/ec1ccbb5eec0680db669246ac312a3636f16ee43/checkpoints
RUN mkdir -p /models && cd /models \
    && curl -fLO $MODELS/graspgen_franka_panda.yml \
    && curl -fLO $MODELS/graspgen_franka_panda_gen.pth \
    && curl -fLO $MODELS/graspgen_franka_panda_dis.pth \
    && printf '%s\n' \
       "3b666d28ffb91001ddb6ba24a2e0c11458478a986b808b493cf6fa9a987c2abd  graspgen_franka_panda.yml" \
       "0597583b89b322d42ceb4e596967d6ed68d1b56cba4039895909ccd5bdc66eff  graspgen_franka_panda_gen.pth" \
       "e47d703c63b54c2d11fbc1effd43898f251b4147250888541e3b16e9c0d19e1c  graspgen_franka_panda_dis.pth" \
       | sha256sum -c -

COPY servers/graspgen_server/server.py /app/server.py
ENV GRASPGEN_CONFIG=/models/graspgen_franka_panda.yml GRASPGEN_PORT=8006
EXPOSE 8006 7000
CMD ["python", "/app/server.py"]
