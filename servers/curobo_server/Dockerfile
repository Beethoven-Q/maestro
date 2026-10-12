# cuRobo motion-planning service with an nvblox collision world: port 8004.
# Built by compose.yaml at the repository root.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git cmake build-essential \
    libbenchmark-dev libgl1 libglib2.0-0 libgomp1 pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

# Python environment validated with this service.
RUN uv venv --python 3.9 /opt/venv
COPY servers/curobo_server/requirements.txt /tmp/requirements.txt
RUN uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 -r /tmp/requirements.txt

# There is no GPU during the build, so every CUDA build names its architectures.
ENV CUDAARCHS="75;80;86;89;90" TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

# nvblox links into PyTorch, which uses the pre-C++11 ABI, so nvblox and its static SQLite,
# glog and gflags are all built with that ABI, as in the tested build.
ENV OLD_ABI=-D_GLIBCXX_USE_CXX11_ABI=0
RUN curl -fsSL -o /tmp/sqlite.tgz https://www.sqlite.org/2022/sqlite-autoconf-3390400.tar.gz \
    && echo "f31d445b48e67e284cf206717cc170ab63cbe4fd7f79a82793b772285e78fdbb  /tmp/sqlite.tgz" | sha256sum -c - \
    && tar xzf /tmp/sqlite.tgz -C /tmp && cd /tmp/sqlite-autoconf-3390400 \
    && CFLAGS="-O2 -fPIC" ./configure --prefix=/opt/deps/sqlite --disable-shared && make -j "$(nproc)" install \
    && git clone --depth 1 --branch v2.2.2 https://github.com/gflags/gflags.git /tmp/gflags \
    && cmake -S /tmp/gflags -B /tmp/gflags/build -DCMAKE_INSTALL_PREFIX=/opt/deps/gflags -DBUILD_SHARED_LIBS=OFF \
       -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_TESTING=OFF -DCMAKE_CXX_FLAGS=$OLD_ABI \
    && cmake --build /tmp/gflags/build -j "$(nproc)" --target install \
    && git clone --depth 1 --branch v0.6.0 https://github.com/google/glog.git /tmp/glog \
    && cmake -S /tmp/glog -B /tmp/glog/build -DCMAKE_INSTALL_PREFIX=/opt/deps/glog -DBUILD_SHARED_LIBS=OFF \
       -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DBUILD_TESTING=OFF -DCMAKE_CXX_FLAGS=$OLD_ABI \
    && cmake --build /tmp/glog/build -j "$(nproc)" --target install \
    && rm -rf /tmp/sqlite* /tmp/gflags /tmp/glog
ARG NVBLOX_COMMIT=7bda93e993594ca69c479dde64213ac35835eb84
ARG NVBLOX_TORCH_COMMIT=e121121c100669848c75e1be97df94a91307e23f
RUN git clone https://github.com/valtsblukis/nvblox.git /opt/nvblox \
    && git -C /opt/nvblox checkout $NVBLOX_COMMIT \
    && cmake -S /opt/nvblox/nvblox -B /opt/nvblox/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
       -DPRE_CXX11_ABI_LINKABLE=ON -DBUILD_REDISTRIBUTABLE=ON -DCMAKE_CXX_FLAGS=$OLD_ABI -DCMAKE_CUDA_FLAGS=$OLD_ABI \
       -DSQLITE3_BASE_PATH=/opt/deps/sqlite -DGLOG_BASE_PATH=/opt/deps/glog -DGFLAGS_BASE_PATH=/opt/deps/gflags \
    && cmake --build /opt/nvblox/build -j "$(nproc)" && cmake --install /opt/nvblox/build \
    && ldconfig && rm -rf /opt/nvblox/build
RUN git clone https://github.com/NVlabs/nvblox_torch.git /opt/nvblox_torch \
    && git -C /opt/nvblox_torch checkout $NVBLOX_TORCH_COMMIT \
    && cd /opt/nvblox_torch && PKG_CONFIG_PATH=/opt/deps/glog/lib/pkgconfig LIBRARY_PATH=/opt/deps/glog/lib \
       sh install.sh "$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" \
    && uv pip install --no-deps --no-build-isolation -e /opt/nvblox_torch

# cuRobo at the tested commit with Maestro's overlay: the CuroboPlanner used by the server and
# the Franka + Robotiq 2F-85 robot model.
ARG CUROBO_COMMIT=ebb71702f3f70e767f40fd8e050674af0288abe8
COPY servers/curobo_server/curobo_overlay /tmp/curobo_overlay
RUN git clone https://github.com/NVlabs/curobo.git /opt/curobo \
    && git -C /opt/curobo checkout $CUROBO_COMMIT \
    && cp -r /tmp/curobo_overlay/. /opt/curobo/src/curobo/ \
    && uv pip install --no-deps --no-build-isolation -e /opt/curobo

COPY servers/curobo_server/server.py /app/server.py
ENV CUROBO_PORT=8004 CUROBO_ROBOT_TYPE=franka_robotiq
EXPOSE 8004
CMD ["python", "/app/server.py"]
