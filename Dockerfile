FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /usr/local/bin/

ARG CUDA_ARCH=89
ARG CERES_VERSION=2.1.0
ARG COLMAP_VERSION=3.12.3

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    Ceres_DIR=/usr/local/lib/cmake/Ceres \
    colmap_DIR=/usr/local/share/colmap \
    CMAKE_PREFIX_PATH=/usr/local \
    PATH=/usr/local/cuda/bin:$PATH \
    UV_LINK_MODE=copy \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    zsh git zip unzip curl wget ca-certificates locales sudo gosu \
    openssh-client \
    build-essential cmake ninja-build pkg-config \
    tmux vim less tree \
    ffmpeg libgl1 libglib2.0-0 \
    libeigen3-dev libsuitesparse-dev \
    libgoogle-glog-dev libgflags-dev \
    libboost-all-dev libsqlite3-dev \
    libflann-dev liblz4-dev \
    libcgal-dev libmetis-dev libfreeimage-dev \
    mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev libglx-dev \
    libopengl-dev libglew-dev \
&& rm -rf /var/lib/apt/lists/* \
&& (locale-gen en_US.UTF-8 >/dev/null 2>&1 || true)

# Build & install Ceres
RUN git clone https://github.com/ceres-solver/ceres-solver.git /tmp/ceres-solver \
&& cd /tmp/ceres-solver \
&& git checkout ${CERES_VERSION} \
&& cmake -S . -B build -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCERES_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
&& cmake --build build -j"$(nproc)" \
&& cmake --install build \
&& rm -rf /tmp/ceres-solver \
&& echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf \
&& ldconfig

# Build & install COLMAP (headless)
RUN git clone https://github.com/colmap/colmap.git /tmp/colmap \
&& cd /tmp/colmap \
&& git checkout ${COLMAP_VERSION} \
&& cmake -S . -B build -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    -DGUI_ENABLED=OFF \
&& cmake --build build -j"$(nproc)" \
&& cmake --install build \
&& rm -rf /tmp/colmap

WORKDIR /workspace

CMD ["sleep", "infinity"]
