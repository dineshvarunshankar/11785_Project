# ARG lets you switch base image at build time:
#   GPU (default): docker build .
#   CPU only:      docker build --build-arg BASE=python:3.10-slim .
ARG BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE}

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ─────────────────────────────────────────────────────────────────────────────
# System dependencies
# build-essential / ninja-build / cmake : compile any C++ extensions if needed
# git / wget                            : fetch assets at build time
# libgl1 / libglib2.0                   : required by OpenCV and Open3D
# libegl1-mesa-dev / libgles2-mesa-dev  : EGL OpenGL (pyrender headless rendering)
# libosmesa6                            : OSMesa software rendering fallback
# libx11-6 / libx11-xcb1 / libxcb1     : X11 client libs for Open3D GUI via X11 forwarding
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch
# Default: cu124 build (CUDA 12.4 — portable, covers sm_50–sm_90: V100, A100, RTX 30xx/40xx)
# CPU build: docker build --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cpu .
# ─────────────────────────────────────────────────────────────────────────────
# Blackwell (sm_120, RTX 50xx): use make build-blackwell → passes cu128 + torch 2.7.0
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    --index-url ${PYTORCH_INDEX}

# ─────────────────────────────────────────────────────────────────────────────
# Python dependencies
# numpy / scipy        : array math, point cloud operations
# trimesh              : 3D mesh loading (gripper geometry)
# pyyaml               : reading configs/config.yaml
# tqdm                 : progress bars
# pillow               : image I/O
# matplotlib           : debug image saving
# open3d               : 3D point cloud visualization
# plotly               : interactive scene plots
# transforms3d         : SE(3)/SO(3) rotation helpers
# opencv-python        : depth image processing
# h5py                 : HDF5 dataset files (ACRONYM)
# pyrender             : headless mesh rendering
# ─────────────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "numpy>=1.24,<2" \
    "scipy>=1.10" \
    "trimesh>=3.20" \
    "pyyaml>=6.0" \
    "tqdm>=4.64" \
    "pillow>=9.0" \
    "matplotlib>=3.7" \
    "open3d>=0.17.0" \
    "plotly>=5.18.0" \
    "transforms3d>=0.4.1" \
    "opencv-python>=4.8.0" \
    "h5py>=3.8.0" \
    "pyrender>=0.1.45"

# ─────────────────────────────────────────────────────────────────────────────
# Working directory — all team modules (p1/ p2/ p3/ p4/) live here
# ─────────────────────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace

# Install contact_graspnet_pytorch (includes Pointnet_Pointnet2_pytorch/provider.py)
# Patch 1 — checkpoints.py: PyTorch 2.6+ changed torch.load default to weights_only=True,
#            but the checkpoint contains numpy objects; must use weights_only=False.
# Patch 2 — contact_graspnet.py: torch.cross without dim is deprecated in PyTorch 2.x;
#            replace with torch.linalg.cross which defaults to dim=-1 (correct for Nx3 vectors).
RUN git clone https://github.com/elchun/contact_graspnet_pytorch.git /opt/cgn \
    && sed -i 's/torch.load(filename)/torch.load(filename, weights_only=False)/g' \
       /opt/cgn/contact_graspnet_pytorch/checkpoints.py \
    && sed -i 's/torch\.cross(/torch.linalg.cross(/g' \
       /opt/cgn/contact_graspnet_pytorch/contact_graspnet.py \
    && pip install --no-cache-dir -e /opt/cgn
ENV PYTHONPATH="/opt/cgn/Pointnet_Pointnet2_pytorch:/opt/cgn/contact_graspnet_pytorch"

# Use EGL for headless OpenGL rendering (required by pyrender inside Docker)
ENV PYOPENGL_PLATFORM=egl
# Avoid matplotlib permission errors when running as non-root (--user flag)
ENV MPLCONFIGDIR=/tmp

# Default: drop into bash
CMD ["/bin/bash"]