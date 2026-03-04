# ARG lets you switch base image at build time:
#   GPU (default): docker build .
#   CPU only:      docker build --build-arg BASE=python:3.10-slim .
ARG BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE}

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ─────────────────────────────────────────────────────────────────────────────
# System dependencies
# build-essential / ninja-build / cmake : compile any C++ extensions
# git / wget                            : fetch assets
# libgl1 / libglib2.0                   : required by OpenCV and Open3D
# libegl1-mesa-dev / libgles2-mesa-dev  : headless OpenGL rendering (pyrender)
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
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch
# Default: GPU build (CUDA 12.1)
# CPU build: docker build --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cpu .
# ─────────────────────────────────────────────────────────────────────────────
# Default: cu124 (portable — covers sm_50 through sm_90: V100, A100, RTX 30xx/40xx)
# Blackwell (sm_120, RTX 50xx): use make build-blackwell → passes cu128
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

# Install contact_graspnet_pytorch and install (includes Pointnet_Pointnet2_pytorch/provider.py)
# Patch 1 — checkpoints.py: PyTorch 2.6+ changed torch.load default to weights_only=True,
#            but the checkpoint contains numpy objects; must use weights_only=False.
# Patch 2 — inference.py: comment out Open3D visualization calls (show_image / visualize_grasps)
#            which crash in headless Docker. Results are already saved to results/ before these lines.
RUN git clone https://github.com/elchun/contact_graspnet_pytorch.git /opt/cgn \
    && sed -i 's/torch.load(filename)/torch.load(filename, weights_only=False)/g' \
       /opt/cgn/contact_graspnet_pytorch/checkpoints.py \
    && sed -i 's/^\(\s*show_image(.*)\)$/# \1/' \
       /opt/cgn/contact_graspnet_pytorch/inference.py \
    && sed -i 's/^\(\s*visualize_grasps(.*)\)$/# \1/' \
       /opt/cgn/contact_graspnet_pytorch/inference.py \
    && pip install --no-cache-dir -e /opt/cgn
ENV PYTHONPATH="/opt/cgn/Pointnet_Pointnet2_pytorch:/opt/cgn/contact_graspnet_pytorch"

# Use EGL for headless OpenGL rendering (required by pyrender inside Docker)
ENV PYOPENGL_PLATFORM=egl
# Use CPU/OSMesa software rendering for Open3D headless mode
ENV OPEN3D_CPU_RENDERING=true

# Default: drop into bash
CMD ["/bin/bash"]