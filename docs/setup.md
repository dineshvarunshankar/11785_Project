# Setup Guide

## What is this project?

We are reproducing **Contact-GraspNet (CGN)** — a model that looks at a 3D scene
(captured by a depth camera) and predicts where a robot hand should grab objects.

Given a depth image of a cluttered table, CGN:
1. Converts the depth image into a 3D point cloud
2. Runs a PointNet++ backbone to process the point cloud
3. Predicts grasp poses — where to place the gripper and how wide to open it
4. Filters and ranks the predictions by confidence score

The original CGN was written in TensorFlow. We are reproducing it in PyTorch using
[this port](https://github.com/elchun/contact_graspnet_pytorch).

---

## How the team is split

| Person | Owns |
|---|---|
| **P1 (this repo)** | Docker environment, running inference, pose math, visualization |
| **P2** | Dataset loading and point cloud preprocessing |
| **P3** | Model prediction heads and loss functions |
| **P4** | Training loop, NMS postprocessing, evaluation metrics |

**P1's job** is to make sure everyone else can get the code running without fighting
their environment. P1 also owns the math that converts the network's raw outputs
into full 3D grasp poses, and the visualization that lets you see what the model predicted.

---

## Prerequisites

You need:
- **Docker** installed on your machine
- **NVIDIA GPU** (recommended) — or use the CPU build for testing only
- **NVIDIA Container Toolkit** (for GPU)
- **Git**

---

## Step 1 — Clone the repo

```bash
git clone --recurse-submodules <repo-url>
cd 11785_Project
```

The `--recurse-submodules` flag fetches the reference PyTorch port
(`contact_graspnet_pytorch/`) which contains the pre-trained checkpoints
and test data. If you forgot this flag:

```bash
git submodule update --init --recursive
```

---

## Step 2 — Build the Docker image

The Docker image bundles Python, PyTorch, CUDA, and all dependencies.
You only need to do this once (or after the Dockerfile changes).

**Standard GPU build** (works on most NVIDIA GPUs — V100, A100, RTX 30xx/40xx):
```bash
make build
```

**Blackwell GPU build** (RTX 5060/5070/5080/5090 — newest 2024/2025 GPUs):
```bash
make build-blackwell
```

**CPU-only build** (Mac, or if you have no GPU):
```bash
make build-cpu
```

This downloads about 3–5 GB and takes 5–10 minutes the first time.

---

## Step 3 — Run inference

```bash
make inference
```

This runs the pre-trained grasp prediction model on all scenes in `test_data/`
and saves results to `results/`:

```
results/
  predictions_0.npz    ← grasp poses + confidence scores for scene 0
  predictions_1.npz
  ...
```

Each `.npz` file contains:
- `pc_full` — the full scene point cloud (Nx3)
- `pred_grasps_cam` — predicted 4×4 grasp poses per object segment
- `scores` — confidence score for each predicted grasp
- `contact_pts` — the contact point on the object surface
- `pc_colors` — RGB colors of the point cloud

---

## Step 4 — Visualize results

```bash
make visualize
```

This opens an **interactive Open3D window** showing:
- The 3D point cloud of the scene (colored by RGB)
- The top-50 predicted grasp poses drawn as wireframe grippers
- Grasps colored by object segment (different color per object)

You can rotate, zoom, and pan the scene in the window.

Visualization runs **inside the Docker container** using X11 forwarding — the window
appears on your screen but renders with all the same dependencies as inference. No
separate environment needed.

> **Linux only**: X11 forwarding requires a running X server (standard on Ubuntu desktop).
> The `make visualize` command runs `xhost +local:docker` automatically to grant access.

To visualize a different scene or change the number of grasps shown:
```bash
docker run --rm -it --gpus all \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(PWD):/workspace \
  cgn_pytorch python p1/visualize_predictions.py \
  --npz results/predictions_3.npz \
  --topk 100
```

---

## Step 5 — Verify everything works

```bash
make test      # runs pose construction unit tests 
make verify    # checks that PointNet++ loads and runs a forward pass
```

Both should complete without errors.

> **Note on `pointnet2_ops` (compiled CUDA extensions):**
> The original TensorFlow CGN required manually compiling C++ CUDA ops (`pointnet2_ops`),
> which needed matching GCC, CUDA, and ninja versions — a common source of env issues.
> The PyTorch port we use (`contact_graspnet_pytorch`) **does not require this** — it uses
> a pure-Python/PyTorch reimplementation of PointNet++ (`Pointnet_Pointnet2_pytorch/models/pointnet2_utils.py`)
> that runs on GPU without any compilation step.
> `make verify` confirms this pure-Python PointNet++ works correctly on your GPU.

---

## Project structure

```
11785_Project/
├── Dockerfile              ← Everything needed to build the containerized environment
├── Makefile                ← All commands: build, inference, visualize, test
├── contact_graspnet_pytorch/  ← Git submodule — the PyTorch port (checkpoints + test data)
├── configs/
│   └── config.yaml         ← Model and training hyperparameters (edit freely)
├── checkpoints/
│   └── contact_graspnet/   ← Pre-trained model weights (symlink into submodule)
├── test_data/              ← Symlink → contact_graspnet_pytorch/test_data/
├── results/                ← Inference outputs saved here (gitignored)
├── p1/
│   ├── pose_utils.py       ← contact_to_grasp_pose(): network output → 4×4 SE(3) pose
│   ├── visualization.py    ← visualize_grasps(): point cloud + grasp wireframes
│   └── visualize_predictions.py  ← loads results/*.npz and calls visualization.py
└── docs/
    └── setup.md            ← This file
```

---

## Understanding `configs/config.yaml`

The config file controls everything about the model — data loading, architecture,
training, and inference. It has four sections:

| Section | What it controls |
|---|---|
| `DATA` | Point cloud size (`num_point: 2048`), gripper width, contact label bins, data augmentation |
| `MODEL` | PointNet++ architecture — SA-MSG layer radii, MLP sizes, which output heads are active |
| `OPTIMIZER` | Training hyperparameters — `batch_size`, `learning_rate`, `max_epoch`, LR decay schedule |
| `TEST` | Inference filtering — `first_thres` / `second_thres` confidence cutoffs, max samples |

**Values you'll most likely want to change:**

```yaml
OPTIMIZER:
  batch_size: 3       # reduce if you run out of GPU memory
  learning_rate: 0.001
  max_epoch: 16       # increase for full training runs

TEST:
  first_thres: 0.23   # lower → more grasps shown; raise → fewer but higher confidence
  second_thres: 0.19
```

**Where it's loaded:** `contact_graspnet_pytorch/contact_graspnet_pytorch/config_utils.py`
reads this file at startup. All other modules receive the config as a dict.
You do not need to touch this file to run inference with the pre-trained checkpoint.

---

## Understanding `pose_utils.py`

The network predicts grasps in a compact 4-DoF format:
- A **contact point** on the object surface
- A **baseline direction** (finger-to-finger axis)
- An **approach direction** (how the gripper approaches the object)
- A **width** (how far apart the fingers should be)

`contact_to_grasp_pose()` converts these into a standard **4×4 SE(3) matrix** that
encodes the full 6-DoF pose (position + orientation) of the gripper in camera coordinates.
This is what P3/P4 will use to interface with the rest of the pipeline.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` | Install Docker (see prerequisites) |
| `permission denied` on `docker` | Run `sudo usermod -aG docker $USER` then log out/in |
| GPU not detected in container | Make sure NVIDIA Container Toolkit is installed |
| `results/` permission error | Run `sudo chown -R $USER:$USER results/` |
| Old GPU (sm_120 warning) | Use `make build-blackwell` instead of `make build` |
| Container name already in use | Run `make down` then retry |

---

## How P2, P3, P4 use P1's work

### P2 — Dataset & Preprocessing

P2 builds the data loader. They can validate their point cloud preprocessing is correct by
visually comparing it against P1's inference outputs:

```python
from p1.visualization import visualize_grasps
import numpy as np

# Load P1's inference output as a reference
ref = np.load("results/predictions_9.npz", allow_pickle=True)

# Load your own preprocessed point cloud
my_pc = your_loader.load("test_data/9.npy")  # should look the same as ref["pc_full"]

# Visualize your pc against P1's predicted grasps — if the pc is wrong, grasps will float in air
visualize_grasps(my_pc, ref["pred_grasps_cam"].item(), ref["scores"].item())
```

P2 also uses the same Docker environment (`make build`) so preprocessing runs in the
exact same Python/CUDA stack as inference.

---

### P3 — Model Heads & Losses

P3 implements the prediction heads. The network outputs raw 4-DoF vectors per point.
P3 uses `contact_to_grasp_pose()` to convert their model's outputs into SE(3) matrices
for visualization and loss computation:

```python
from p1.pose_utils import contact_to_grasp_pose

# Your model's raw outputs
contact_pts   = model_output["contact_pts"]    # Nx3
base_dirs     = model_output["base_dirs"]      # Nx3
approach_dirs = model_output["approach_dirs"]  # Nx3
widths        = model_output["widths"]         # N

# Convert to Nx4x4 SE(3) poses — ready for visualization or ADDS loss
grasp_poses = contact_to_grasp_pose(contact_pts, base_dirs, approach_dirs, widths)
```

P3 can then compare their model's grasp poses against P1's pre-trained baseline using
`make visualize` to see if the predictions look physically reasonable.

---

### P4 — Training Loop & Evaluation

P4 trains the model and runs evaluation. Use P1's saved baseline predictions
as the reference to beat:

```python
# P1's pre-trained baseline (already in results/)
baseline = np.load("results/predictions_9.npz", allow_pickle=True)
baseline_scores = baseline["scores"].item()

# After P4 trains model, run  visualizer on their outputs
python p1/visualize_predictions.py --npz your_results/predictions_9.npz
```

P4 also uses `make inference` to regenerate the baseline at any time with the
pre-trained checkpoint, providing a stable reference point for the comparison table
in the final report.
