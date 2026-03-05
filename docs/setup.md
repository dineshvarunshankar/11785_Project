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

## How P2, P3, P4 benefit from P1

- **P2** can use the Docker environment to run preprocessing scripts reproducibly,
  and use `p1/visualization.py` to visually verify that loaded point clouds look correct.

- **P3** can use `p1/pose_utils.py`'s `contact_to_grasp_pose()` to convert their
  predicted contact points and directions into 4×4 SE(3) pose matrices — the same
  format the visualization and evaluation code expects.

- **P4** can use `make inference` to get baseline results from the pre-trained checkpoint,
  and `make visualize` to see what good inference outputs look like before training
  their own model.

---

## Prerequisites

You need:
- **Docker** installed on your machine
- **NVIDIA GPU** (recommended) — or use the CPU build for testing only
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

First, set up the lightweight visualization environment (one-time):
```bash
make setup-venv
```

Then visualize any result:
```bash
make visualize
```

This opens an **interactive Open3D window** showing:
- The 3D point cloud of the scene (colored by RGB)
- The top-50 predicted grasp poses drawn as wireframe grippers
- Grasps colored by object segment (different color per object)

You can rotate, zoom, and pan the scene in the window.

To visualize a different scene or change the number of grasps shown:
```bash
cgn_visualize_venv/bin/python p1/visualize_predictions.py \
  --npz results/predictions_3.npz \
  --topk 100
```

---

## Step 5 — Verify everything works

```bash
make test      # runs pose construction unit tests (pure math, no GPU needed)
make verify    # checks that PointNet++ loads and runs a forward pass
```

Both should complete without errors.

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
