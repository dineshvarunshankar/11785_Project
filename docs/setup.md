# Getting Started

This project runs inside a **Docker container** — a self-contained environment that
installs all dependencies automatically. You don't need to install Python, PyTorch,
or any library manually.

---

## Step 1 — Install Docker

Choose your OS:

**Linux**
```bash
# Install Docker
sudo apt-get install docker.io
# Install NVIDIA Container Toolkit (needed for GPU)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

**Windows**
1. Enable WSL2: open PowerShell as Administrator and run `wsl --install`, then restart
2. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
3. Install the latest [NVIDIA driver for Windows](https://www.nvidia.com/Download/index.aspx) (GPU support comes automatically via WSL2 — no extra toolkit needed)

**Mac**
1. Install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. Note: Mac has no NVIDIA GPU, so the CPU build is used (see Step 2)

---

## Step 2 — Clone and Build

```bash
git clone <repo-url>
cd 11785_Project
```

**With a GPU (Linux / Windows):**
```bash
make build
```

**Without a GPU (Mac or CPU-only machine):**
```bash
make build-cpu
```

This step downloads the base image and installs all dependencies. It takes
**5–10 minutes** the first time. Subsequent builds are fast.

---

## Step 3 — Run Inference

```bash
make inference
```

This runs the grasp prediction model on all scenes in `test_data/` and saves results to `results/`:

```
results/
  predictions_0.npz     ← raw grasp poses and confidence scores
  renders/0.png         ← visualization of top predicted grasps
```

---

## Step 4 — Verify Everything Works

```bash
make verify           # checks GPU + PointNet++ layers
make test             # runs pose construction unit tests
```

Both should print `All checks passed` / `All tests passed`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` | Install Docker (Step 1) |
| `nvidia-smi` not found | Install NVIDIA driver |
| `Error: no CUDA device` | Use `make build-cpu` instead |
| `permission denied` on `docker` | Run `sudo usermod -aG docker $USER` then log out and back in |
| Port/container already in use | Run `docker rm cgn_run` then retry |

---

## Configuring the Model

All parameters live in `configs/config.yaml`. You can edit this file on your
host machine — changes are reflected immediately without rebuilding the container.

Key settings:

| Setting | Default | What it does |
|---|---|---|
| `TEST.first_thres` | `0.23` | Minimum confidence to keep a grasp |
| `TEST.num_samples` | `200` | Maximum number of grasps per scene |
