# Contact-GraspNet — Team Task Split 

Reproducing Contact-GraspNet (Sundermeyer et al., ICRA 2021) in PyTorch as the midterm baseline.

- CGN original codebase: https://github.com/NVlabs/contact_graspnet
- Codebase: https://github.com/elchun/contact_graspnet_pytorch  
- Paper: https://arxiv.org/abs/2103.14127

Every implementation decision traces back to either the PyTorch repo or the paper.
Nothing gets added or changed without the team agreeing first. We've to make sure that the pyTorch implementation mimics the CGN tensorflow version.

Rules:
- No hardcoded paths; read from `configs/config.yaml`.
- One branch per person: `p1/setup`, `p2/data`, `p3/model`, `p4/training`.
- If the PyTorch port differs from the paper, document it; do not silently change behavior.

---

## P1 — Environment, Inference, SE(3) Pose Construction, Visualization

Own everything needed to run inference end-to-end and render outputs correctly.

Tasks:
- Create a reproducible environment (Docker or equivalent) that everyone can run.
- Compile `pointnet2_ops` (PointNet++ CUDA/C++ extensions) and verify it imports.
- Pin working versions (torch/CUDA/GCC/ninja) and document setup in `docs/setup.md`.
- Download pre-trained checkpoints and run inference on provided test scenes:
  - `python contact_graspnet_pytorch/inference.py --np_path="test_data/*.npy" --local_regions --filter_grasps`
- Implement/verify `contact_to_grasp_pose()` (4-DoF representation to full 6-DoF SE(3) pose) and coordinate conventions used for visualization.
- Build qualitative visualization (Open3D or equivalent):
  - Point cloud + top-k predicted grasps colored by confidence
  - Save example outputs

Deliverables:
- `Dockerfile` (or environment scripts)
- `docs/setup.md`
- Proof of working inference (command + saved output)
- Pose construction module + minimal tests
- Visualization script + example renders

---

## P2 — Dataset, Loader, Point Cloud Preprocessing

Own everything related to data acquisition, formatting, and preprocessing.

Tasks:
- Acquire ACRONYM grasps + ShapeNet meshes, and any provided preprocessed scene/contact data.
- Prefer preprocessed/manifold meshes if available; avoid running slow watertightness pipelines unless required.
- Ensure directory structure matches training code expectations.
- Implement/verify preprocessing used by the repo:
  - Depth back-projection to point cloud (using intrinsics `K`)
  - Farthest Point Sampling (target N = 20,000 points)
  - Centering/normalization conventions
- Implement dataset loader that returns exactly what training expects (shapes + dtypes).
- Provide a small visualization notebook demonstrating correct loading (scene + GT overlay if available).

Deliverables:
- `docs/data.md` (download sources, folder layout, file formats, assumptions)
- Dataset loader + preprocess code
- Data visualization notebook

---

## P3 — Model Heads & Symmetry-Aware Losses

Own the prediction heads and losses (including the gripper symmetry issue).

Tasks:
- Implement/verify prediction heads per point:
  - Confidence
  - Approach vector
  - Baseline vector
  - Gripper width
- Implement the symmetry-aware loss for parallel-jaw gripper ambiguity (180° rotational symmetry):
  - Use a symmetric objective (e.g., take the min loss between target and flipped target) rather than a single-direction cosine/L2 loss.
- Implement loss unit tests (especially symmetry edge cases).
- Compare and document loss weights vs paper and repo config.

Deliverables:
- Heads module
- Losses module (symmetry-aware) + unit tests
- `docs/architecture.md` section: heads + losses mapping to paper, plus any config mismatches

---

## P4 — Training Loop, Postprocessing (NMS), Evaluation

Own the training/eval pipeline and the final numbers.

Tasks:
- Implement/verify training loop and logging:
  - Log each loss component separately (confidence, approach, baseline, width).
  - Checkpointing and reproducible configs.
- Implement grasp filtering / NMS for 3D poses:
  - Use pose similarity (contact-point distance + angular thresholds), not only Euclidean distance.
- Implement evaluation consistent with the paper’s protocol and this repo’s splits:
  - Seen/unseen split handling if applicable
  - Metrics export to CSV
- Produce baseline results and comparison table vs paper (include thresholds/settings used).

Deliverables:
- Training script + logs
- `postprocess/nms.py` (or equivalent)
- `evaluate.py`
- `results/baseline_metrics.csv`
