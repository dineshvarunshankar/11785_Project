"""
Verifies that PointNet++ layers import correctly and run a forward pass on GPU.

The PyTorch port uses pure-PyTorch PointNet++ layers from
Pointnet_Pointnet2_pytorch/models/pointnet2_utils.py — no C++/CUDA compilation
needed (unlike the original TF CGN which required compiling tf_ops).

Run inside the Docker container:
    python p1/verify_pointnet2.py
"""

import sys
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'contact_graspnet_pytorch',
                                'Pointnet_Pointnet2_pytorch'))


def verify_imports():
    from models.pointnet2_utils import (
        PointNetSetAbstractionMsg,
        PointNetSetAbstraction,
        PointNetFeaturePropagation,
    )
    print("PASS  PointNet++ layers imported")
    return PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


def verify_forward_pass(PointNetSetAbstractionMsg, PointNetSetAbstraction,
                        PointNetFeaturePropagation):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Running on: {device}")

    # Small SA-MSG layer: 512 points, two radius scales
    sa = PointNetSetAbstractionMsg(
        npoint=512,
        radius_list=[0.02, 0.04],
        nsample_list=[32, 64],
        in_channel=0,          # 0 extra features (xyz only); layer adds +3 internally
        mlp_list=[[32, 32, 64], [64, 64, 128]],
    ).to(device)

    # Layer expects [B, C, N] (channels first) — it permutes internally to [B, N, C]
    xyz   = torch.randn(2, 3, 2048).to(device)   # B=2, C=3 (xyz), N=2048
    points = None

    new_xyz, new_points = sa(xyz, points)
    assert new_xyz.shape == (2, 3, 512), f"Unexpected shape: {new_xyz.shape}"
    print(f"PASS  SA-MSG forward pass  input={list(xyz.shape)} → output={list(new_xyz.shape)}")


def verify_cuda():
    if torch.cuda.is_available():
        print(f"PASS  CUDA available  ({torch.cuda.get_device_name(0)})")
        print(f"      torch={torch.__version__}  CUDA={torch.version.cuda}")
    else:
        print("WARN  CUDA not available — running on CPU")


if __name__ == "__main__":
    print(f"torch version : {torch.__version__}")
    verify_cuda()
    layers = verify_imports()
    verify_forward_pass(*layers)
    print("\nAll checks passed.")
