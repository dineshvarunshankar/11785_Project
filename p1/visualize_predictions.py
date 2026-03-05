"""
Load inference results from results/predictions_9.npz and render
the top-K predicted grasps on the point cloud, saving a PNG.

Usage:
    python p1/visualize_predictions.py
    python p1/visualize_predictions.py --npz results/predictions_9.npz --topk 50 --out results/viz_9.png
"""

import argparse
import numpy as np
import sys
import os

# Make p1/ importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p1.visualization import visualize_grasps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",  default="results/predictions_9.npz",
                        help="Path to predictions .npz file")
    parser.add_argument("--topk", type=int, default=50,
                        help="Number of top grasps to render per segment")
    parser.add_argument("--out",  default=None,
                        help="Output PNG path (default: same dir as npz, viz_<name>.png)")
    args = parser.parse_args()

    if not os.path.exists(args.npz):
        print(f"File not found: {args.npz}")
        sys.exit(1)

    data = np.load(args.npz, allow_pickle=True)
    pc_full         = data["pc_full"]           # Nx3
    pred_grasps_cam = data["pred_grasps_cam"].item()  # dict: seg_id to Mx4x4
    scores          = data["scores"].item()            # dict: seg_id to M
    pc_colors       = data.get("pc_colors", None)
    if pc_colors is not None:
        pc_colors = pc_colors

    print(f"Loaded: {args.npz}")
    print(f"  Point cloud: {pc_full.shape}")
    for k, v in pred_grasps_cam.items():
        if len(v) == 0:
            print(f"  Segment {k}: 0 grasps (skipped)")
            continue
        print(f"  Segment {k}: {len(v)} grasps, "
              f"scores [{scores[k].min():.3f}, {scores[k].max():.3f}]")

    # Default output path: results/viz_9.png
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.npz))[0]  # predictions_9
        name = base.replace("predictions_", "viz_")              # viz_9
        args.out = os.path.join(os.path.dirname(args.npz), name + ".png")

    visualize_grasps(
        pc_full,
        pred_grasps_cam,
        scores,
        pc_colors=pc_colors,
        topk=args.topk,
        save_path=args.out,
        show=True,
    )
    print(f"Done. Render saved to: {args.out}")


if __name__ == "__main__":
    main()
