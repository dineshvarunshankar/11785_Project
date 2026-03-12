"""Open3D visualization for one preprocessed CGN scene.

Behavior mirrors `p1/visualization.py`:
- optional offscreen image save
- optional interactive Open3D window
"""

import argparse
from pathlib import Path

import numpy as np


def world_to_camera(points_world: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """Transform world-frame points to camera frame."""
    rot = camera_pose[:3, :3]
    trans = camera_pose[:3, 3]
    return points_world @ rot.T + trans[None, :]


def world_vec_to_camera(vec_world: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """Rotate world-frame direction vectors into camera frame."""
    rot = camera_pose[:3, :3]
    return vec_world @ rot.T


def choose_indices(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return rng.choice(n, size=max_points, replace=False)


def make_lineset(starts: np.ndarray, dirs: np.ndarray, length: float, color: np.ndarray, o3d):
    """Create an Open3D LineSet from start points and direction vectors."""
    ends = starts + dirs * length
    points = np.vstack([starts, ends])
    n = starts.shape[0]
    lines = np.column_stack([np.arange(n), np.arange(n) + n]).astype(np.int32)
    colors = np.repeat(color[None, :], n, axis=0)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def main() -> None:
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "open3d is required for this visualization. Install it with: pip install open3d"
        ) from exc

    parser = argparse.ArgumentParser(description="Visualize preprocessed CGN scene with Open3D.")
    parser.add_argument("--scene-path", type=Path, required=True, help="Path to scene .npz file.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output PNG path for offscreen render.",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Show interactive Open3D window (default).",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable interactive window.",
    )
    parser.add_argument("--max-scene-points", type=int, default=8000, help="Max scene points to render.")
    parser.add_argument("--max-contact-points", type=int, default=3000, help="Max GT contacts to render.")
    parser.add_argument("--num-arrows", type=int, default=200, help="Number of direction arrows.")
    parser.add_argument("--arrow-length", type=float, default=0.03, help="Arrow line length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")
    parser.set_defaults(show=True)
    args = parser.parse_args()

    if not args.scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {args.scene_path}")

    scene = dict(np.load(args.scene_path, allow_pickle=False))
    required = {
        "pc_cam",
        "camera_pose",
        "pos_contact_points",
        "pos_contact_dirs",
        "pos_approach_dirs",
    }
    missing = required - set(scene.keys())
    if missing:
        raise KeyError(f"Missing keys in scene file: {sorted(missing)}")

    pc_cam = scene["pc_cam"].astype(np.float32)
    camera_pose = scene["camera_pose"].astype(np.float32)
    pos_contact_points_w = scene["pos_contact_points"].astype(np.float32)
    pos_contact_dirs_w = scene["pos_contact_dirs"].astype(np.float32)
    pos_approach_dirs_w = scene["pos_approach_dirs"].astype(np.float32)

    pos_contact_points_c = world_to_camera(pos_contact_points_w, camera_pose)
    pos_contact_dirs_c = world_vec_to_camera(pos_contact_dirs_w, camera_pose)
    pos_approach_dirs_c = world_vec_to_camera(pos_approach_dirs_w, camera_pose)

    rng = np.random.default_rng(args.seed)
    scene_idx = choose_indices(pc_cam.shape[0], args.max_scene_points, rng)
    contact_idx = choose_indices(pos_contact_points_c.shape[0], args.max_contact_points, rng)
    arrow_pool = contact_idx
    arrow_idx = choose_indices(arrow_pool.shape[0], min(args.num_arrows, arrow_pool.shape[0]), rng)
    arrow_contact_idx = arrow_pool[arrow_idx]

    scene_pc = o3d.geometry.PointCloud()
    scene_pc.points = o3d.utility.Vector3dVector(pc_cam[scene_idx])
    scene_pc.paint_uniform_color([0.75, 0.75, 0.75])

    gt_contacts_pc = o3d.geometry.PointCloud()
    gt_contacts_pc.points = o3d.utility.Vector3dVector(pos_contact_points_c[contact_idx])
    gt_contacts_pc.paint_uniform_color([1.0, 0.0, 0.0])

    contact_dir_lines = make_lineset(
        starts=pos_contact_points_c[arrow_contact_idx],
        dirs=pos_contact_dirs_c[arrow_contact_idx],
        length=args.arrow_length,
        color=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        o3d=o3d,
    )
    approach_dir_lines = make_lineset(
        starts=pos_contact_points_c[arrow_contact_idx],
        dirs=pos_approach_dirs_c[arrow_contact_idx],
        length=args.arrow_length,
        color=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        o3d=o3d,
    )
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.0, 0.0, 0.0])

    print(f"Scene file: {args.scene_path}")
    print(f"Rendered points: {len(scene_idx)} scene, {len(contact_idx)} GT contacts, {len(arrow_contact_idx)} arrows")
    print("Color legend: gray=scene, red=GT contacts, blue=contact dirs, green=approach dirs")

    geometries = [scene_pc, gt_contacts_pc, contact_dir_lines, approach_dir_lines, frame]

    # Offscreen save path, similar to p1/visualization.py.
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 720)

        mat_pc = o3d.visualization.rendering.MaterialRecord()
        mat_pc.shader = "defaultUnlit"

        mat_line = o3d.visualization.rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 2.0

        for idx, geom in enumerate(geometries):
            mat = mat_line if isinstance(geom, o3d.geometry.LineSet) else mat_pc
            renderer.scene.add_geometry(f"geom_{idx}", geom, mat)

        renderer.scene.set_background([1, 1, 1, 1])
        bounds = renderer.scene.bounding_box
        center = bounds.get_center()
        extent = np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound())
        eye = center + np.array([0, 0, -extent], dtype=np.float32)
        up = np.array([0, -1, 0], dtype=np.float32)
        renderer.setup_camera(60.0, center.astype(np.float32), eye, up)

        image = renderer.render_to_image()
        image_np = np.asarray(image)
        from PIL import Image as PILImage

        PILImage.fromarray(image_np).save(args.out)
        print(f"Saved render to {args.out}")

    if args.show:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"CGN Overlay - {args.scene_path.name}",
            width=1280,
            height=900,
        )


if __name__ == "__main__":
    main()
