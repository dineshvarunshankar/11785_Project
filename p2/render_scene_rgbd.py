"""Visualize preprocessed CGN scenes in world frame with interactive Plotly viewer.

Two modes:
  --from-preprocessed (default): loads pc_cam + camera_pose from the stored
      .npz and transforms to world frame.  No PyRender needed.
  --render-new: renders a fresh view via PyRender (random camera).

With --overlay-contacts, GT grasp contact points are shown coloured by
proximity to the visible surface.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import yaml
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "contact_graspnet_pytorch"))


def get_valid_scene_contacts(data_path, scene_contacts_path):
    """Return (scene_contacts_dir, valid_ids) mirroring the dataloader logic."""
    scene_contacts_dir = os.path.join(data_path, scene_contacts_path)
    corrupt_list = [
        "001780", "002100", "002486", "004861", "004866", "007176", "008283",
        "005723", "001993", "001165", "006828", "007238", "008980", "005623",
        "001993", "008509", "007996", "009296",
    ]
    paths = sorted(glob.glob(os.path.join(scene_contacts_dir, "*")))
    valid = []
    for p in paths:
        cid = os.path.basename(p).split(".")[0]
        if cid not in corrupt_list:
            valid.append(cid)
    return scene_contacts_dir, valid


# ---------------------------------------------------------------------------
# Plotly interactive viewer
# ---------------------------------------------------------------------------

def show_interactive_plotly(pc_xyz, pc_rgb=None,
                           cam_position=None, cam_forward=None,
                           scene_info=None, show_contacts=True,
                           max_render_pts=30000, max_contact_pts=3000,
                           seed=42, title="ACRONYM Scene", out_html=None):
    """Open an interactive 3D scatter plot in the browser using Plotly.

    All geometry is expected in world frame.
    """
    import plotly.graph_objects as go

    rng = np.random.default_rng(seed)
    traces = []

    # --- Point cloud ---
    if pc_xyz.shape[0] > max_render_pts:
        idx = rng.choice(pc_xyz.shape[0], max_render_pts, replace=False)
    else:
        idx = np.arange(pc_xyz.shape[0])
    pts = pc_xyz[idx, :3]

    if pc_rgb is not None and len(pc_rgb) == len(pc_xyz):
        rgb = pc_rgb[idx]
        colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    else:
        colors = "gray"

    traces.append(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=colors, opacity=0.8),
        name=f"Scene PC ({len(idx)} pts)",
    ))

    # --- GT contact points, coloured by proximity to visible surface ---
    if show_contacts and scene_info is not None:
        from scipy.spatial import cKDTree
        cp = scene_info["scene_contact_points"].reshape(-1, 3).astype(np.float32)
        if cp.shape[0] > max_contact_pts:
            cidx = rng.choice(cp.shape[0], max_contact_pts, replace=False)
        else:
            cidx = np.arange(cp.shape[0])
        cpts = cp[cidx]

        tree = cKDTree(pc_xyz)
        dists, _ = tree.query(cpts, k=1)
        near_mask = dists < 0.03

        traces.append(go.Scatter3d(
            x=cpts[near_mask, 0], y=cpts[near_mask, 1], z=cpts[near_mask, 2],
            mode="markers",
            marker=dict(size=2.5, color="lime", opacity=0.8),
            name=f"GT contacts NEAR surface ({near_mask.sum()} pts, <3cm)",
        ))
        traces.append(go.Scatter3d(
            x=cpts[~near_mask, 0], y=cpts[~near_mask, 1], z=cpts[~near_mask, 2],
            mode="markers",
            marker=dict(size=1.5, color="red", opacity=0.4),
            name=f"GT contacts OCCLUDED ({(~near_mask).sum()} pts, >3cm)",
        ))

    # --- Table top outline (Z = 0.3) ---
    tz = 0.3
    corners = np.array([
        [-0.5, -0.6, tz], [0.5, -0.6, tz], [0.5, 0.6, tz],
        [-0.5, 0.6, tz], [-0.5, -0.6, tz],
    ])
    traces.append(go.Scatter3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        mode="lines", line=dict(color="green", width=4),
        name="Table surface edge",
    ))

    # --- Camera position + viewing direction arrow ---
    if cam_position is not None:
        traces.append(go.Scatter3d(
            x=[cam_position[0]], y=[cam_position[1]], z=[cam_position[2]],
            mode="markers+text",
            marker=dict(size=8, color="magenta", symbol="diamond"),
            text=["Camera"], textposition="top center",
            name="Camera position",
        ))
    if cam_position is not None and cam_forward is not None:
        arrow_len = 0.3
        end = cam_position + cam_forward * arrow_len
        traces.append(go.Scatter3d(
            x=[cam_position[0], end[0]],
            y=[cam_position[1], end[1]],
            z=[cam_position[2], end[2]],
            mode="lines",
            line=dict(color="magenta", width=6),
            name="Camera viewing direction",
        ))

    # --- World axes at origin ---
    axis_len = 0.15
    for ax, color, label in [(0, "red", "X"), (1, "green", "Y"), (2, "blue", "Z")]:
        end = np.zeros(3)
        end[ax] = axis_len
        traces.append(go.Scatter3d(
            x=[0, end[0]], y=[0, end[1]], z=[0, end[2]],
            mode="lines+text",
            line=dict(color=color, width=5),
            text=["", label], textposition="top center",
            name=f"World {label} axis",
            showlegend=(ax == 0),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (world)", yaxis_title="Y (world)", zaxis_title="Z (world)",
            aspectmode="data",
        ),
        legend=dict(x=0, y=1),
        width=1200, height=800,
    )

    if out_html:
        fig.write_html(str(out_html), auto_open=False)
        print(f"Saved interactive HTML → {out_html}")
    fig.show()
    print("Opened interactive viewer in browser.")


# ---------------------------------------------------------------------------
# Mode 1: load from preprocessed .npz (default)
# ---------------------------------------------------------------------------

def load_from_preprocessed(scene_index, cfg):
    """Load the preprocessed scene and transform pc_cam to world frame.

    Returns (pc_world, cam_position, cam_forward, scene_id).
    The stored camera_pose is a world-to-centered-camera transform.
    """
    data_path = cfg["DATA"]["data_path"]
    sc_dir, valid_ids = get_valid_scene_contacts(
        data_path, cfg["DATA"]["scene_contacts_path"]
    )
    scene_id = valid_ids[scene_index]

    preprocessed_dir = os.path.join(PROJECT_ROOT, "data", "preprocessed_data")
    preprocessed_path = os.path.join(preprocessed_dir, f"scene_{scene_index:06d}.npz")
    scene = dict(np.load(preprocessed_path, allow_pickle=False))

    pc_cam = scene["pc_cam"].astype(np.float32)
    camera_pose = scene["camera_pose"].astype(np.float32)

    # camera_pose: world → centered-camera (OpenCV-like after dataloader conversion)
    #   p_cam = p_world @ R.T + t          (row-vector convention)
    # Inverse:
    #   p_world = (p_cam - t) @ R
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    pc_world = (pc_cam - t[None, :]) @ R

    # Camera viewing direction in world: Z-axis of camera frame mapped to world.
    # In the stored convention camera looks along +Z (OpenCV).
    # d_world such that d_world @ R.T = [0, 0, 1]  →  d_world = [0, 0, 1] @ R = R[2, :]
    cam_forward_world = R[2, :].copy()
    cam_forward_world /= np.linalg.norm(cam_forward_world)

    # Camera position in world.
    # The centered-cam origin [0,0,0] maps to the PC centroid (because of mean-centering).
    # The actual camera is at [0,0,0] in the ORIGINAL (un-centered) cam frame,
    # which is at -pc_mean in the centered frame.
    # We estimate pc_mean[2] (mean depth) from the depth spread of pc_cam.
    # Original depths are positive (camera in front of scene), centered Z straddles 0.
    # The original min depth ≈ near clipping (≈0.04m), so:
    #   min_original = min_centered + pc_mean[2]  ≈  znear
    #   pc_mean[2]  ≈  znear - min(pc_cam[:, 2])
    # For X,Y the mean is typically small; we approximate as 0.
    znear = 0.04
    pc_mean_z = znear - pc_cam[:, 2].min()
    pc_mean_approx = np.array([0.0, 0.0, pc_mean_z])

    # Camera at -pc_mean in centered cam, mapped to world:
    cam_pos_centered = -pc_mean_approx
    cam_pos_world = (cam_pos_centered - t) @ R

    print(f"Loaded preprocessed scene_{scene_index:06d} (scene_id={scene_id})")
    print(f"  PC: {pc_cam.shape[0]} pts, camera forward (world): "
          f"[{cam_forward_world[0]:.3f}, {cam_forward_world[1]:.3f}, {cam_forward_world[2]:.3f}]")
    print(f"  Camera position (world, approx): "
          f"[{cam_pos_world[0]:.3f}, {cam_pos_world[1]:.3f}, {cam_pos_world[2]:.3f}]")

    scene_info = np.load(os.path.join(sc_dir, scene_id + ".npz"), allow_pickle=True)

    return pc_world, None, cam_pos_world, cam_forward_world, scene_info, scene_id


# ---------------------------------------------------------------------------
# Mode 2: render a fresh view via PyRender
# ---------------------------------------------------------------------------

def render_new_view(scene_index, cfg, seed):
    """Render the scene from a random camera via PyRender."""
    import trimesh.transformations as tra
    from contact_graspnet_pytorch.scene_renderer import SceneRenderer

    data_path = cfg["DATA"]["data_path"]
    sc_dir, valid_ids = get_valid_scene_contacts(
        data_path, cfg["DATA"]["scene_contacts_path"]
    )
    scene_id = valid_ids[scene_index]
    scene_info = np.load(os.path.join(sc_dir, scene_id + ".npz"), allow_pickle=True)

    os.environ["PYOPENGL_PLATFORM"] = "egl"
    renderer = SceneRenderer(caching=True, intrinsics=cfg["DATA"]["intrinsics"])
    full_paths = [os.path.join(data_path, p) for p in scene_info["obj_paths"]]
    renderer.change_scene(full_paths, scene_info["obj_scales"], scene_info["obj_transforms"])

    import pyrender
    cam_node = renderer._camera_node
    renderer._scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=5.0),
        pose=np.eye(4), parent_node=cam_node, name="headlight",
    )
    for angle in [np.pi / 3, -np.pi / 3]:
        renderer._scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
            pose=tra.euler_matrix(0, angle, 0),
            parent_node=cam_node, name=f"sidelight_{angle:.1f}",
        )

    np.random.seed(seed)
    cam_orientations = []
    elev = np.array(cfg["DATA"]["view_sphere"]["elevation"]) / 180.0
    for az in np.linspace(0, np.pi * 2, 30):
        for el in np.linspace(elev[0], elev[1], 30):
            cam_orientations.append(tra.euler_matrix(0, -el, az))
    coord_tf = tra.euler_matrix(np.pi / 2, 0, 0).dot(tra.euler_matrix(0, np.pi / 2, 0))
    idx = np.random.randint(0, len(cam_orientations))
    distance = cfg["DATA"]["view_sphere"]["distance_range"][0] + \
        np.random.rand() * (cfg["DATA"]["view_sphere"]["distance_range"][1] -
                            cfg["DATA"]["view_sphere"]["distance_range"][0])
    ext = np.eye(4)
    ext[0, 3] += distance
    ext = cam_orientations[idx].dot(ext).dot(coord_tf)
    ext[2, 3] += renderer._table_dims[2]
    ext[:3, :2] = -ext[:3, :2]

    color, depth, pc_homo, cam_pose_out = renderer.render(ext, render_pc=True)
    pc_cv = pc_homo[:, :3]
    pc_gl = pc_cv.copy()
    pc_gl[:, 1] = -pc_gl[:, 1]
    pc_gl[:, 2] = -pc_gl[:, 2]
    R_c2w = cam_pose_out[:3, :3]
    t_c2w = cam_pose_out[:3, 3]
    pc_world = (R_c2w @ pc_gl.T).T + t_c2w[None, :]

    h, w = depth.shape
    mask = np.where(depth > 0)
    pc_rgb = color[mask[0], mask[1]]

    cam_forward = -cam_pose_out[:3, 2]
    cam_forward /= np.linalg.norm(cam_forward)

    return pc_world, pc_rgb, t_c2w, cam_forward, scene_info, scene_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed CGN scene in world frame (Plotly)."
    )
    parser.add_argument("--scene-index", type=int, default=0,
                        help="Preprocessed scene index (0 → scene_000000).")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlay-contacts", action="store_true",
                        help="Show GT grasp contact points.")
    parser.add_argument("--max-render-pts", type=int, default=30000)
    parser.add_argument("--render-new", action="store_true",
                        help="Render a fresh random view via PyRender instead of "
                             "loading the stored preprocessed camera.")
    args = parser.parse_args()

    config_path = os.path.join(
        PROJECT_ROOT, "contact_graspnet_pytorch",
        "contact_graspnet_pytorch", "config.yaml",
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.render_new:
        pc_world, pc_rgb, cam_pos, cam_fwd, scene_info, scene_id = \
            render_new_view(args.scene_index, cfg, args.seed)
    else:
        pc_world, pc_rgb, cam_pos, cam_fwd, scene_info, scene_id = \
            load_from_preprocessed(args.scene_index, cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.out_dir / f"scene_{args.scene_index:06d}_interactive.html"

    show_interactive_plotly(
        pc_world, pc_rgb,
        cam_position=cam_pos,
        cam_forward=cam_fwd,
        scene_info=scene_info if args.overlay_contacts else None,
        show_contacts=args.overlay_contacts,
        max_render_pts=args.max_render_pts,
        seed=args.seed,
        title=f"Scene {scene_id} – World Frame"
              f"{' (preprocessed camera)' if not args.render_new else ' (random camera)'}",
        out_html=html_path,
    )


if __name__ == "__main__":
    main()
