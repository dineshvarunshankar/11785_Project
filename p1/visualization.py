"""
Visualizes predicted grasps on a point cloud scene using Open3D.
Logic mirrors contact_graspnet/contact_graspnet/visualization_utils.py::draw_grasps
and contact_graspnet_pytorch/contact_graspnet_pytorch/visualization_utils_o3d.py
"""

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# Panda gripper geometry constants (metres)
# These define the U-shaped wireframe in the gripper's local coordinate frame.
# z-axis = approach direction, x-axis = baseline (finger-to-finger)
_APPROACH_DEPTH  = 0.0584   # wrist to finger root along z
_GRIPPER_DEPTH   = 0.1034   # wrist to fingertip along z


def _gripper_control_points(opening):
    """
    Returns 7x3 control points of the Panda gripper wireframe in local frame.
    The path order draws: wrist → baseline_mid → left_root → left_tip
                          → left_root → right_root → right_tip

    Args:
        opening (float): gripper opening width in metres

    Returns:
        np.ndarray: 7x3 control points
    """
    hw = opening / 2.0  # half-width

    wrist        = [0,   0, 0]
    baseline_mid = [0,   0, _APPROACH_DEPTH]
    left_root    = [-hw, 0, _APPROACH_DEPTH]
    left_tip     = [-hw, 0, _GRIPPER_DEPTH]
    right_root   = [ hw, 0, _APPROACH_DEPTH]
    right_tip    = [ hw, 0, _GRIPPER_DEPTH]

    return np.array([
        wrist,
        baseline_mid,
        left_root,
        left_tip,
        left_root,      # backtrack to draw right side
        right_root,
        right_tip,
    ], dtype=np.float64)


def _gripper_lines_in_world(grasp_pose, opening):
    """
    Transforms gripper control points from local frame to world frame
    using the 4x4 grasp pose, then builds Open3D line segments.

    Args:
        grasp_pose (np.ndarray): 4x4 SE(3) grasp pose
        opening    (float):      gripper opening width in metres

    Returns:
        tuple: (pts Nx3, connections Nx2) in world coordinates
    """
    pts_local = _gripper_control_points(opening)  # 7x3

    # Apply rotation and translation: pts_world = R @ pts_local.T + t
    pts_world = (grasp_pose[:3, :3] @ pts_local.T).T + grasp_pose[:3, 3]

    # Connect consecutive points: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6
    N = len(pts_world)
    connections = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1)

    return pts_world, connections


def visualize_grasps(pc, pred_grasps_cam, scores,
                     pc_colors=None, topk=50,
                     gripper_openings=None, default_opening=0.08,
                     save_path=None, show=True):
    """
    Renders the scene point cloud and top-k predicted grasps colored by score.

    When multiple segments are present, each segment gets a distinct color.
    When only one segment exists (key -1, full scene), grasps are colored
    individually by confidence using the viridis colormap — matching CGN behavior.

    Args:
        pc               (np.ndarray):        Nx3 scene point cloud
        pred_grasps_cam  (dict[int, ndarray]): segment_id → Mx4x4 grasp poses
        scores           (dict[int, ndarray]): segment_id → M confidence scores
        pc_colors        (np.ndarray):         Nx3 RGB colors for point cloud (0-255)
        topk             (int):                number of top grasps to show per segment
        gripper_openings (dict[int, ndarray]): segment_id → M opening widths (metres)
        default_opening  (float):              fallback opening if not provided
        save_path        (str):                if set, saves render as PNG to this path
        show             (bool):               open interactive Open3D window
    """
    geometries = []

    # -- Point cloud --
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    if pc_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pc_colors.astype(np.float64) / 255.0)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(pcd)

    # -- Colormaps --
    cm_segments = plt.get_cmap("rainbow")   # one color per segment
    cm_scores   = plt.get_cmap("viridis")   # per-grasp score color

    seg_keys = [k for k in pred_grasps_cam if np.any(pred_grasps_cam[k])]
    single_segment = (len(seg_keys) == 1)

    for seg_idx, k in enumerate(seg_keys):
        grasps_k = pred_grasps_cam[k]
        scores_k = scores[k]

        # Select top-k by score
        n = min(topk, len(grasps_k))
        top_idcs = np.argsort(scores_k)[::-1][:n]
        grasps_k = grasps_k[top_idcs]
        scores_k = scores_k[top_idcs]

        openings_k = (
            gripper_openings[k][top_idcs]
            if gripper_openings is not None and k in gripper_openings
            else np.full(n, default_opening)
        )

        # Assign colors
        if single_segment:
            s_min, s_max = scores_k.min(), scores_k.max()
            span = s_max - s_min if s_max > s_min else 1.0
            grasp_colors = [
                cm_scores((s - s_min) / span)[:3] for s in scores_k
            ]
        else:
            base_color = cm_segments(seg_idx / max(len(seg_keys) - 1, 1))[:3]
            grasp_colors = [base_color] * n

        # Build LineSet for all grasps in this segment
        all_pts   = []
        all_conns = []
        all_colors = []
        offset = 0

        for i, (g, opening, color) in enumerate(
            zip(grasps_k, openings_k, grasp_colors)
        ):
            pts, conns = _gripper_lines_in_world(g, opening)
            all_pts.append(pts)
            all_conns.append(conns + offset)
            all_colors.extend([list(color)] * len(conns))
            offset += len(pts)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
        line_set.lines  = o3d.utility.Vector2iVector(np.vstack(all_conns))
        line_set.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        geometries.append(line_set)

    # -- Save headless render --
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        render = o3d.visualization.rendering.OffscreenRenderer(1280, 720)
        mat_pc = o3d.visualization.rendering.MaterialRecord()
        mat_pc.shader = "defaultUnlit"
        mat_line = o3d.visualization.rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 2.0
        for idx, geom in enumerate(geometries):
            if isinstance(geom, o3d.geometry.LineSet):
                render.scene.add_geometry(f"geom_{idx}", geom, mat_line)
            else:
                render.scene.add_geometry(f"geom_{idx}", geom, mat_pc)
        render.scene.set_background([1, 1, 1, 1])
        bounds = render.scene.bounding_box
        center = bounds.get_center()
        extent = np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound())
        eye    = center + np.array([0, 0, -extent])
        up     = np.array([0, -1, 0])
        render.setup_camera(60.0, center.astype(np.float32),
                            eye.astype(np.float32),
                            up.astype(np.float32))
        img = render.render_to_image()
        # Convert Open3D image to numpy and save with PIL (more reliable than o3d.io.write_image)
        img_np = np.asarray(img)
        from PIL import Image as PILImage
        PILImage.fromarray(img_np).save(save_path)
        print(f"Saved render to {save_path}")

    # -- Interactive window --
    if show:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="CGN Predicted Grasps",
            width=1280, height=720,
        )
