"""
Converts CGN network outputs (4-DoF) into full SE(3) grasp poses (6-DoF).
Math mirrors contact_graspnet/contact_graspnet/contact_graspnet.py::build_6d_grasp.
"""

import numpy as np

# Panda gripper: distance from gripper frame origin to fingertip along approach axis in meters
GRIPPER_DEPTH = 0.1034


def contact_to_grasp_pose(contact_pts, base_dirs, approach_dirs, widths):
    """
    Build Nx4x4 SE(3) grasp poses from per-point network predictions.

    Args:
        contact_pts  (np.ndarray): Nx3  contact point on object surface
        base_dirs    (np.ndarray): Nx3  gripper baseline direction (finger-to-finger)
        approach_dirs(np.ndarray): Nx3  approach direction (z-axis of gripper frame)
        widths       (np.ndarray): N    predicted gripper opening width in meters

    Returns:
        np.ndarray: Nx4x4 homogeneous grasp poses in camera coordinates
    """
    N = len(contact_pts)
    grasps = np.eye(4)[None].repeat(N, axis=0)  # Nx4x4 identity init

    # Normalize input directions
    z = approach_dirs / (np.linalg.norm(approach_dirs, axis=1, keepdims=True) + 1e-8)

    # y-axis: cross(z, base) then normalize — perpendicular to both
    x_raw = base_dirs / (np.linalg.norm(base_dirs, axis=1, keepdims=True) + 1e-8)
    y = np.cross(z, x_raw)
    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)

    # Recompute x = cross(y, z) to guarantee orthonormality (Gram-Schmidt)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    # Rotation matrix columns: [x | y | z]
    grasps[:, :3, 0] = x
    grasps[:, :3, 1] = y
    grasps[:, :3, 2] = z

    # Translation: center of baseline, pulled back from contact along approach
    grasps[:, :3, 3] = (
        contact_pts
        + (widths[:, None] / 2.0) * x
        - GRIPPER_DEPTH * z
    )

    return grasps


# ── Tests ─────────────────────────────────────────────────────────────────────

def _test_output_shape(): # should produce Nx4x4 output
    N = 10
    pts  = np.random.randn(N, 3)
    base = np.random.randn(N, 3)
    app  = np.random.randn(N, 3)
    w    = np.random.uniform(0, 0.08, N)
    out  = contact_to_grasp_pose(pts, base, app, w)
    assert out.shape == (N, 4, 4), "Shape mismatch"
    print("PASS  output shape:", out.shape)


def _test_rotation_orthonormal(): # R^T*R = I
    N = 50
    pts  = np.random.randn(N, 3)
    base = np.random.randn(N, 3)
    app  = np.random.randn(N, 3)
    w    = np.random.uniform(0, 0.08, N)
    out  = contact_to_grasp_pose(pts, base, app, w)
    R = out[:, :3, :3]
    I = R @ R.transpose(0, 2, 1)          # should be identity for each
    err = np.max(np.abs(I - np.eye(3)))
    assert err < 1e-6, f"Rotation not orthonormal, max err={err:.2e}"
    print(f"PASS  rotation orthonormal (max err={err:.2e})")


def _test_homogeneous_row(): # bottom row should be [0, 0, 0, 1]
    N = 5
    out = contact_to_grasp_pose(
        np.random.randn(N, 3),
        np.random.randn(N, 3),
        np.random.randn(N, 3),
        np.random.uniform(0, 0.08, N),
    )
    assert np.allclose(out[:, 3, :], [0, 0, 0, 1]), "Bottom row not [0,0,0,1]"
    print("PASS  homogeneous bottom row")


def _test_known_pose(): # sanity check with axis-aligned inputs
    """
    Sanity check with axis-aligned inputs whose result is hand-verifiable.
    base=[1,0,0], approach=[0,0,1] -> R=I, t = contact + width/2*x - depth*z
    """
    contact = np.array([[0.0, 0.0, 0.5]])
    base    = np.array([[1.0, 0.0, 0.0]])
    approach= np.array([[0.0, 0.0, 1.0]])
    width   = np.array([0.08])

    out = contact_to_grasp_pose(contact, base, approach, width)

    expected_t = np.array([0.0 + 0.04, 0.0, 0.5 - GRIPPER_DEPTH])
    assert np.allclose(out[0, :3, 3], expected_t, atol=1e-6), \
        f"Translation wrong: {out[0,:3,3]} vs {expected_t}"
    print("PASS  known pose translation")


if __name__ == "__main__":
    _test_output_shape()
    _test_rotation_orthonormal()
    _test_homogeneous_row()
    _test_known_pose()
    print("\nAll tests passed.")
