import pytest
import numpy as np
from scipy.linalg import expm # For direct comparison if needed
from wildcat_slam.geometry import (
    so3_hat, so3_vee, so3_exp, so3_log,
    se3_hat, se3_vee, se3_exp, se3_log,
    linear_interpolate_translation, RotInterpolate,
    CubicBSpline
)

# Fixtures for random data
@pytest.fixture
def random_so3_vector():
    return np.random.rand(3) * np.pi # Angle up to pi

@pytest.fixture
def random_small_so3_vector():
    # Smaller angles for better behavior of logm near identity
    return (np.random.rand(3) - 0.5) * 0.2 # Small values around 0, up to ~0.1 rad

@pytest.fixture
def random_se3_vector():
    v = (np.random.rand(3) - 0.5) * 10 # Translations up to +/- 5 units
    omega = (np.random.rand(3) - 0.5) * 0.2 # Small rotations
    return np.concatenate((v, omega))

def test_so3_hat_properties(random_so3_vector):
    omega = random_so3_vector
    Omega_hat = so3_hat(omega)
    assert Omega_hat.shape == (3,3)
    # Check skew-symmetry
    assert np.allclose(Omega_hat, -Omega_hat.T)
    # Check known values
    assert Omega_hat[0,1] == -omega[2]
    assert Omega_hat[0,2] == omega[1]
    assert Omega_hat[1,2] == -omega[0]

def test_so3_hat_vee_roundtrip(random_so3_vector):
    omega_orig = random_so3_vector
    omega_rt = so3_vee(so3_hat(omega_orig))
    assert np.allclose(omega_orig, omega_rt)

def test_so3_vee_properties():
    mat = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    vec = so3_vee(mat)
    assert np.allclose(vec, [1, 2, 3])
    with pytest.raises(ValueError, match="skew-symmetric"): # دقیق تر کردن شرط
        so3_vee(np.eye(3)) # Not skew-symmetric

def test_so3_exp_log_roundtrip(random_small_so3_vector):
    omega_orig_vec = random_small_so3_vector
    omega_orig_hat = so3_hat(omega_orig_vec)

    R = so3_exp(omega_orig_hat)
    # Check if R is indeed in SO(3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "R @ R.T is not Identity"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "Determinant of R is not 1"

    omega_rt_hat = so3_log(R)

    # so3_log should return a skew-symmetric matrix
    assert np.allclose(omega_rt_hat, -omega_rt_hat.T, atol=1e-7), "log(R) is not skew-symmetric"
    assert np.allclose(omega_orig_hat, omega_rt_hat, atol=1e-7) # Increased tolerance

def test_so3_exp_known_values():
    # 90 deg rotation around z-axis
    omega_z_90_vec = np.array([0, 0, np.pi/2])
    omega_z_90_hat = so3_hat(omega_z_90_vec)
    R_z_90_expected = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    R_z_90_actual = so3_exp(omega_z_90_hat)
    assert np.allclose(R_z_90_actual, R_z_90_expected, atol=1e-7)

    # Log of this rotation
    omega_log_hat = so3_log(R_z_90_expected)
    assert np.allclose(omega_log_hat, omega_z_90_hat, atol=1e-7)


def test_se3_hat_properties(random_se3_vector):
    xi = random_se3_vector
    Xi_hat = se3_hat(xi)
    assert Xi_hat.shape == (4,4)
    # Check bottom row is zero
    assert np.allclose(Xi_hat[3,:], 0)
    # Check top-left 3x3 is skew-symmetric (from so3_hat)
    assert np.allclose(Xi_hat[:3,:3], -Xi_hat[:3,:3].T)
    # Check translation part
    assert np.allclose(Xi_hat[:3,3], xi[:3])

def test_se3_hat_vee_roundtrip(random_se3_vector):
    xi_orig = random_se3_vector
    xi_rt = se3_vee(se3_hat(xi_orig))
    assert np.allclose(xi_orig, xi_rt)

def test_se3_exp_log_roundtrip(random_se3_vector):
    xi_orig = random_se3_vector
    # Using small rotations for se3_vector to ensure logm behaves well
    # The random_se3_vector fixture already generates small rotations.
    xi_orig_hat = se3_hat(xi_orig)

    T = se3_exp(xi_orig_hat)
    # Check if T is in SE(3)
    R_part = T[:3,:3]
    t_part = T[:3,3]
    assert np.allclose(R_part @ R_part.T, np.eye(3), atol=1e-6), "R part of T is not SO(3)"
    assert np.isclose(np.linalg.det(R_part), 1.0, atol=1e-6), "Determinant of R part is not 1"
    assert np.allclose(T[3,:], [0,0,0,1]), "Bottom row of T is incorrect"

    xi_rt_hat = se3_log(T)

    # Check if xi_rt_hat is in se(3)
    assert np.allclose(xi_rt_hat[3,:], 0, atol=1e-7), "Bottom row of log(T) is not zero"
    assert np.allclose(xi_rt_hat[:3,:3], -xi_rt_hat[:3,:3].T, atol=1e-7), "Rot part of log(T) not skew-symmetric"

    assert np.allclose(xi_orig_hat, xi_rt_hat, atol=1e-7)


def test_linear_interpolate_translation():
    t1 = np.array([1.0, 2.0, 3.0])
    t2 = np.array([5.0, 6.0, 7.0])

    # Alpha = 0
    assert np.allclose(linear_interpolate_translation(t1, t2, 0.0), t1)
    # Alpha = 1
    assert np.allclose(linear_interpolate_translation(t1, t2, 1.0), t2)
    # Alpha = 0.5
    t_mid_expected = np.array([3.0, 4.0, 5.0])
    assert np.allclose(linear_interpolate_translation(t1, t2, 0.5), t_mid_expected)

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        linear_interpolate_translation(t1, t2, -0.1)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        linear_interpolate_translation(t1, t2, 1.1)


def test_rot_interpolate():
    # r1_vec, r2_vec are so(3) vectors (axis-angle scaled by angle)
    r1_v = np.array([0.0, 0.0, 0.1]) # Small rotation around z by 0.1 rad
    r2_v = np.array([0.0, 0.0, 0.5]) # Rotation around z by 0.5 rad

    # Alpha = 0
    R_interp_0 = RotInterpolate(r1_v, r2_v, 0.0)
    R_expected_0 = so3_exp(so3_hat(r1_v))
    assert np.allclose(R_interp_0, R_expected_0, atol=1e-7)

    # Alpha = 1
    R_interp_1 = RotInterpolate(r1_v, r2_v, 1.0)
    R_expected_1 = so3_exp(so3_hat(r2_v))
    assert np.allclose(R_interp_1, R_expected_1, atol=1e-7)

    # Alpha = 0.5
    r_mid_v = (r1_v + r2_v) / 2.0 # Expected so(3) vector for midpoint = [0,0,0.3]
    R_interp_0_5 = RotInterpolate(r1_v, r2_v, 0.5)
    R_expected_0_5 = so3_exp(so3_hat(r_mid_v))
    assert np.allclose(R_interp_0_5, R_expected_0_5, atol=1e-7)

    # Test with a more complex rotation (not just around one axis)
    r1_complex = np.array([0.1, 0.05, -0.02])
    r2_complex = np.array([-0.05, 0.15, 0.1])

    alpha_c = 0.3
    r_interp_c_v_expected = (1 - alpha_c) * r1_complex + alpha_c * r2_complex
    R_expected_c = so3_exp(so3_hat(r_interp_c_v_expected))
    R_interp_c = RotInterpolate(r1_complex, r2_complex, alpha_c)
    assert np.allclose(R_interp_c, R_expected_c, atol=1e-7)

    # Test if alpha can be outside [0,1] as RotInterpolate was relaxed
    RotInterpolate(r1_v, r2_v, 1.5) # Should not raise error based on current geometry.py
    RotInterpolate(r1_v, r2_v, -0.5)


# --- CubicBSpline Tests (testing the placeholder behavior) ---
@pytest.fixture
def sample_spline_data():
    T0 = np.eye(4)
    T1 = se3_exp(se3_hat(np.array([1.0, 0.5, 0.2, 0, 0, 0.1]))) # dx=1, dy=0.5, dz=0.2, rot_z=0.1
    T2 = se3_exp(se3_hat(np.array([2.0, 1.0, 0.4, 0, 0, 0.2])))
    T3 = se3_exp(se3_hat(np.array([3.0, 1.5, 0.6, 0, 0, 0.3])))
    control_poses = [T0, T1, T2, T3]
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    return control_poses, timestamps

def test_spline_init(sample_spline_data):
    control_poses, timestamps = sample_spline_data
    spline = CubicBSpline(control_poses, timestamps)
    assert len(spline.control_poses_se3) == len(control_poses)
    assert np.allclose(spline.timestamps, timestamps)

    with pytest.raises(ValueError, match="at least 4 control poses"):
        CubicBSpline(control_poses[:3], timestamps[:3])
    with pytest.raises(ValueError, match="must match"):
        CubicBSpline(control_poses, timestamps[:3])


def test_spline_endpoints(sample_spline_data):
    control_poses, timestamps = sample_spline_data
    spline = CubicBSpline(control_poses, timestamps)

    # Test evaluation at the exact start and end timestamps
    pose_at_start = spline.evaluate_pose(timestamps[0])
    assert np.allclose(pose_at_start, control_poses[0], atol=1e-7)

    pose_at_end = spline.evaluate_pose(timestamps[-1])
    assert np.allclose(pose_at_end, control_poses[-1], atol=1e-7)

    # Test clamping behavior (current placeholder behavior)
    pose_before_start = spline.evaluate_pose(timestamps[0] - 0.5)
    assert np.allclose(pose_before_start, control_poses[0], atol=1e-7)

    pose_after_end = spline.evaluate_pose(timestamps[-1] + 0.5)
    assert np.allclose(pose_after_end, control_poses[-1], atol=1e-7)


def test_spline_midpoint_lerp_behavior(sample_spline_data):
    """
    Tests the current LERP-like behavior of the spline placeholder at a midpoint.
    This is NOT a test of correct B-spline math, only the current placeholder.
    """
    control_poses, timestamps = sample_spline_data
    spline = CubicBSpline(control_poses, timestamps)

    # Midpoint between T0 and T1 (timestamps[0] and timestamps[1])
    t_mid = (timestamps[0] + timestamps[1]) / 2.0 # = 0.5

    # Expected LERP result:
    T0 = control_poses[0]
    T1 = control_poses[1]
    alpha = 0.5

    # Expected translation
    trans0 = T0[:3,3]
    trans1 = T1[:3,3]
    expected_trans_mid = linear_interpolate_translation(trans0, trans1, alpha)

    # Expected rotation (simplified: interpolate so(3) vectors of T0, T1)
    # R0 = I, so log(R0) = 0 vector
    # R1 = T1[:3,:3], log(R1) = so3_vee(so3_log(R1))
    # r_mid_vec = 0.5 * so3_vee(so3_log(T0[:3,:3])) + 0.5 * so3_vee(so3_log(T1[:3,:3]))
    # This is what the placeholder's RotInterpolate effectively does if inputs were so(3) vecs.
    # The placeholder spline uses: R_interp = R1 @ so3_exp(alpha * so3_log(R1.T @ R2))
    # For T0=Identity, R0=Identity. R_rel = R0.T @ R1 = R1.
    # omega_hat_rel = so3_log(R1)
    # R_interp = R0 @ so3_exp(alpha * omega_hat_rel) = so3_exp(alpha * so3_log(R1))

    R0 = T0[:3,:3] # Identity
    R1 = T1[:3,:3]

    # From spline.evaluate_pose logic for midpoint between T0 and T1:
    # R_interp = R0 @ so3_exp(0.5 * so3_log(R0.T @ R1))
    # Since R0 is Identity, R_interp = so3_exp(0.5 * so3_log(R1))
    xi1_vec = se3_vee(se3_log(T1)) # xi1 = [v1, omega1]
    omega1_vec = xi1_vec[3:]       # This is log(R1) in vector form if T0 was identity for se3_log.
                                   # More directly: omega1_vec = so3_vee(so3_log(R1))

    expected_R_mid = so3_exp(0.5 * so3_log(R1)) # Since R0 is identity
                                               # and T0 is identity, log(R1) is the effective omega1_vec
                                               # used by the LERP logic in evaluate_pose.

    expected_pose_mid = np.eye(4)
    expected_pose_mid[:3,:3] = expected_R_mid
    expected_pose_mid[:3,3] = expected_trans_mid

    actual_pose_mid = spline.evaluate_pose(t_mid)

    # Print for debugging if it fails
    # print("T0:\n", T0)
    # print("T1:\n", T1)
    # print("xi1_vec (from T1):\n", xi1_vec) # This is log(T1) assuming T_ref=Identity
    # print("omega1_vec (from T1 rot part):\n", so3_vee(so3_log(R1)))
    # print("Actual Mid Trans:", actual_pose_mid[:3,3])
    # print("Expected Mid Trans:", expected_pose_mid[:3,3])
    # print("Actual Mid Rot:\n", actual_pose_mid[:3,:3])
    # print("Expected Mid Rot:\n", expected_pose_mid[:3,:3])

    assert np.allclose(actual_pose_mid, expected_pose_mid, atol=1e-6)


def test_slerp_quaternions_known(): # Not directly in plan, but good utility if needed later
    from wildcat_slam.geometry import slerp_quaternions # Corrected import
    # Scipy uses [x,y,z,w] convention for Rotation.as_quat()
    # Let's assume our slerp uses [x,y,z,w] for this test to match scipy easily
    from scipy.spatial.transform import Rotation as R

    # Quaternion for identity
    q1 = R.from_matrix(np.eye(3)).as_quat() # [0,0,0,1]

    # Quaternion for 90 deg rotation around Z
    R_z_90 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    q2 = R.from_matrix(R_z_90).as_quat() # [0,0,sin(pi/4),cos(pi/4)]

    # Midpoint (alpha=0.5) should be 45 deg rotation around Z
    q_mid_expected = R.from_matrix(so3_exp(so3_hat(np.array([0,0,np.pi/4])))).as_quat()

    q_mid_actual = slerp_quaternions(q1, q2, 0.5)
    assert np.allclose(q_mid_actual, q_mid_expected, atol=1e-7) or \
           np.allclose(q_mid_actual, -q_mid_expected, atol=1e-7) # Quaternions are double-cover

    # Alpha = 0
    q_0_actual = slerp_quaternions(q1, q2, 0.0)
    assert np.allclose(q_0_actual, q1, atol=1e-7)  or \
           np.allclose(q_0_actual, -q1, atol=1e-7)

    # Alpha = 1
    q_1_actual = slerp_quaternions(q1, q2, 1.0)
    assert np.allclose(q_1_actual, q2, atol=1e-7) or \
           np.allclose(q_1_actual, -q2, atol=1e-7)

    # Test antipodal case
    q2_anti = -q2
    dot_val = np.dot(q1, q2_anti) # Should be negative
    assert dot_val < 0
    q_mid_anti_actual = slerp_quaternions(q1, q2_anti, 0.5)
    # Should still give same result as slerp(q1,q2,0.5) because q2_anti is flipped internally
    assert np.allclose(q_mid_anti_actual, q_mid_expected, atol=1e-7) or \
           np.allclose(q_mid_anti_actual, -q_mid_expected, atol=1e-7)

    # Test very close quaternions
    q_close = R.from_rotvec([0,0,1e-5]).as_quat()
    q_mid_close = slerp_quaternions(q1, q_close, 0.5)
    q_mid_close_expected = R.from_rotvec([0,0,0.5e-5]).as_quat()
    assert np.allclose(q_mid_close, q_mid_close_expected, atol=1e-7) or \
           np.allclose(q_mid_close, -q_mid_close_expected, atol=1e-7)
