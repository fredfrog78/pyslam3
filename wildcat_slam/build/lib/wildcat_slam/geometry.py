import numpy as np
from scipy.linalg import expm, logm

# SO(3) operations

def so3_hat(omega):
    """
    Maps a 3-vector omega to its corresponding skew-symmetric matrix (so(3)).
    omega: (3,) array
    Returns: (3,3) skew-symmetric matrix
    """
    if not isinstance(omega, np.ndarray) or omega.shape != (3,):
        raise ValueError("Input omega must be a (3,) numpy array.")
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

def so3_vee(Omega):
    """
    Maps a skew-symmetric matrix Omega (so(3)) to its corresponding 3-vector.
    Omega: (3,3) skew-symmetric matrix
    Returns: (3,) array
    """
    if not isinstance(Omega, np.ndarray) or Omega.shape != (3,3):
        raise ValueError("Input Omega must be a (3,3) numpy array.")
    # Check for skew-symmetry (optional, but good for robustness)
    if not np.allclose(Omega, -Omega.T):
        # Allow for small numerical errors if it's result of expm/logm
        if not np.allclose(Omega, -Omega.T, atol=1e-7): # Increased tolerance
             raise ValueError("Input Omega must be skew-symmetric.")
    return np.array([Omega[2,1], Omega[0,2], Omega[1,0]])


def so3_exp(omega_hat):
    """
    Computes the SO(3) matrix from an so(3) element omega_hat (a skew-symmetric matrix).
    This is the matrix exponential.
    omega_hat: (3,3) skew-symmetric matrix (so(3) element)
    Returns: (3,3) rotation matrix (SO(3) element)
    """
    if not isinstance(omega_hat, np.ndarray) or omega_hat.shape != (3,3):
        raise ValueError("Input omega_hat must be a (3,3) numpy array.")
    # Optional: check if omega_hat is actually skew-symmetric
    # if not np.allclose(omega_hat, -omega_hat.T):
    #     raise ValueError("Input omega_hat must be skew-symmetric for so3_exp.")
    return expm(omega_hat)

def so3_log(R):
    """
    Computes the so(3) element (skew-symmetric matrix) from an SO(3) rotation matrix.
    This is the matrix logarithm.
    R: (3,3) rotation matrix (SO(3) element)
    Returns: (3,3) skew-symmetric matrix (so(3) element)
    """
    if not isinstance(R, np.ndarray) or R.shape != (3,3):
        raise ValueError("Input R must be a (3,3) numpy array.")
    # Optional: check if R is a valid rotation matrix
    # if not np.allclose(R.T @ R, np.eye(3)) or not np.isclose(np.linalg.det(R), 1.0):
    #    raise ValueError("Input R must be a valid SO(3) rotation matrix.")

    # logm can return complex results if R is not perfectly SO(3) due to numerical noise.
    # We take the real part. For valid SO(3), imaginary part should be negligible.
    log_R = logm(R)
    if np.max(np.abs(np.imag(log_R))) > 1e-9: # Check if imaginary part is significant
        # This might indicate R is far from SO(3)
        # Depending on strictness, could raise error or just warn
        # print("Warning: so3_log encountered a matrix with significant imaginary part in logarithm.")
        pass

    # Ensure the result is skew-symmetric by averaging with -result.T
    # This helps clean up numerical inaccuracies from logm for matrices close to SO(3)
    skew_symmetric_log_R = (log_R - log_R.T) / 2.0
    return np.real(skew_symmetric_log_R)


# SE(3) operations (represented as 4x4 homogeneous matrices)

def se3_hat(xi):
    """
    Maps a 6-vector xi (twist coordinates: v, omega) to its corresponding
    4x4 matrix representation in se(3).
    xi: (6,) array [v_x, v_y, v_z, omega_x, omega_y, omega_z]
    Returns: (4,4) matrix in se(3)
    """
    if not isinstance(xi, np.ndarray) or xi.shape != (6,):
        raise ValueError("Input xi must be a (6,) numpy array.")
    v = xi[:3]
    omega = xi[3:]
    Omega_hat = so3_hat(omega)
    T_xi = np.zeros((4,4))
    T_xi[:3,:3] = Omega_hat
    T_xi[:3,3] = v
    return T_xi

def se3_vee(Xi_hat):
    """
    Maps a 4x4 matrix Xi_hat in se(3) back to its 6-vector twist coordinates.
    Xi_hat: (4,4) matrix in se(3)
    Returns: (6,) array [v_x, v_y, v_z, omega_x, omega_y, omega_z]
    """
    if not isinstance(Xi_hat, np.ndarray) or Xi_hat.shape != (4,4):
        raise ValueError("Input Xi_hat must be a (4,4) numpy array.")
    # Optional: check if bottom row is all zeros
    if not np.allclose(Xi_hat[3,:], 0):
        raise ValueError("Input Xi_hat must have its bottom row as zeros for se(3).")

    Omega_hat = Xi_hat[:3,:3]
    v = Xi_hat[:3,3]
    omega = so3_vee(Omega_hat)
    return np.concatenate((v, omega))

def se3_exp(xi_hat):
    """
    Computes the SE(3) matrix from an se(3) element xi_hat (4x4 matrix).
    This is the matrix exponential for SE(3).
    xi_hat: (4,4) matrix (se(3) element)
    Returns: (4,4) homogeneous transformation matrix (SE(3) element)
    """
    if not isinstance(xi_hat, np.ndarray) or xi_hat.shape != (4,4):
        raise ValueError("Input xi_hat must be a (4,4) numpy array.")
    # Optional: check if xi_hat is a valid se(3) element
    # e.g., bottom row is zero, top-left 3x3 is skew-symmetric
    return expm(xi_hat)

def se3_log(T):
    """
    Computes the se(3) element (4x4 matrix) from an SE(3) transformation matrix.
    T: (4,4) homogeneous transformation matrix (SE(3) element)
    Returns: (4,4) matrix (se(3) element)
    """
    if not isinstance(T, np.ndarray) or T.shape != (4,4):
        raise ValueError("Input T must be a (4,4) numpy array.")
    # Optional: check if T is a valid SE(3) matrix
    # e.g., R part is SO(3), bottom row is [0,0,0,1]
    # R_part = T[:3,:3]
    # if not np.allclose(R_part.T @ R_part, np.eye(3)) or not np.isclose(np.linalg.det(R_part), 1.0):
    #    raise ValueError("Rotation part of T is not a valid SO(3) matrix.")
    # if not np.allclose(T[3,:], [0,0,0,1]):
    #    raise ValueError("Bottom row of T is not [0,0,0,1].")

    log_T = logm(T)
    # Similar to so3_log, ensure the result has the correct se(3) structure if T is noisy
    # The rotational part should be skew-symmetric, bottom row should be zero
    log_T_real = np.real(log_T) # logm can produce complex for matrices not perfectly SE(3)

    # Make rotational part skew-symmetric
    R_log = log_T_real[:3,:3]
    R_log_skew = (R_log - R_log.T) / 2.0

    result_xi_hat = np.zeros((4,4))
    result_xi_hat[:3,:3] = R_log_skew
    result_xi_hat[:3,3] = log_T_real[:3,3]

    return result_xi_hat

# Interpolation

def linear_interpolate_translation(t1, t2, alpha):
    """
    Linearly interpolates between two translation vectors t1 and t2.
    t1, t2: (3,) arrays representing translation vectors.
    alpha: float, interpolation factor (0 <= alpha <= 1).
           alpha=0 returns t1, alpha=1 returns t2.
    Returns: (3,) array, interpolated translation vector.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1.")
    return (1 - alpha) * t1 + alpha * t2

def RotInterpolate(r1_vec, r2_vec, alpha):
    """
    Interpolates between two rotations r1 and r2, represented by their so(3) vectors.
    This typically means converting to SO(3), finding relative rotation, scaling, and applying.
    A common method is Spherical Linear Interpolation (Slerp) on quaternions,
    or interpolating on the Lie algebra (so(3)) and then exponentiating.
    The Wildcat paper mentions "linearly interpolating ... on so(3) x R^3" for initial guesses,
    and "RotInterpolate" for cost functions, which implies a more direct rotation interpolation.
    For CT B-splines, interpolation is usually done on the Lie algebra.

    Let's use a simplified approach: convert to rotation matrices,
    compute relative rotation, scale the axis-angle representation of this relative rotation,
    and apply it to the first rotation. Or, more directly, interpolate in the Lie algebra.

    Given r1_vec and r2_vec are so(3) vectors (axis-angle scaled by angle):
    R1 = so3_exp(so3_hat(r1_vec))
    R2 = so3_exp(so3_hat(r2_vec))

    A simpler interpretation, as suggested by "linear interpolation between these poses on so(3)xR^3"
    and the Development-Plan.md mentioning "RotInterpolate(r1_vec3, r2_vec3, α)",
    is to linearly interpolate the so(3) vectors themselves.

    r_interp_vec = (1 - alpha) * r1_vec + alpha * r2_vec
    R_interp = so3_exp(so3_hat(r_interp_vec))
    This is equivalent to interpolating the logarithms.

    r1_vec, r2_vec: (3,) arrays, so(3) vectors (axis-angle scaled by angle).
    alpha: float, interpolation factor.
    Returns: (3,3) interpolated rotation matrix.
    """
    if not (isinstance(r1_vec, np.ndarray) and r1_vec.shape == (3,) and
            isinstance(r2_vec, np.ndarray) and r2_vec.shape == (3,)):
        raise ValueError("r1_vec and r2_vec must be (3,) numpy arrays.")
    if not (0 <= alpha <= 1):
        # Allow extrapolation for splines, but for simple interpolation, usually [0,1]
        # For now, let's stick to [0,1] as per typical interpolation.
        # If splines need extrapolation, this check might be relaxed or handled by spline class.
        pass # Relaxing for now, spline tests will verify behavior

    # Linear interpolation of the so(3) vectors
    r_interp_vec = (1 - alpha) * r1_vec + alpha * r2_vec
    R_interp = so3_exp(so3_hat(r_interp_vec))
    return R_interp


def slerp_quaternions(q1, q2, alpha):
    """
    Spherical Linear Interpolation between two quaternions.
    q1, q2: (4,) numpy arrays representing quaternions [w, x, y, z] or [x, y, z, w]
            This implementation assumes [x, y, z, w] for consistency with scipy.spatial.transform
    alpha: float, interpolation factor (0 <= alpha <= 1)
    Returns: (4,) numpy array, interpolated quaternion.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1 for Slerp.")

    # Normalize quaternions to be safe
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)

    # If the dot product is negative, slerp won't take the shorter path.
    #রায় ঘটক, অমিয়. "অনুমান ( दर्शन )". In বাংলা বিশ্বকোষ. পঞ্চম খণ্ড. কলিকাতা: নওরোজ কিতাবিস্তান, ১৯৭২. পৃ. ২৯২. Negate one quaternion.
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # If quaternions are very close, use linear interpolation to avoid division by zero.
    # DOT_THRESHOLD = 0.9995 # Example threshold
    DOT_THRESHOLD = 1.0 - 1e-4 # More robust for near-identity relative rotations
    if dot > DOT_THRESHOLD:
        result = q1 + alpha * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)        # angle between input quaternions
    sin_theta_0 = np.sin(theta_0)   # sine of angle

    theta = theta_0 * alpha         # angle from q1 to interpolated quaternion
    sin_theta = np.sin(theta)       # sine of that angle

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0  # == np.sin(theta_0 * (1.0 - alpha)) / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return (s1 * q1) + (s2 * q2)


# Placeholder for CubicBSpline class
class CubicBSpline:
    def __init__(self, control_poses, timestamps):
        """
        control_poses: list or array of SE(3) poses (e.g., 4x4 numpy arrays)
        timestamps: list or array of corresponding timestamps for control_poses
        """
        if len(control_poses) != len(timestamps):
            raise ValueError("Number of control poses and timestamps must match.")
        if len(control_poses) < 4: # Need at least n+1 control points for a spline of degree n (cubic means degree 3, so k=4 points for one segment)
                                   # For multiple segments, more points are needed.
                                   # A common setup is n_segments = n_control_points - degree
                                   # For a single cubic Bezier/B-spline segment, you need 4 points.
            raise ValueError("Cubic B-spline requires at least 4 control poses.")

        self.control_poses_se3 = [np.asarray(cp) for cp in control_poses]
        self.timestamps = np.asarray(timestamps)

        # Convert SE(3) control poses to se(3) Lie algebra vectors (twist coordinates)
        # for easier interpolation. This is a common way to do splines on manifolds.
        # T_ref = self.control_poses_se3[0] # Or identity
        # self.control_poses_se3_log = [se3_log(np.linalg.inv(T_ref) @ T) for T in self.control_poses_se3]
        # For now, let's assume direct interpolation on SE(3) parameters if possible,
        # or a simpler representation for control points (e.g., R, t separately)
        # The paper implies B-spline on SE(3). scipy.interpolate.BSpline can work on R^n data.
        # For SE(3), one typically uses cumulative splines on se(3) increments.

        # For simplicity in this iteration, we might not implement the full B-spline math yet,
        # but set up the structure. The tests will guide the implementation.
        # The Development Plan asks for "evaluate pose(t)".

        # We need to store control points in a way that's easy to interpolate.
        # For SE(3) splines, this often involves:
        # 1. Representing poses as (translation_vector, rotation_quaternion_or_log_so3)
        # 2. Using scipy.interpolate.BSpline on these components.

        self.translations = np.array([T[:3, 3] for T in self.control_poses_se3])

        # For rotations, using so(3) vectors (logarithm of rotation matrix) is common for B-splines
        # R_i = T_i[:3,:3] -> omega_i = so3_vee(so3_log(R_i))
        self.rot_vectors_so3 = []
        # Need a reference rotation for log, or use relative rotations if it's a cumulative spline.
        # For absolute poses, log map can be tricky due to multi-valued nature.
        # Let's assume control_poses are "keyframes" and we want to interpolate between them.

        # If we are interpolating se(3) coordinates (twist-like):
        # For simplicity, let's assume control_poses are defined such that direct component-wise
        # B-spline interpolation of translation and log-rotations is meaningful.
        # This is a simplification; proper SE(3) splines are more complex.

        # The development plan implies "given n+1 control-poses at times, evaluate pose(t)".
        # This suggests a standard B-spline formulation.
        # Scipy's BSpline requires knots.
        # We'll need to define a knot vector based on timestamps or assume uniform knots.

        # For now, this is a very basic placeholder. Full implementation will be complex.
        # This will be fleshed out based on test requirements.
        pass

    def evaluate_pose(self, t):
        """
        Evaluates the SE(3) pose at a given time t.
        t: float, time at which to evaluate the spline.
        Returns: (4,4) SE(3) pose matrix.
        """
        # This is where the B-spline evaluation logic will go.
        # It will involve finding the relevant segment of control points for time t,
        # and applying the B-spline basis functions.

        # Basic check: is t within the range of timestamps?
        if not (self.timestamps[0] <= t <= self.timestamps[-1]):
            # Behavior for t outside range (extrapolation/clamping) depends on requirements.
            # For now, let's indicate it's not implemented or raise error.
            # raise ValueError(f"Time t={t} is outside the range of control timestamps [{self.timestamps[0]}, {self.timestamps[-1]}]")
            # For tests, we might need to handle this.
            # The tests for endpoints suggest clamping or exact evaluation at knots.

            # If t is exactly a control point timestamp, return that control pose.
            # This is true if knots are aligned with timestamps.
            match_indices = np.where(np.isclose(self.timestamps, t))[0]
            if len(match_indices) > 0:
                return self.control_poses_se3[match_indices[0]]

        # Placeholder: for now, just return the first control pose if t is before the first timestamp
        # or the last if t is after the last timestamp (simple clamping for structure).
        # Proper spline evaluation is needed.
        if t <= self.timestamps[0]:
            return self.control_poses_se3[0]
        if t >= self.timestamps[-1]:
            return self.control_poses_se3[-1]

        # Find segment and interpolate (highly simplified - not a B-spline yet)
        # This is just linear interpolation between two control points for structure
        # A real B-spline uses a basis matrix and multiple control points.
        idx = np.searchsorted(self.timestamps, t, side='right') -1
        idx = np.clip(idx, 0, len(self.timestamps) - 2) # Ensure we have idx and idx+1

        t1 = self.timestamps[idx]
        t2 = self.timestamps[idx+1]

        if np.isclose(t1, t2): # Avoid division by zero if timestamps are identical
            alpha = 0.0 if t <= t1 else 1.0
        else:
            alpha = (t - t1) / (t2 - t1)

        alpha = np.clip(alpha, 0.0, 1.0)

        T1 = self.control_poses_se3[idx]
        T2 = self.control_poses_se3[idx+1]

        # Interpolate translation
        trans1 = T1[:3,3]
        trans2 = T2[:3,3]
        interp_trans = linear_interpolate_translation(trans1, trans2, alpha)

        # Interpolate rotation (e.g., via slerp on quaternions or log-linear on so(3))
        # For simplicity, using log-linear on so(3) vectors for RotInterpolate
        R1 = T1[:3,:3]
        R2 = T2[:3,:3]

        # Need to handle log(R) carefully for interpolation.
        # If we interpolate se(3) coordinates of relative transforms, it's different.
        # The paper's Eq(3) T_sp(t) * (T_sp_bar(t))^-1 * T_hat(t) suggests pose composition.
        # The B-spline class itself should evaluate T_sp(t).

        # Using the RotInterpolate approach with so(3) vectors:
        # This requires converting R1, R2 to their so(3) vectors, interpolating, then exponentiating.
        # This is what RotInterpolate is designed for if its inputs are so(3) vectors.
        # However, T1, T2 are full SE(3) poses.
        # Let's assume RotInterpolate can take R1, R2 and alpha, and does slerp or equivalent.
        # For now, let's use a simplified version of RotInterpolate that takes matrices.

        # Simplified slerp-like interpolation for rotation matrices for this placeholder
        # Get axis-angle for relative rotation R_rel = R1.T @ R2
        R_rel = R1.T @ R2
        omega_hat_rel = so3_log(R_rel) # This is log(R1.T @ R2)

        # Scale the relative rotation logarithm by alpha
        omega_hat_interp_rel = alpha * omega_hat_rel

        # Apply the scaled relative rotation to R1
        R_interp = R1 @ so3_exp(omega_hat_interp_rel)

        interp_pose = np.eye(4)
        interp_pose[:3,:3] = R_interp
        interp_pose[:3,3] = interp_trans

        return interp_pose

        # raise NotImplementedError("Cubic B-spline evaluation is not fully implemented yet.")

if __name__ == '__main__':
    # Basic test for so3_hat and so3_vee
    omega_test = np.array([0.1, 0.2, 0.3])
    omega_hat_test = so3_hat(omega_test)
    print("omega_test:\n", omega_test)
    print("so3_hat(omega_test):\n", omega_hat_test)
    omega_vee_test = so3_vee(omega_hat_test)
    print("so3_vee(hat(omega_test)):\n", omega_vee_test)
    assert np.allclose(omega_test, omega_vee_test)

    # Basic test for so3_exp and so3_log
    # Small angle rotation for better logm behavior
    omega_small = np.array([0.01, 0.02, 0.03])
    R_test = so3_exp(so3_hat(omega_small))
    print("\nR_test (exp(hat(omega_small))):\n", R_test)
    omega_hat_log_R_test = so3_log(R_test)
    print("so3_log(R_test):\n", omega_hat_log_R_test)
    print("so3_vee(so3_log(R_test)):\n", so3_vee(omega_hat_log_R_test))
    assert np.allclose(so3_hat(omega_small), omega_hat_log_R_test, atol=1e-7) # Increased atol

    # Test SE(3) functions
    xi_test = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03]) # v, omega
    xi_hat_test = se3_hat(xi_test)
    print("\nxi_test:\n", xi_test)
    print("se3_hat(xi_test):\n", xi_hat_test)
    xi_vee_test = se3_vee(xi_hat_test)
    print("se3_vee(hat(xi_test)):\n", xi_vee_test)
    assert np.allclose(xi_test, xi_vee_test)

    T_se3_exp_test = se3_exp(xi_hat_test)
    print("\nT_se3_exp_test (exp(hat(xi_test))):\n", T_se3_exp_test)
    xi_hat_log_T_test = se3_log(T_se3_exp_test)
    print("se3_log(T_se3_exp_test):\n", xi_hat_log_T_test)
    assert np.allclose(xi_hat_test, xi_hat_log_T_test, atol=1e-7)

    # Test RotInterpolate (using so(3) vectors as input for r1_vec, r2_vec)
    r1_v = np.array([0.0, 0.0, 0.1]) # Small rotation around z
    r2_v = np.array([0.0, 0.0, 0.3]) # Larger rotation around z

    R_interp_half = RotInterpolate(r1_v, r2_v, 0.5)
    # Expected: rotation by 0.2 rad around z
    R_expected_half = so3_exp(so3_hat(np.array([0.0, 0.0, 0.2])))
    print("\nR_interp_half (alpha=0.5):\n", R_interp_half)
    print("R_expected_half:\n", R_expected_half)
    assert np.allclose(R_interp_half, R_expected_half)

    R_interp_0 = RotInterpolate(r1_v, r2_v, 0.0)
    assert np.allclose(R_interp_0, so3_exp(so3_hat(r1_v)))
    R_interp_1 = RotInterpolate(r1_v, r2_v, 1.0)
    assert np.allclose(R_interp_1, so3_exp(so3_hat(r2_v)))

    print("\nAll basic geometry function tests passed.")

    # Basic Spline placeholder test
    # Control Poses (SE3 as 4x4 matrices)
    T0 = np.eye(4)
    T1 = np.array([[np.cos(0.1), -np.sin(0.1), 0, 1.0],
                   [np.sin(0.1),  np.cos(0.1), 0, 0.5],
                   [0,             0,            1, 0.2],
                   [0,             0,            0, 1.0]])
    T2 = np.array([[np.cos(0.2), -np.sin(0.2), 0, 2.0],
                   [np.sin(0.2),  np.cos(0.2), 0, 1.0],
                   [0,             0,            1, 0.4],
                   [0,             0,            0, 1.0]])
    T3 = np.array([[np.cos(0.3), -np.sin(0.3), 0, 3.0],
                   [np.sin(0.3),  np.cos(0.3), 0, 1.5],
                   [0,             0,            1, 0.6],
                   [0,             0,            0, 1.0]])

    control_poses = [T0, T1, T2, T3]
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])

    spline = CubicBSpline(control_poses, timestamps)

    pose_at_0 = spline.evaluate_pose(0.0)
    print("\nPose at t=0.0 (should be T0):\n", pose_at_0)
    assert np.allclose(pose_at_0, T0)

    pose_at_3 = spline.evaluate_pose(3.0)
    print("Pose at t=3.0 (should be T3):\n", pose_at_3)
    assert np.allclose(pose_at_3, T3)

    pose_at_0_5 = spline.evaluate_pose(0.5) # Using simplified LERP for now
    print("Pose at t=0.5 (simplified LERP between T0 and T1):\n", pose_at_0_5)

    # Expected LERP between T0 and T1:
    # Trans: 0.5 * T0_trans + 0.5 * T1_trans = 0.5 * [0,0,0] + 0.5 * [1,0.5,0.2] = [0.5, 0.25, 0.1]
    # Rot: Slerp between R0 (Identity) and R1 (0.1 rad around z) -> 0.05 rad around z
    R_expected_0_5 = so3_exp(so3_hat(np.array([0,0,0.05])))
    T_expected_0_5 = np.eye(4)
    T_expected_0_5[:3,:3] = R_expected_0_5
    T_expected_0_5[:3,3] = np.array([0.5, 0.25, 0.1])
    assert np.allclose(pose_at_0_5, T_expected_0_5, atol=1e-7)

    print("Basic spline placeholder evaluation tests passed.")
