import pytest
import numpy as np
from wildcat_slam.odometry import IMUIntegrator, initial_pose_interpolation, match_surfels
from wildcat_slam.geometry import so3_exp, so3_hat, se3_log, so3_vee, se3_exp, se3_hat

# --- IMUIntegrator Tests ---

def test_imu_integrator_stationary():
    """Test IMUIntegrator with stationary IMU data (only gravity)."""
    gravity_vec = np.array([0, 0, -9.81])
    integrator = IMUIntegrator(gravity=gravity_vec)

    # IMU readings for a stationary sensor:
    # Accelerometer measures reaction to gravity, so it points upwards.
    # If gravity is (0,0,-g), acc_b should be (0,0,+g) in body frame if aligned with world.
    # If R_wb is Identity, acc_b = R_bw * (0 - g_w) = Identity * -g_w = -g_w
    # The formula in integrator is: acc_w_true = R_wb @ acc_b_raw + g_w
    # For stationary, acc_w_true should be 0.
    # So, 0 = R_wb @ acc_b_raw + g_w  => R_wb @ acc_b_raw = -g_w
    # If R_wb = Identity, then acc_b_raw = -g_w.
    # So, if g_w = [0,0,-9.81], then acc_b_raw should be [0,0,9.81]

    acc_b_stationary = -gravity_vec # If body frame is aligned with world, acc reading is -gravity
    ang_vel_b_stationary = np.array([0.0, 0.0, 0.0])

    dt = 0.01  # 100 Hz
    total_time = 1.0  # Simulate for 1 second
    num_steps = int(total_time / dt)

    current_ts = 0.0
    # Add initial pose at t=0
    integrator.timestamps.append(current_ts)
    integrator.poses.append(np.copy(integrator.current_pose))
    integrator.velocities.append(np.copy(integrator.current_velocity))


    for i in range(num_steps):
        current_ts += dt
        integrator.integrate_measurement(acc_b_stationary, ang_vel_b_stationary, dt, current_ts)

    timestamps, poses = integrator.get_poses()
    _, velocities = integrator.get_velocities()

    final_pose = poses[-1]
    final_velocity = velocities[-1]

    # Final position should be close to initial (0,0,0)
    assert np.allclose(final_pose[:3, 3], [0, 0, 0], atol=1e-2), \
        f"Final position error: {final_pose[:3, 3]}"

    # Final orientation should be close to initial (Identity)
    assert np.allclose(final_pose[:3, :3], np.eye(3), atol=1e-3), \
        f"Final orientation error: \n{final_pose[:3, :3]}"

    # Final velocity should be close to initial (0,0,0)
    # acc_w = R_wb @ acc_b + g_w = I @ (-g_w) + g_w = 0. So velocity should remain 0.
    assert np.allclose(final_velocity, [0,0,0], atol=1e-2), \
        f"Final velocity error: {final_velocity}"


def test_imu_integrator_no_measurements():
    integrator = IMUIntegrator()
    # Call integrate with dt=0 or no calls
    integrator.integrate_measurement(np.zeros(3), np.zeros(3), 0.0, 0.0) # dt=0
    ts, poses = integrator.get_poses()
    assert len(ts) == 1 # Should only have initial pose if dt=0 or no integration
    assert np.allclose(poses[0], np.eye(4))

    # Test with a non-zero timestamp for dt=0 call
    integrator = IMUIntegrator()
    integrator.timestamps.append(0.0)
    integrator.poses.append(np.eye(4))
    integrator.velocities.append(np.zeros(3))
    integrator.integrate_measurement(np.zeros(3),np.zeros(3), 0.0, 0.1)
    ts, poses = integrator.get_poses()
    assert len(ts) == 2
    assert ts[1] == 0.1
    assert np.allclose(poses[1], np.eye(4)) # Pose should not change


# --- initial_pose_interpolation Tests ---

def test_initial_pose_interpolation_basic():
    # Timestamps for IMU poses
    imu_timestamps = np.array([0.0, 1.0])

    # Known IMU poses (SE(3) matrices)
    T0 = np.eye(4)
    # T1: translation (1,0,0), rotation 90 deg around Z
    R1 = so3_exp(so3_hat(np.array([0, 0, np.pi/2])))
    T1 = np.eye(4)
    T1[:3,:3] = R1
    T1[:3,3] = np.array([1.0, 0.0, 0.0])

    imu_poses_se3 = np.array([T0, T1])

    # Query time at midpoint
    query_times = np.array([0.5])

    interpolated_poses = initial_pose_interpolation(imu_timestamps, imu_poses_se3, query_times)
    assert interpolated_poses.shape == (1, 4, 4)
    mid_pose = interpolated_poses[0]

    # Expected midpoint:
    # Translation: linear interpolation -> (0.5, 0, 0)
    expected_trans_mid = np.array([0.5, 0.0, 0.0])
    assert np.allclose(mid_pose[:3,3], expected_trans_mid, atol=1e-7)

    # Rotation: RotInterpolate applied to so(3) vectors of T0 and T1
    # log(T0_rot) = log(I) = [0,0,0]
    # log(T1_rot) = [0,0,pi/2]
    # Interpolated so(3) vector = 0.5*[0,0,0] + 0.5*[0,0,pi/2] = [0,0,pi/4]
    # Expected rotation matrix: exp(hat([0,0,pi/4])) (45 deg around Z)
    expected_R_mid = so3_exp(so3_hat(np.array([0, 0, np.pi/4])))
    assert np.allclose(mid_pose[:3,:3], expected_R_mid, atol=1e-7)


def test_initial_pose_interpolation_endpoints_and_outside():
    imu_timestamps = np.array([1.0, 2.0, 3.0])
    T_identity = np.eye(4)
    # Dummy poses, structure is what matters for this test
    imu_poses_se3 = np.array([
        T_identity,
        se3_exp(se3_hat(np.array([0.1,0.2,0.3, 0.01,0.02,0.03]))),
        se3_exp(se3_hat(np.array([0.2,0.4,0.6, 0.02,0.04,0.06])))
    ])

    # Query times at exact IMU timestamps, before first, and after last
    query_times = np.array([0.5, 1.0, 2.0, 3.0, 3.5])

    interpolated_poses = initial_pose_interpolation(imu_timestamps, imu_poses_se3, query_times)
    assert interpolated_poses.shape == (len(query_times), 4, 4)

    # Query before first: should be clamped to first pose
    assert np.allclose(interpolated_poses[0], imu_poses_se3[0], atol=1e-7)
    # Query at first IMU timestamp
    assert np.allclose(interpolated_poses[1], imu_poses_se3[0], atol=1e-7)
    # Query at second IMU timestamp
    assert np.allclose(interpolated_poses[2], imu_poses_se3[1], atol=1e-7)
    # Query at third IMU timestamp
    assert np.allclose(interpolated_poses[3], imu_poses_se3[2], atol=1e-7)
    # Query after last: should be clamped to last pose
    assert np.allclose(interpolated_poses[4], imu_poses_se3[2], atol=1e-7)


def test_initial_pose_interpolation_input_validation():
    with pytest.raises(ValueError, match="At least two IMU poses"):
        initial_pose_interpolation(np.array([0.0]), np.array([np.eye(4)]), np.array([0.5]))

    with pytest.raises(ValueError, match="imu_timestamps must be a 1D numpy array"):
        initial_pose_interpolation(np.array([[0.0],[1.0]]), np.array([np.eye(4),np.eye(4)]), np.array([0.5]))

    with pytest.raises(ValueError, match="imu_poses_se3 must be an M x 4 x 4"):
        initial_pose_interpolation(np.array([0.0,1.0]), np.array([np.eye(3),np.eye(3)]), np.array([0.5]))

    with pytest.raises(ValueError, match="must have the same length M"):
        initial_pose_interpolation(np.array([0.0,1.0,2.0]), np.array([np.eye(4),np.eye(4)]), np.array([0.5]))


# --- match_surfels Tests ---

def test_match_surfels_logic():
    surfel_dtype = [
        ('mean', '3f8'), ('normal', '3f8'),
        ('timestamp_mean', 'f8'), ('resolution', 'f8')
        # Not including 'score' or 'id' as match_surfels doesn't use them
    ]

    # --- Test Data Setup ---
    # Surfel A0: Base for matching
    # Surfel A1: Another distinct surfel in set A
    # Surfel A2: Outlier in set A, not expected to match anything
    # Surfel A3: Will have a time-violating partner in B
    # Surfel A4: Will have a dist-violating partner in B

    surfels_A_list = [
        # A0: mean, normal, ts, res
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.1, 0.5),  # A0
        ([5.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.2, 0.5),  # A1
        ([10.0, 10.0, 10.0], [1.0, 0.0, 0.0], 0.3, 0.8), # A2 (outlier descriptor)
        ([0.0, 5.0, 0.0], [0.0, 1.0, 0.0], 0.4, 0.5),  # A3 (for time test)
        ([5.0, 5.0, 0.0], [0.0, 0.0, 1.0], 0.5, 0.5),  # A4 (for dist test)
    ]
    surfels_A = np.array(surfels_A_list, dtype=surfel_dtype)

    # Surfel B0: Exact match for A0 (descriptor and time gap ok)
    # Surfel B1: Close descriptor to A1, should match if A1 is its NN too.
    # Surfel B2: Outlier, far from all A's
    # Surfel B3: Exact descriptor match for A3, but timestamp too close
    # Surfel B4: Exact descriptor match for A4, but artificially make its stored descriptor far for dist_thresh test (or test query result)
    # Surfel B5: One-way NN to A0, but A0's NN is B0 (to test mutual NN)

    surfels_B_list = [
        # B0: matches A0
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.1, 0.5),
        # B1: matches A1 (make it slightly different for realism, but still NN)
        ([5.1, 0.1, 0.0], [0.0, 0.0, 1.0], 1.2, 0.5),
        # B2: outlier descriptor
        ([-10.0, -10.0, -10.0], [0.0, 1.0, 0.0], 1.3, 0.7),
        # B3: time violation with A3 (ts_A3=0.4, ts_B3=0.45, gap=0.05)
        ([0.0, 5.0, 0.0], [0.0, 1.0, 0.0], 0.45, 0.5),
        # B4: dist violation with A4 (descriptor for A4 is [5,5,0, 0,0,1, 0.5])
        #     Let B4 be identical geom+ts but we'll test dist_A_to_B[A4_idx]
        #     This is implicitly tested by dist_thresh. If B4 is identical to A4, dist=0.
        #     To test dist_thresh, we need a pair that *would* be mutual NN,
        #     but their descriptor distance (from tree.query) is > dist_thresh.
        #     Let's make B4 slightly further from A4 in descriptor space.
        ([5.0, 5.0, 0.0], [0.1, 0.0, 0.9], 1.5, 0.5), # Normal slightly different -> diff descriptor
        # B5: Not a mutual NN with any A. Say A0 is NN of B5, but B0 is NN of A0.
        ([0.01, 0.01, 0.01], [0.0, 0.0, 1.0], 1.6, 0.5)
    ]
    surfels_B = np.array(surfels_B_list, dtype=surfel_dtype)

    descriptor_dist_thresh = 0.5 # Max L2 distance in 7D descriptor space
    time_gap_thresh = 0.1       # Min time diff: abs(ts_A - ts_B) >= time_gap_thresh

    # Expected matches:
    # (A0, B0): Descriptors are identical (dist=0). Time gap |0.1-1.1|=1.0 >= 0.1. OK.
    # (A1, B1): Descriptors are close. A1=[5,0,0,0,0,1,0.5], B1=[5.1,0.1,0,0,0,1,0.5]. Dist = sqrt(0.1^2+0.1^2) = sqrt(0.02) approx 0.14.
    #            This is < dist_thresh=0.5. Time gap |0.2-1.2|=1.0 >= 0.1. OK.
    #            (Need to ensure mutual NN condition holds for this setup).

    # A2 (outlier) should not match.
    # (A3, B3): Descriptors identical. Time gap |0.4-0.45|=0.05 < 0.1. REJECT.
    # (A4, B4): A4=[5,5,0,0,0,1,0.5], B4=[5,5,0,0.1,0,0.9,0.5].
    #            desc_A4 = [5,5,0,0,0,1,0.5]
    #            desc_B4 = [5,5,0,0.1,0,0.9,0.5]
    #            Diff vec = [0,0,0, 0.1,0,-0.1,0]. Dist = sqrt(0.1^2 + (-0.1)^2) = sqrt(0.02) approx 0.14.
    #            This is < dist_thresh=0.5. Time gap |0.5-1.5|=1.0 >=0.1. This would be a match *if* mutual NN.
    #            Let's make B4's descriptor normal component further to fail dist_thresh:
    #            B4_mod: normal=[0.5,0,sqrt(1-0.5^2)=0.866]. desc_B4_mod=[5,5,0, 0.5,0,0.866, 0.5]
    #            Diff vec = [0,0,0, 0.5,0,-0.134,0]. Dist = sqrt(0.5^2 + (-0.134)^2) = sqrt(0.25 + 0.017) = sqrt(0.267) approx 0.517.
    #            This is > dist_thresh=0.5. So (A4, B4_mod) should be rejected by distance.
    #            Let's update B4 in surfels_B_list for this.
    surfels_B_list[4] = ([5.0, 5.0, 0.0], [0.5, 0.0, np.sqrt(1-0.5**2)], 1.5, 0.5) # B4 updated for dist test
    surfels_B = np.array(surfels_B_list, dtype=surfel_dtype)


    # A0 index=0, A1 index=1
    # B0 index=0, B1 index=1
    expected_matches = [(0,0), (1,1)] # Assuming A0-B0 and A1-B1 are mutual and pass filters

    # Call the function
    # Original signature: match_surfels(surfels_current_window, surfels_map, k=1, time_gap_thresh=0.1, dist_thresh=1.0)
    # My plan used descriptor_dist_thresh. The function uses dist_thresh. Sticking to dist_thresh.
    actual_matches = match_surfels(surfels_A, surfels_B, time_gap_thresh=time_gap_thresh, dist_thresh=descriptor_dist_thresh)

    # Convert to sets for comparison to ignore order
    assert set(actual_matches) == set(expected_matches), \
        f"Expected {expected_matches}, got {actual_matches}"

    # Test case for non-mutual NN:
    # A_nm = ([0,0,0],[0,0,1],0.1,0.5) -> desc_A_nm = [0,0,0,0,0,1,0.5]
    # B_nm1 = ([0.01,0,0],[0,0,1],1.1,0.5) -> desc_B_nm1, close to A_nm
    # B_nm2 = ([10,0,0],[0,0,1],1.2,0.5) -> desc_B_nm2, far from A_nm
    # Tree_B query for A_nm's desc -> B_nm1
    # Tree_A query for B_nm1's desc -> A_nm (Mutual) -> Match
    #
    # A_nm = ([0,0,0],[0,0,1],0.1,0.5)
    # A_other = ([-0.01,0,0],[0,0,1],0.05,0.5) # Slightly closer to B_nm1 than A_nm is
    # B_nm1 = ([0.005,0,0],[0,0,1],1.1,0.5) # B_nm1 is between A_nm and A_other
    # Tree_B query for A_nm's desc -> B_nm1
    # Tree_A query for B_nm1's desc -> A_other (Not mutual with A_nm) -> No match for (A_nm, B_nm1)
    surfels_A_nonmutual = np.array([
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.1, 0.5),    # A_nm (idx 0)
        ([-0.01, 0.0, 0.0], [0.0, 0.0, 1.0], 0.05, 0.5) # A_other (idx 1)
    ], dtype=surfel_dtype)
    surfels_B_nonmutual = np.array([
        ([0.005, 0.0, 0.0], [0.0, 0.0, 1.0], 1.1, 0.5)   # B_nm1 (idx 0)
    ], dtype=surfel_dtype)

    # desc(A_nm) = [0,0,0,...]
    # desc(A_other) = [-0.01,0,0,...]
    # desc(B_nm1) = [0.005,0,0,...]
    # dist(desc(A_nm), desc(B_nm1)) = 0.005
    # dist(desc(A_other), desc(B_nm1)) = |-0.01 - 0.005| = 0.015
    # So, NN of desc(B_nm1) in A is A_nm.
    # NN of desc(A_nm) in B is B_nm1. This IS a mutual match.
    # My example for non-mutual was flawed.
    # Let's try:
    # A0: (0,0,0), ts=0.1
    # A1: (10,0,0), ts=0.2
    # B0: (0.1,0,0), ts=1.1 (NN to A0)
    # B1: (0.2,0,0), ts=1.2 (NN to A0 if A0's desc is [0,0,0] and B0 is [0.1,0,0], B1 is [0.2,0,0])
    #                   A0's NN is B0.
    #                   B0's NN is A0. -> (A0,B0) is mutual.
    #
    # To break mutuality:
    # A0=(0,0,0)
    # A1=(0.05,0,0) # A point very close to A0
    # B0=(0.02,0,0) # Target point
    # desc(A0)=[0,...], desc(A1)=[0.05,...], desc(B0)=[0.02,...]
    # Query B0 against A_descriptors: dist(B0,A0)=0.02, dist(B0,A1)=0.03. So NN of B0 in A is A0.
    # Query A0 against B_descriptors (only B0): NN of A0 in B is B0.
    # This is still mutual.
    # The mutual check is: for A[i] query B -> B[j]. Then for B[j] query A -> A[k]. Match if k==i.
    # This is already handled by the implementation.

    # Test with empty inputs (already in placeholder, let's ensure it's here)
    empty_surfels = np.array([], dtype=surfel_dtype)
    assert match_surfels(empty_surfels, surfels_A, time_gap_thresh, descriptor_dist_thresh) == []
    assert match_surfels(surfels_A, empty_surfels, time_gap_thresh, descriptor_dist_thresh) == []
    assert match_surfels(empty_surfels, empty_surfels, time_gap_thresh, descriptor_dist_thresh) == []

    # Test case where one set is empty after descriptor creation (e.g. if surfels had no 'mean' field)
    # This is implicitly covered if surfels_X.shape[0] == 0 leads to desc_X.shape[0] == 0.
    # The code already checks `if desc_A.shape[0] == 0 or desc_B.shape[0] == 0: return []`
    # and `if not surfels_current_window.shape or not surfels_map.shape: return []`
    # So this should be fine.

    # Test all surfels in A match all surfels in B (e.g. A=B)
    surfels_C = np.array([
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.1, 0.5),
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.2, 0.5),
    ], dtype=surfel_dtype)
    surfels_D = np.array([ # Timestamps shifted to pass time_gap_thresh
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.1, 0.5),
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.2, 0.5),
    ], dtype=surfel_dtype)
    expected_CD_matches = [(0,0), (1,1)]
    actual_CD_matches = match_surfels(surfels_C, surfels_D, time_gap_thresh=0.1, dist_thresh=0.01)
    assert set(actual_CD_matches) == set(expected_CD_matches)


# --- OdometryWindow Tests ---
from wildcat_slam.odometry import OdometryWindow

def test_odometry_window_init():
    window_duration = 2.0 # seconds
    imu_frequency = 100.0 # Hz
    num_samples = 10 # Number of sample poses for optimization

    odom_window = OdometryWindow(window_duration, imu_frequency, num_samples)

    assert odom_window.window_duration == window_duration
    assert odom_window.imu_frequency == imu_frequency
    assert odom_window.num_sample_poses == num_samples
    assert odom_window.r_cor_samples.shape == (num_samples, 3)
    assert odom_window.t_cor_samples.shape == (num_samples, 3)
    assert np.all(odom_window.r_cor_samples == 0)
    assert np.all(odom_window.t_cor_samples == 0)
    assert len(odom_window.sample_pose_timestamps) == num_samples
    assert np.isclose(odom_window.sample_pose_timestamps[0], 0)
    assert np.isclose(odom_window.sample_pose_timestamps[-1], window_duration)

def test_odometry_window_cauchy_weights():
    odom_window = OdometryWindow(1.0, 100.0, 5) # Dummy params
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    c = 1.0
    weights_expected = 1.0 / (1.0 + (residuals / c)**2)
    # weights_expected = [1/(1+4), 1/(1+1), 1/1, 1/(1+1), 1/(1+4)]
    #                  = [0.2, 0.5, 1.0, 0.5, 0.2]

    weights_actual = odom_window._cauchy_m_estimator_weights(residuals, c)
    assert np.allclose(weights_actual, weights_expected)

def test_odometry_window_optimize_runs_placeholder(monkeypatch):
    """
    Tests that optimize_window can run with its current placeholder structure.
    It won't actually optimize correctly, but it shouldn't crash.
    We'll check if r_cor_samples and t_cor_samples change from zero.
    """
    window_duration = 1.0
    imu_frequency = 100.0
    num_samples = 5 # Must be >= 4 for current _update_estimated_imu_trajectory_from_samples logic if using CubicBSpline like path

    odom_window = OdometryWindow(window_duration, imu_frequency, num_samples)

    # Need some dummy data for the optimization to proceed
    # Provide minimal estimated_imu_poses_se3 and imu_timestamps
    # These are used by the placeholder _update_estimated_imu_trajectory_from_samples
    # and checked at the start of optimize_window
    num_imu_points = int(window_duration * imu_frequency)
    if num_imu_points == 0: num_imu_points = 2 # Ensure at least some points

    odom_window.imu_timestamps = np.linspace(0, window_duration, num_imu_points)
    # Initial estimated_imu_poses_se3 can be all identity
    odom_window.estimated_imu_poses_se3 = [np.eye(4) for _ in range(num_imu_points)]

    # The placeholder optimize_window uses random Jacobians and residuals.
    # So, r_cor_samples and t_cor_samples should change from their initial zero values.
    initial_r_cor = np.copy(odom_window.r_cor_samples)
    initial_t_cor = np.copy(odom_window.t_cor_samples)

    odom_window.optimize_window(num_iterations=1, irls_iterations=1) # Run minimal iterations

    # Check if samples have changed (due to random J, r in placeholder)
    # If num_imu_residuals was 0 (e.g., if imu_timestamps was empty), they might not change.
    # The test setup ensures imu_timestamps is not empty.
    assert not np.allclose(odom_window.r_cor_samples, initial_r_cor), "r_cor_samples did not change"
    assert not np.allclose(odom_window.t_cor_samples, initial_t_cor), "t_cor_samples did not change"

    # Test with no residuals (e.g. no IMU timestamps for the J_imu part)
    odom_window_no_imu_ts = OdometryWindow(window_duration, imu_frequency, num_samples)
    odom_window_no_imu_ts.estimated_imu_poses_se3 = [np.eye(4)] # Still need this to pass initial check
    # odom_window_no_imu_ts.imu_timestamps is empty

    initial_r_no_ts = np.copy(odom_window_no_imu_ts.r_cor_samples)
    initial_t_no_ts = np.copy(odom_window_no_imu_ts.t_cor_samples)

    # optimize_window should print "No residuals to process" and not change samples
    # or "Not enough data to optimize window" if estimated_imu_poses_se3 is also empty.
    # Let's ensure estimated_imu_poses_se3 is non-empty but imu_timestamps is.
    # The first check in optimize_window is:
    # if not self.estimated_imu_poses_se3 or len(self.imu_timestamps) == 0:
    # So if imu_timestamps is empty, it will print warning and return.

    # To properly test the "no residuals" path inside the loop, we need imu_timestamps
    # but then need to ensure the random J/r generation path for IMU is skipped.
    # This is hard with current placeholder.
    # For now, let's assume the "Warning: Not enough data..." path is covered if imu_timestamps is empty.

    # What if there are IMU timestamps but no surfel matches (which is the default)?
    # The random J_imu, r_imu should still be generated and cause an update.
    # This is covered by the first part of the test.

    # Test the "Not enough data to optimize" path:
    odom_window_no_data = OdometryWindow(window_duration, imu_frequency, num_samples)
    # estimated_imu_poses_se3 is empty, imu_timestamps is empty
    r_before = np.copy(odom_window_no_data.r_cor_samples)
    t_before = np.copy(odom_window_no_data.t_cor_samples)
    odom_window_no_data.optimize_window(num_iterations=1, irls_iterations=1)
    # Should not have changed because it returns early
    assert np.allclose(odom_window_no_data.r_cor_samples, r_before)
    assert np.allclose(odom_window_no_data.t_cor_samples, t_before)
