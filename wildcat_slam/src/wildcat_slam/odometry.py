import numpy as np
from .geometry import so3_exp, so3_hat, RotInterpolate, linear_interpolate_translation, se3_log, so3_vee

class IMUIntegrator:
    def __init__(self, initial_pose=None, initial_velocity=None, gravity=np.array([0,0,-9.81])):
        """
        Initializes the IMU Integrator.
        Args:
            initial_pose (np.ndarray, optional): Initial SE(3) pose (4x4 matrix). Defaults to identity.
            initial_velocity (np.ndarray, optional): Initial velocity (3D vector in world frame). Defaults to zero.
            gravity (np.ndarray, optional): Gravity vector in the world frame. Defaults to [0,0,-9.81].
        """
        self.current_pose = np.eye(4) if initial_pose is None else np.copy(initial_pose)
        self.current_velocity = np.zeros(3) if initial_velocity is None else np.copy(initial_velocity)
        self.gravity_w = np.asarray(gravity) # Gravity in world frame

        self.timestamps = []
        self.poses = [] # List to store SE(3) poses (4x4 matrices)
        self.velocities = [] # List to store velocities

        # Store initial state if provided
        if initial_pose is not None:
             # Assuming initial timestamp is 0 for now, or needs to be passed
            self.timestamps.append(0.0) # Placeholder for initial timestamp
            self.poses.append(self.current_pose)
            self.velocities.append(self.current_velocity)


    def integrate_measurement(self, acc_b, ang_vel_b, dt, current_timestamp):
        """
        Integrates a single IMU measurement.
        Args:
            acc_b (np.ndarray): Acceleration in body frame (3D vector).
            ang_vel_b (np.ndarray): Angular velocity in body frame (3D vector).
            dt (float): Time delta since the last measurement.
            current_timestamp (float): Timestamp of the current measurement.
        """
        if dt <= 0:
            # print(f"Warning: dt is {dt}, skipping integration step.")
            # If dt is zero, pose doesn't change, but we might want to record the state
            # For now, if dt is non-positive, we don't update but just record current state again
            # if not self.timestamps or self.timestamps[-1] != current_timestamp:
            # This logic might be complex if timestamps are not strictly increasing or dt is often zero.
            # Simplest: if dt is zero, state doesn't change from previous step.
            # However, the problem implies we are given samples and should produce poses.
            # Let's assume dt will generally be positive.
            # If the goal is to produce poses at 100Hz from IMU samples, dt is usually fixed.
            if not self.timestamps or self.timestamps[-1] < current_timestamp:
                 self.timestamps.append(current_timestamp)
                 self.poses.append(np.copy(self.current_pose))
                 self.velocities.append(np.copy(self.current_velocity))
            return

        R_wb = self.current_pose[:3, :3]  # Current rotation from body to world
        t_wb = self.current_pose[:3, 3]   # Current translation in world

        # --- Rotation update ---
        # Convert angular velocity to so(3) matrix
        ang_vel_hat = so3_hat(ang_vel_b)
        # Rotation increment using matrix exponential
        delta_R = so3_exp(ang_vel_hat * dt) # R_body_t+dt_from_body_t
        # Update orientation: R_wb(t+dt) = R_wb(t) * R_bb'(t, dt)
        R_wb_new = R_wb @ delta_R
        R_wb_new, _ = np.linalg.qr(R_wb_new) # Orthogonalize to prevent drift from SO(3)

        # --- Translation update (Euler step) ---
        # Acceleration in world frame: a_w = R_wb * a_b + g_w
        # (Note: IMU measures specific force, which includes gravity: a_imu = R_bw * (a_true - g_w) )
        # So, a_true = R_wb * a_imu + g_w
        acc_w = R_wb @ acc_b + self.gravity_w # This assumes acc_b is true acceleration without gravity effect
                                            # If acc_b is raw IMU accelerometer reading (specific force):
                                            # acc_w_true = R_wb @ acc_b - self.gravity_w (if g_w is +9.81 down)
                                            # Or R_wb @ acc_b + g_w (if g_w is -9.81 down, as per our default)
                                            # The Wildcat paper Eq(7) a_tau = R(tau)^T (w_a(tau) - g) + ...
                                            # implies w_a(tau) is true world acc. a_tau is IMU reading.
                                            # So, R(tau) a_tau = w_a(tau) - g => w_a(tau) = R(tau)a_tau + g
                                            # This matches our current formulation.

        # Update velocity: v(t+dt) = v(t) + a_w(t) * dt
        v_wb_new = self.current_velocity + acc_w * dt

        # Update position: p(t+dt) = p(t) + v(t) * dt  (Euler)
        # Or more accurately: p(t+dt) = p(t) + v(t)*dt + 0.5*a_w(t)*dt^2 (Second order Euler)
        # Or use new velocity for average: p(t) + (v(t) + v(t+dt))/2 * dt
        # For simple Euler, as per plan:
        t_wb_new = t_wb + self.current_velocity * dt
        # Using second order for position:
        # t_wb_new = t_wb + self.current_velocity * dt + 0.5 * acc_w * dt * dt


        # Update state
        self.current_pose = np.eye(4)
        self.current_pose[:3, :3] = R_wb_new
        self.current_pose[:3, 3] = t_wb_new
        self.current_velocity = v_wb_new

        # Store results
        self.timestamps.append(current_timestamp)
        self.poses.append(np.copy(self.current_pose))
        self.velocities.append(np.copy(self.current_velocity))

    def get_poses(self):
        return np.array(self.timestamps), np.array(self.poses)

    def get_velocities(self):
        return np.array(self.timestamps), np.array(self.velocities)


def initial_pose_interpolation(imu_timestamps, imu_poses_se3, query_times):
    """
    Interpolates SE(3) poses at query_times given a sequence of IMU poses.
    Uses RotInterpolate for rotation (linear interpolation of so(3) vectors)
    and linear interpolation for translation.

    Args:
        imu_timestamps (np.ndarray): (M,) array of timestamps for IMU poses. Must be sorted.
        imu_poses_se3 (np.ndarray): (M, 4, 4) array of SE(3) poses corresponding to imu_timestamps.
        query_times (np.ndarray): (Q,) array of times at which to interpolate poses.

    Returns:
        np.ndarray: (Q, 4, 4) array of interpolated SE(3) poses.
    """
    if not (isinstance(imu_timestamps, np.ndarray) and imu_timestamps.ndim == 1):
        raise ValueError("imu_timestamps must be a 1D numpy array.")
    if not (isinstance(imu_poses_se3, np.ndarray) and imu_poses_se3.ndim == 3 and imu_poses_se3.shape[1:] == (4,4)):
        raise ValueError("imu_poses_se3 must be an M x 4 x 4 numpy array.")
    if len(imu_timestamps) != len(imu_poses_se3):
        raise ValueError("imu_timestamps and imu_poses_se3 must have the same length M.")
    if len(imu_timestamps) < 2:
        raise ValueError("At least two IMU poses are required for interpolation.")
    if not np.all(np.diff(imu_timestamps) >= 0): # Allow for same timestamp if dt was 0
        # Should be strictly increasing if dt > 0 for all steps
        # For safety, if not strictly increasing, it can cause issues with searchsorted
        # Let's assume for now it's sorted.
        # Consider adding a check for strict monotonicity if needed.
        pass


    interpolated_poses_list = []

    # Extract translations and so(3) vectors from IMU poses
    imu_translations = imu_poses_se3[:, :3, 3]
    imu_rot_vectors_so3 = np.array([so3_vee(se3_log(T)[:3,:3]) for T in imu_poses_se3])
    # Note: se3_log(T)[:3,:3] gives the so(3) matrix part of the logarithm of T.
    # This is a common way to get the rotation vector for interpolation.

    for t_query in query_times:
        # Find bracketing IMU timestamps/poses
        # np.searchsorted returns insertion point to maintain order.
        # 'right' means if t_query matches an imu_timestamp, it gets the index of that timestamp.
        # So, idx will be such that imu_timestamps[idx-1] <= t_query < imu_timestamps[idx]
        idx = np.searchsorted(imu_timestamps, t_query, side='right')

        if idx == 0: # Query time is before or at the first IMU timestamp
            interp_pose = np.copy(imu_poses_se3[0])
        elif idx == len(imu_timestamps): # Query time is after or at the last IMU timestamp
            interp_pose = np.copy(imu_poses_se3[-1])
        else:
            # Perform interpolation
            t1 = imu_timestamps[idx-1]
            t2 = imu_timestamps[idx]

            T1_se3 = imu_poses_se3[idx-1]
            T2_se3 = imu_poses_se3[idx]

            r1_vec = imu_rot_vectors_so3[idx-1]
            r2_vec = imu_rot_vectors_so3[idx]

            trans1 = imu_translations[idx-1]
            trans2 = imu_translations[idx]

            if np.isclose(t1, t2): # Avoid division by zero if timestamps are identical
                alpha = 0.0 if t_query <= t1 else 1.0
            else:
                alpha = (t_query - t1) / (t2 - t1)

            # Ensure alpha is within [0,1] for interpolation between two points
            # (though RotInterpolate might handle extrapolation if needed by splines later)
            alpha = np.clip(alpha, 0.0, 1.0)

            # Interpolate translation
            interp_trans = linear_interpolate_translation(trans1, trans2, alpha)

            # Interpolate rotation using RotInterpolate with so(3) vectors
            interp_R = RotInterpolate(r1_vec, r2_vec, alpha)

            interp_pose = np.eye(4)
            interp_pose[:3,:3] = interp_R
            interp_pose[:3,3] = interp_trans

        interpolated_poses_list.append(interp_pose)

    return np.array(interpolated_poses_list)


def match_surfels(surfels_current_window, surfels_map, k=1, time_gap_thresh=0.1, dist_thresh=1.0):
    """
    Matches surfels from the current window to a map of surfels.
    Placeholder for now. Will use k-d tree on 7D descriptors.
    (x,y,z,nx,ny,nz,res)
    Mutual nearest-neighbours; reject close-time pairs.

    Args:
        surfels_current_window (np.ndarray): Structured array of surfels from current time window.
        surfels_map (np.ndarray): Structured array of surfels from the map/previous windows.
        k (int): Number of nearest neighbors to consider (for kNN).
        time_gap_thresh (float): Minimum time difference between matched surfels.
        dist_thresh (float): Maximum distance for a valid match in descriptor space.

    Returns:
        list: List of (index_in_current, index_in_map) tuples for matched pairs.
    """
    # This function will be complex and likely use scipy.spatial.KDTree
    # For Iteration 4, the plan is:
    # "Build kNN correspondences (e.g. sklearn.neighbors or KDTree from SciPy)"
    # "kd‐tree on 7‐D descriptors (x,y,z,nx,ny,nz,res), mutual nearest‐neighbours; reject close‐time pairs"

    # For now, returning an empty list as a placeholder.
    # Actual implementation will involve:
    # 1. Constructing 7D descriptors for both sets of surfels.
    #    Descriptor: [mean_x, mean_y, mean_z, normal_x, normal_y, normal_z, resolution]
    # 2. Building a KDTree on `surfels_map` descriptors.
    # 3. Querying the KDTree for each surfel in `surfels_current_window` to find k nearest neighbors.
    # 4. (Optional but good for robustness) Building a KDTree on `surfels_current_window` descriptors
    #    and querying for each map surfel to find its neighbors in the current window (for mutual NN check).
    # 5. Filtering matches based on distance threshold in descriptor space.
    # 6. Filtering matches based on time_gap_thresh (using 'timestamp_mean').
    # 7. Implementing mutual nearest neighbor check if k > 1 or for 1-NN cross-check.

    if not surfels_current_window.shape or not surfels_map.shape: # Check if arrays are empty
        return []

    # Example of how descriptors would be formed:
    # desc_current = np.hstack((
    #     surfels_current_window['mean'],
    #     surfels_current_window['normal'],
    #     surfels_current_window['resolution'][:, np.newaxis]
    # ))
    # desc_map = np.hstack((
    #     surfels_map['mean'],
    #     surfels_map['normal'],
    #     surfels_map['resolution'][:, np.newaxis]
    # ))

    # Actual KDTree and matching logic here...

    return [] # Placeholder


class OdometryWindow:
    def __init__(self, window_duration, imu_frequency, num_sample_poses):
        """
        Initializes the Local Continuous-Time Odometry Window.

        Args:
            window_duration (float): Duration of the sliding window in seconds.
            imu_frequency (float): Frequency of IMU measurements (e.g., 100 Hz).
            num_sample_poses (int): Number of discrete sample poses within the window for optimization.
        """
        self.window_duration = window_duration
        self.imu_frequency = imu_frequency
        self.num_sample_poses = num_sample_poses

        # Data storage for the current window
        self.imu_timestamps = [] # Timestamps of IMU measurements in the window
        self.imu_accelerometer_readings = [] # Raw accelerometer readings
        self.imu_gyroscope_readings = [] # Raw gyroscope readings

        self.surfels = np.array([], dtype=[ # Structured array for surfels in the window
            ('mean', '3f8'), ('normal', '3f8'), ('score', 'f8'),
            ('timestamp_mean', 'f8'), ('resolution', 'f8'), ('id', 'i4') # Added id for tracking
        ])
        self.surfel_matches = [] # List of (idx_surfel1, idx_surfel2) in self.surfels

        # Optimization variables and state
        # Sample poses (T_cor_i) represented by their Lie algebra vectors (r_cor, t_cor)
        # These are corrections to an initial trajectory estimate.
        # r_cor: (num_sample_poses, 3) for rotation corrections (so(3) vectors)
        # t_cor: (num_sample_poses, 3) for translation corrections
        self.r_cor_samples = np.zeros((num_sample_poses, 3))
        self.t_cor_samples = np.zeros((num_sample_poses, 3))

        # Timestamps for these sample poses, typically equidistant within the window
        self.sample_pose_timestamps = np.linspace(0, window_duration, num_sample_poses, endpoint=True) # Relative to window start

        # The actual trajectory (poses at IMU rate) estimated by the B-spline or interpolation from samples
        # This would be T_hat(t) from the paper.
        self.estimated_imu_poses_se3 = [] # List of 4x4 SE(3) poses at each self.imu_timestamps

        # IMU biases (can be part of the state to optimize if desired, or fixed)
        self.bias_acc = np.zeros(3)
        self.bias_gyro = np.zeros(3)

        # TODO: Add placeholders for Jacobians, residuals, Hessian/A^T A matrix, etc.
        # self.jacobian_imu = None
        # self.jacobian_surfel = None
        # self.residuals_imu = None
        # self.residuals_surfel = None
        # self.hessian_approx = None # A.T @ A for Gauss-Newton

    def add_imu_measurement(self, timestamp, acc_b, ang_vel_b):
        # TODO: Add measurement, manage window sliding (FIFO or similar)
        pass

    def add_surfels(self, new_surfels):
        # TODO: Add new surfels to the window, manage their lifetime
        pass

    def update_surfel_correspondences(self):
        # TODO: Call match_surfels based on current surfel positions (derived from estimated_imu_poses_se3)
        pass

    def _cauchy_m_estimator_weights(self, residuals, c=1.0):
        """Computes weights using Cauchy M-estimator."""
        weights = 1.0 / (1.0 + (residuals / c)**2)
        return weights

    def optimize_window(self, num_iterations=5, irls_iterations=3):
        """
        Performs iterative optimization of the current window.
        Alternates between:
        1. Assembling Jacobians and residuals.
        2. Solving the linear system (Gauss-Newton step).
        3. Updating sample poses (r_cor, t_cor).
        4. Updating the continuous trajectory (estimated_imu_poses_se3) using spline from sample poses.
        5. (Optionally) Re-calculating surfel correspondences if poses change significantly.
        Uses IRLS for robustness.
        """

        if not self.estimated_imu_poses_se3 or len(self.imu_timestamps) == 0:
            print("Warning: Not enough data to optimize window (IMU poses or timestamps missing).")
            return

        # Outer loop for Gauss-Newton iterations (or general non-linear solve steps)
        for iteration in range(num_iterations):
            # Inner loop for IRLS (re-weighting)
            # Initial weights are typically 1.0
            weights_imu = np.ones(len(self.imu_timestamps)) # Assuming one residual per IMU measurement for simplicity
            weights_surfel = np.ones(len(self.surfel_matches)) # Assuming one residual per surfel match

            for irls_iter in range(irls_iterations):
                # 1. Assemble Jacobians and residuals (J, r)
                #    This will involve:
                #    - IMU cost: f_tau_imu (Eq 9, 10, 11 from Wildcat paper)
                #      - Derivatives w.r.t. r_cor_i, t_cor_i via interpolated correction poses.
                #    - Surfel matching cost: f_s,s' (Eq 6)
                #      - Derivatives w.r.t. r_cor_i, t_cor_i.
                #    This step is complex and requires careful math for derivatives.

                # Placeholder for J and r assembly
                # Total number of optimization variables: num_sample_poses * 6 (3 for rot, 3 for trans)
                num_opt_vars = self.num_sample_poses * 6

                # Example: if we have N_imu residuals and N_surfel residuals
                # J_imu would be (N_imu, num_opt_vars)
                # J_surfel would be (N_surfel, num_opt_vars)
                # r_imu (N_imu,)
                # r_surfel (N_surfel,)

                # --- Placeholder: Simulate J and r ---
                # This needs to be replaced with actual calculations based on current state
                # (self.r_cor_samples, self.t_cor_samples, self.estimated_imu_poses_se3,
                #  self.surfels, self.surfel_matches, self.imu_accelerometer_readings, etc.)

                # Simulate some residuals and Jacobians for structure
                num_imu_residuals = len(self.imu_timestamps)
                # Each IMU measurement could contribute multiple residuals (accel, gyro, bias)
                # For now, assume one combined residual vector per IMU time for simplicity of J shape

                if num_imu_residuals == 0 and len(self.surfel_matches) == 0:
                    print(f"Iteration {iteration+1}, IRLS {irls_iter+1}: No residuals to process.")
                    continue # Skip to next outer iteration or break

                J_list = []
                r_list = []
                current_weights_list = []

                if num_imu_residuals > 0:
                    J_imu = np.random.rand(num_imu_residuals, num_opt_vars) * 0.1 # Small random Jacobians
                    r_imu = np.random.rand(num_imu_residuals) * 0.01      # Small random residuals
                    J_list.append(J_imu * np.sqrt(weights_imu)[:, np.newaxis])
                    r_list.append(r_imu * np.sqrt(weights_imu))
                    current_weights_list.append(weights_imu)


                if len(self.surfel_matches) > 0:
                    J_surfel = np.random.rand(len(self.surfel_matches), num_opt_vars) * 1.0
                    r_surfel = np.random.rand(len(self.surfel_matches)) * 0.1
                    J_list.append(J_surfel * np.sqrt(weights_surfel)[:, np.newaxis])
                    r_list.append(r_surfel * np.sqrt(weights_surfel))
                    current_weights_list.append(weights_surfel)

                if not J_list: # No residuals
                    print(f"Iteration {iteration+1}, IRLS {irls_iter+1}: No residuals to process after weighting.")
                    continue

                J = np.vstack(J_list)
                r = np.concatenate(r_list)
                # --- End Placeholder ---

                # 2. Solve the linear system: (J^T J) dx = -J^T r
                #    This is the normal equation for dx = -(A^T W A)^-1 A^T W r where A=J, r=residuals
                #    Or, if J is (J_sqrt_w) and r is (r_sqrt_w), then (J^T J) dx = -J^T r

                # Add damping (Levenberg-Marquardt like)
                lambda_lm = 1e-4
                A = J.T @ J + lambda_lm * np.eye(num_opt_vars)
                b = -J.T @ r

                try:
                    delta_x = np.linalg.solve(A, b) # delta_x contains updates for r_cor and t_cor
                except np.linalg.LinAlgError:
                    print(f"Warning: Singular matrix in iteration {iteration+1}, IRLS {irls_iter+1}. Skipping update.")
                    # Could try increasing lambda_lm or break
                    continue

                # 3. Update sample poses (r_cor, t_cor)
                # delta_x is (num_opt_vars,), reshape or split it for r_cor and t_cor updates
                delta_r_cor = delta_x[:self.num_sample_poses * 3].reshape(self.num_sample_poses, 3)
                delta_t_cor = delta_x[self.num_sample_poses * 3:].reshape(self.num_sample_poses, 3)

                self.r_cor_samples += delta_r_cor
                self.t_cor_samples += delta_t_cor

                # 4. Update the continuous trajectory (estimated_imu_poses_se3) using a B-spline
                #    or simpler interpolation from the updated (r_cor_samples, t_cor_samples).
                #    This step involves applying the corrections to a base trajectory.
                #    The Wildcat paper Eq(3): T_hat_new(t) = T_sp(t) * (T_sp_bar(t))^-1 * T_hat_old(t)
                #    where T_sp is from corrected sample poses, T_sp_bar is from uncorrected.
                #    This requires a base trajectory T_hat_old(t) and the ability to evaluate splines.
                #    For now, this is a placeholder.
                self._update_estimated_imu_trajectory_from_samples()


                # 5. Update IRLS weights based on new residuals (if not last IRLS iter)
                if irls_iter < irls_iterations - 1:
                    # Recompute residuals r_imu, r_surfel with updated poses
                    # For placeholder, we don't have true residuals, so we can't update weights meaningfully.
                    # In a real scenario:
                    # new_r_imu = compute_imu_residuals(...)
                    # new_r_surfel = compute_surfel_residuals(...)
                    # weights_imu = self._cauchy_m_estimator_weights(new_r_imu)
                    # weights_surfel = self._cauchy_m_estimator_weights(new_r_surfel)
                    pass # Placeholder for weight update

            # (Optional) Re-calculate surfel correspondences if poses changed significantly
            # self.update_surfel_correspondences()

            # Check for convergence (e.g., if delta_x is small)
            if np.linalg.norm(delta_x) < 1e-5:
                print(f"Converged in {iteration+1} iterations.")
                break

        # print(f"Optimization finished after {num_iterations} iterations.")
        pass

    def _update_estimated_imu_trajectory_from_samples(self):
        """
        Updates self.estimated_imu_poses_se3 based on the current
        self.r_cor_samples and self.t_cor_samples.
        This would typically involve:
        1. Having a base trajectory (e.g., from IMUIntegrator initially, or previous optimization).
        2. Defining SE(3) sample poses from this base + current corrections (r_cor, t_cor).
           T_sample_i = SE3_exp( [t_cor_i, r_cor_i] ) @ T_base_sample_i
        3. Fitting a B-spline to these corrected SE(3) sample poses.
        4. Evaluating this spline at self.imu_timestamps to get self.estimated_imu_poses_se3.

        For now, this is a placeholder. It needs the CubicBSpline class to be functional
        and a strategy for handling the base trajectory and applying corrections.
        """
        # Placeholder: Assume self.estimated_imu_poses_se3 was initialized somehow
        # and this function would refine it.
        # For the tests, we might need to mock this or provide a simple interpolation.

        # If we assume r_cor_samples and t_cor_samples are corrections to an Identity/zero initial state
        # for the sample poses themselves (not to a running trajectory yet):
        if len(self.imu_timestamps) == 0 or self.num_sample_poses == 0:
            return

        # Create SE(3) sample poses from r_cor and t_cor
        # These are T_cor_i in the paper's notation (correction poses)
        # The paper implies T_hat(t_i) <- T_cor_i * T_hat(t_i)
        # So r_cor, t_cor are parameters of T_cor_i.
        # Let's assume for now that r_cor, t_cor define absolute sample poses for simplicity
        # in a placeholder spline.

        sample_poses_for_spline = []
        for i in range(self.num_sample_poses):
            # T_corr_i = se3_exp(se3_hat( np.concatenate((self.t_cor_samples[i], self.r_cor_samples[i])) ))
            # This T_corr_i would then be applied to a base trajectory.
            # For a simple placeholder spline without a base trajectory:
            # Assume r_cor, t_cor directly define the pose T_i = (exp(r_cor_i_hat), t_cor_i)
            R_sample_i = so3_exp(so3_hat(self.r_cor_samples[i]))
            t_sample_i = self.t_cor_samples[i]
            T_sample_i = np.eye(4)
            T_sample_i[:3,:3] = R_sample_i
            T_sample_i[:3,3] = t_sample_i
            sample_poses_for_spline.append(T_sample_i)

        if len(sample_poses_for_spline) < 4 and len(sample_poses_for_spline) > 0 : # CubicBSpline needs at least 4
            # If fewer than 4 sample poses, cannot use CubicBSpline.
            # Fallback to simpler interpolation or error. For testing, this might be an issue.
            # For now, just use what we have, CubicBSpline will complain if not enough.
            # Or, if num_sample_poses < 4, this path shouldn't be taken or CubicBSpline should handle it.
            # Let's assume num_sample_poses will be >= 4 for spline usage.
            pass


        if self.num_sample_poses >= 4: # Crude check, CubicBSpline has its own internal check
            # This is where a proper B-spline (e.g., from geometry.py) would be used.
            # The current CubicBSpline in geometry.py is also a placeholder.
            # spline = CubicBSpline(control_poses=sample_poses_for_spline, timestamps=self.sample_pose_timestamps)
            # self.estimated_imu_poses_se3 = [spline.evaluate_pose(t) for t in self.imu_timestamps]

            # Using our simpler initial_pose_interpolation for now as a stand-in for spline evaluation
            # if self.imu_timestamps are the query_times.
            if len(self.imu_timestamps) > 0 and len(sample_poses_for_spline) >=2 :
                 self.estimated_imu_poses_se3 = initial_pose_interpolation(
                     self.sample_pose_timestamps, # Timestamps of r_cor, t_cor
                     np.array(sample_poses_for_spline), # SE(3) poses from r_cor, t_cor
                     self.imu_timestamps # Query at IMU rate
                 )
            elif len(sample_poses_for_spline) == 1: # Only one sample pose, replicate it
                 self.estimated_imu_poses_se3 = [sample_poses_for_spline[0] for _ in self.imu_timestamps]
            else:
                 self.estimated_imu_poses_se3 = [] # Not enough data to interpolate

        elif self.num_sample_poses > 0 : # e.g. 1 to 3 sample poses
            # Fallback if not enough for cubic spline: e.g. replicate first pose or simple linear
            if len(self.imu_timestamps) > 0:
                if len(sample_poses_for_spline) == 1:
                    self.estimated_imu_poses_se3 = [sample_poses_for_spline[0] for _ in self.imu_timestamps]
                elif len(sample_poses_for_spline) >= 2: # Use linear interpolation
                     self.estimated_imu_poses_se3 = initial_pose_interpolation(
                         self.sample_pose_timestamps,
                         np.array(sample_poses_for_spline),
                         self.imu_timestamps
                     )
                else: # No sample poses
                    self.estimated_imu_poses_se3 = []
            else: # No imu timestamps
                self.estimated_imu_poses_se3 = []
        else: # No sample poses
            self.estimated_imu_poses_se3 = []


        # Ensure it's a list of SE(3) matrices if not empty
        if not isinstance(self.estimated_imu_poses_se3, list) and self.estimated_imu_poses_se3.ndim == 3:
            self.estimated_imu_poses_se3 = [pose for pose in self.estimated_imu_poses_se3]
        elif isinstance(self.estimated_imu_poses_se3, list) and len(self.estimated_imu_poses_se3) > 0:
            if not isinstance(self.estimated_imu_poses_se3[0], np.ndarray): # If it became list of lists
                 self.estimated_imu_poses_se3 = [np.array(p) for p in self.estimated_imu_poses_se3]


    def get_optimized_poses(self):
        """Returns the final estimated IMU poses after optimization."""
        return np.array(self.imu_timestamps), np.array(self.estimated_imu_poses_se3)

# Placeholder for OdometryRunner which would manage OdometryWindow instances over time
# class OdometryRunner:
#     def __init__(self, args):
#         self.args = args
#         self.current_window = OdometryWindow(...)
#         # ...
#
#     def process_frame(self, lvx_frame_data):
#         # Extract points, timestamps, IMU data from lvx_frame_data
#         # Add to self.current_window
#         # Call self.current_window.optimize_window()
#         # If window needs to slide, create new window, pass state
#         pass
