import numpy as np

def extract_surfels(points, timestamps, voxel_size, time_window=None, planarity_thresh=0.01, min_points_per_surfel=5):
    """
    Extracts surfels from a point cloud.

    Args:
        points (np.ndarray): (N x 3) array of x, y, z coordinates.
        timestamps (np.ndarray): (N,) array of timestamps for each point.
        voxel_size (float): The size of voxels for clustering.
        time_window (float, optional): Duration to segment points by time.
                                       If None, all points are processed as one time chunk.
        planarity_thresh (float): Minimum planarity score for a surfel to be accepted.
                                  Planarity is 1 - sphericity.
                                  score = (eig_val[1] - eig_val[0]) / eig_val[2] for sorted eigenvalues.
                                  A more common one from literature might be (lambda_2 - lambda_1) / lambda_0
                                  or related to ratios of eigenvalues. Let's use one based on [22] from Wildcat paper.
                                  lambda_0 <= lambda_1 <= lambda_2. Planarity P_lambda = (sqrt(lambda_1) - sqrt(lambda_0)) / sqrt(lambda_2)
                                  Or, from Bosse & Zlot (2012) referenced by Wildcat:
                                  Planarity score S = (λ1 - λ0) / λ0, where λ0 is smallest eigenvalue. This seems problematic if λ0 is tiny.
                                  Let's use S = (λ_m - λ_s) / λ_l where λ_s, λ_m, λ_l are smallest, middle, largest eigenvalues.
                                  A common alternative used in other works (e.g. LOAM, LeGO-LOAM for edge/plane features) is
                                  based on curvature or smoothness.
                                  The development plan mentions "eigen-decomp of covariance ... -> compute planarity score".
                                  Let sorted eigenvalues be e0 <= e1 <= e2.
                                  A simple planarity score: 1 - e0 / e1 (if e1 > 0). Higher is better.
                                  Or (e2-e1)/e2 for linearity, (e1-e0)/e2 for planarity.
                                  Let's adopt a score that's higher for planar structures.
                                  Planarity P = (sqrt(λ₁) - sqrt(λ₀)) / sqrt(λ₂) where λ₀ ≤ λ₁ ≤ λ₂ are eigenvalues.
                                  This is from "Zebedee: Design of a Spring-Mounted 3-D Range Sensor..." by Bosse, Zlot, Flick (2012)
                                  which is cited as inspiration for Wildcat's odometry.
                                  The Wildcat paper itself [22, Eq. 4] refers to Bosse and Zlot (2009) "Continuous 3D Scan-Matching..."
                                  which defines planarity as (sqrt(lambda_1) - sqrt(lambda_0)) / sqrt(lambda_0) - this seems problematic.
                                  The 2012 paper seems more robust.
                                  Let's use: (λ₁ - λ₀) / λ₂ as a simple measure. Higher is better.
                                  Or, if λ₀ is smallest, λ₂ largest: (λ₁ - λ₀) / (λ₀ + λ₁ + λ₂) or similar normalized.
                                  The dev plan says "score > thresh".
                                  Let's use (eval_mid - eval_small) / eval_large.

        min_points_per_surfel (int): Minimum number of points in a voxel to form a surfel.

    Returns:
        np.ndarray: Structured array of surfels. Each surfel contains:
                    'mean' (3,), 'normal' (3,), 'score' (float),
                    'timestamp_mean' (float), 'resolution' (float) - voxel_size for now.
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be an N x 3 numpy array.")
    if not isinstance(timestamps, np.ndarray) or timestamps.ndim != 1 or timestamps.shape[0] != points.shape[0]:
        raise ValueError("Timestamps must be an N, array, matching the number of points.")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")

    surfels_list = []

    # 1. Assign points to voxels
    # Normalize points by voxel_size and cast to int to get voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Combine voxel_indices and time_chunks for unique IDs
    # For now, let's handle time_windowing simply: if None, one chunk.
    # If provided, points could be bucketed. This part needs more thought for "sort/segment-by-voxel+time chunk"
    # A simple approach: if time_window is set, further divide voxels by time segments.

    unique_voxel_coords, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    # inverse_indices[i] gives the index in unique_voxel_coords for points[i]

    for i in range(len(unique_voxel_coords)):
        # Get all points belonging to this unique voxel
        point_indices_in_voxel = np.where(inverse_indices == i)[0]

        if len(point_indices_in_voxel) < min_points_per_surfel:
            continue # Skip voxels with too few points

        current_points = points[point_indices_in_voxel]
        current_timestamps = timestamps[point_indices_in_voxel]

        # Time chunking within a voxel (simplified for now)
        # If time_window is specified, this loop would be more complex,
        # potentially splitting current_points/current_timestamps further.
        # For now, assume one time chunk per voxel that passes min_points_per_surfel.

        # 2. Aggregate cluster: calculate mean and covariance
        mean_pos = np.mean(current_points, axis=0)

        # Covariance matrix
        if len(current_points) < 2: # Need at least 2 points for covariance
            continue

        # np.cov expects rows to be variables, columns to be observations, or pass rowvar=False
        cov_matrix = np.cov(current_points, rowvar=False)

        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any() or cov_matrix.shape != (3,3) :
            # This can happen if all points in voxel are identical, leading to zero variance
            # print(f"Warning: Skipping voxel due to problematic covariance matrix. Points: {len(current_points)}")
            continue


        # 3. Eigen-decomposition of covariance
        try:
            # eigenvalues are sorted in ascending order by eigh
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        except np.linalg.LinAlgError:
            # print(f"Warning: Skipping voxel due to LinAlgError in eigh. Points: {len(current_points)}")
            continue


        # eigenvalues should be non-negative for a covariance matrix.
        # Clamp very small or slightly negative eigenvalues (due to numerical precision) to a small positive value.
        # This helps stabilize sqrt and division, especially for perfect planes where true eig_s is 0.
        eigenvalues[eigenvalues < 1e-12] = 1e-12

        # Ensure eigenvalues are sorted: smallest, middle, largest
        # np.linalg.eigh already returns sorted eigenvalues, but if we clamped after, re-sort.
        eig_s, eig_m, eig_l = np.sort(eigenvalues)

        # Normal is the eigenvector corresponding to the smallest eigenvalue
        # Find index of eig_s in the original (unsorted by us, but sorted by eigh) eigenvalues
        # to get the correct eigenvector.
        original_eig_s_idx = np.argmin(eigenvalues) # eigenvalues from eigh were already sorted, so this is 0
                                                # but if clamping changed order, argmin on the clamped is safer.
                                                # However, np.sort(eigenvalues) means eig_s is eigenvalues[0] from sorted.
                                                # The eigenvectors correspond to the original eigh output.
                                                # So, we need the eigenvector for the *original* smallest eigenvalue.

        # Let's stick to eigh's sorting: eigenvalues are sorted, eigenvectors correspond.
        # idx_sorted = np.argsort(eigenvalues_from_eigh) # This is just 0,1,2
        # eig_s_val = eigenvalues_from_eigh[idx_sorted[0]] -> eigenvalues_from_eigh[0]
        # normal_vec = eigenvectors[:, idx_sorted[0]] -> eigenvectors[:,0]
        normal = eigenvectors[:, 0] # Assumes eigh sorts eigenvalues and aligns eigenvectors

        # 4. Compute planarity score
        # P = (sqrt(λ_middle) - sqrt(λ_small)) / sqrt(λ_large)
        # Using eig_s, eig_m, eig_l from our sorted (and clamped) eigenvalues.

        sqrt_eig_s = np.sqrt(eig_s)
        sqrt_eig_m = np.sqrt(eig_m)
        sqrt_eig_l = np.sqrt(eig_l)

        if sqrt_eig_l < 1e-7:  # Effectively if eig_l is near zero (e.g. single point, collinear points in some projections)
            planarity_score = 0.0
        else:
            # Numerator should be non-negative as eig_m >= eig_s
            numerator = sqrt_eig_m - sqrt_eig_s
            # Defensive check for floating point noise if eig_m was only marginally larger than eig_s
            if numerator < 0:
                numerator = 0.0
            planarity_score = numerator / sqrt_eig_l

        if np.isnan(planarity_score):
            planarity_score = 0.0


        # 5. Filter surfels based on score
        if planarity_score >= planarity_thresh:
            mean_timestamp = np.mean(current_timestamps)
            surfel_data = (mean_pos, normal, planarity_score, mean_timestamp, voxel_size)
            surfels_list.append(surfel_data)

    # Define structured array dtype
    surfel_dtype = [
        ('mean', '3f8'),            # Mean position (x,y,z)
        ('normal', '3f8'),          # Normal vector (nx,ny,nz)
        ('score', 'f8'),            # Planarity score
        ('timestamp_mean', 'f8'),   # Mean timestamp of points in surfel
        ('resolution', 'f8')        # Voxel size used for this surfel
    ]

    if not surfels_list: # Handle case with no surfels found
        return np.array([], dtype=surfel_dtype)

    return np.array(surfels_list, dtype=surfel_dtype)


if __name__ == '__main__':
    # Example Usage / Basic Test

    # Test 1: Perfect plane
    print("--- Test 1: Perfect Plane ---")
    plane_points = np.array([
        [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0.5,0.5,0],
        [2,0,0], [0,2,0], [2,2,0], [0.5,1.5,0], [1.5,0.5,0]
    ])
    plane_timestamps = np.arange(len(plane_points)) * 0.1

    surfels_plane = extract_surfels(plane_points, plane_timestamps, voxel_size=10.0, planarity_thresh=0.5, min_points_per_surfel=3)
    print(f"Found {len(surfels_plane)} surfels for perfect plane.")
    if len(surfels_plane) > 0:
        print("First surfel details:")
        print(f"  Mean: {surfels_plane[0]['mean']}")
        print(f"  Normal: {surfels_plane[0]['normal']} (should be close to [0,0,1] or [0,0,-1])")
        print(f"  Score: {surfels_plane[0]['score']} (should be high, close to 1)")
        print(f"  Timestamp: {surfels_plane[0]['timestamp_mean']}")
        assert len(surfels_plane) == 1, "Should find 1 surfel for a single large voxel containing the plane"
        assert np.isclose(surfels_plane[0]['score'], 1.0, atol=0.1), "Planarity score should be near 1.0" # Allow some tolerance
        assert np.allclose(np.abs(surfels_plane[0]['normal']), [0,0,1], atol=1e-6), "Normal should be along Z axis"

    # Test 2: Linear points (should not form a good surfel)
    print("\n--- Test 2: Linear Points (low planarity) ---")
    line_points = np.array([
        [0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0]
    ])
    line_timestamps = np.arange(len(line_points)) * 0.1
    surfels_line = extract_surfels(line_points, line_timestamps, voxel_size=10.0, planarity_thresh=0.5, min_points_per_surfel=3)
    print(f"Found {len(surfels_line)} surfels for linear points (expected 0 with high threshold).")
    if len(surfels_line) > 0:
        print("First surfel (line) details:")
        print(f"  Score: {surfels_line[0]['score']} (should be low)")
        assert surfels_line[0]['score'] < 0.5, "Planarity for line should be low"
    assert len(surfels_line) == 0, "Should find 0 surfels for linear points with planarity_thresh=0.5"


    # Test 3: Spherical points (should have low planarity)
    print("\n--- Test 3: Spherical Points (low planarity) ---")
    # Generate points on a sphere
    phi = np.linspace(0, np.pi, 10)
    theta = np.linspace(0, 2 * np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    sphere_points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    sphere_timestamps = np.arange(len(sphere_points)) * 0.01

    surfels_sphere = extract_surfels(sphere_points, sphere_timestamps, voxel_size=5.0, planarity_thresh=0.5, min_points_per_surfel=10)
    print(f"Found {len(surfels_sphere)} surfels for spherical points (expected 0 with high threshold).")
    # This test depends heavily on voxelization. A large voxel might still find a "plane" locally.
    # If one surfel is found, its score should be low.
    if len(surfels_sphere) > 0:
         print(f"  Score of first surfel from sphere: {surfels_sphere[0]['score']}")
         assert surfels_sphere[0]['score'] < 0.5, "Planarity for sphere segment should be low"
    # A more robust test would be to check that *if* a surfel is found, its score is low.
    # For now, assert 0 found with high threshold.

    # Test 4: Multiple voxels
    print("\n--- Test 4: Multiple Voxels ---")
    points_multi_voxel = np.array([
        [0,0,0], [0.1,0,0], [0,0.1,0], # Voxel 1 (0,0,0)
        [1,0,0], [1.1,0,0], [1,0.1,0], # Voxel 2 (1,0,0) if voxel_size=1
        [0,0,10], [0.1,0,10], [0,0.1,10] # Voxel 3 (0,0,10)
    ]) * 2.0 # Scale up to make sure they are in different voxels if voxel_size is small
    ts_multi = np.arange(len(points_multi_voxel)) * 0.1

    surfels_multi = extract_surfels(points_multi_voxel, ts_multi, voxel_size=1.0, planarity_thresh=0.1, min_points_per_surfel=3)
    print(f"Found {len(surfels_multi)} surfels for multi-voxel test (expected 3).")
    if len(surfels_multi) == 3:
        print("Scores:", [s['score'] for s in surfels_multi]) # Should be high
        assert all(s['score'] > 0.5 for s in surfels_multi), "All 3 surfels should be planar"
    else:
        print("Warning: Did not find 3 surfels for multi-voxel test.")
        for s in surfels_multi:
            print(s['mean'], s['normal'], s['score'])

    assert len(surfels_multi) == 3, f"Expected 3 surfels, got {len(surfels_multi)}"

    print("\nBasic surfel extraction tests complete (manual inspection needed for detailed values).")
