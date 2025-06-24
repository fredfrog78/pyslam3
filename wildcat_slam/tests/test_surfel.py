import pytest
import numpy as np
from wildcat_slam.surfel import extract_surfels

# Helper to generate points on a plane with optional noise
def generate_planar_points(num_points=1000, plane_normal=np.array([0.0,0.0,1.0]), plane_center=np.array([0.0,0.0,0.0]), size=5.0, noise_std=0.01):
    plane_normal = np.asarray(plane_normal, dtype=float)
    plane_center = np.asarray(plane_center, dtype=float)
    plane_normal /= np.linalg.norm(plane_normal)

    # Create basis vectors for the plane
    if np.allclose(plane_normal, [0,0,1]) or np.allclose(plane_normal, [0,0,-1]):
        v1 = np.array([1.0,0.0,0.0])
    else:
        # Choose a non-collinear vector to cross with plane_normal
        temp_vec = np.array([0.0,0.0,1.0])
        if np.allclose(np.abs(np.dot(plane_normal, temp_vec)), 1.0): # if plane_normal is along Z axis
            temp_vec = np.array([0.0,1.0,0.0]) # use Y axis instead

        v1 = np.cross(plane_normal, temp_vec)
        v1 /= np.linalg.norm(v1)

    v2 = np.cross(plane_normal, v1) # v2 will be float due to v1 and plane_normal being float
    # v2 /= np.linalg.norm(v2) # Already normalized if plane_normal and v1 are unit and orthogonal

    # Generate random points in the plane's coordinate system
    rand_coords1 = (np.random.rand(num_points) - 0.5) * size # Range: -size/2 to size/2
    rand_coords2 = (np.random.rand(num_points) - 0.5) * size

    points = plane_center + rand_coords1[:, np.newaxis] * v1 + rand_coords2[:, np.newaxis] * v2

    if noise_std > 0:
        # Add noise along the plane normal and also a bit within the plane
        noise_normal = np.random.normal(0, noise_std, (num_points, 1)) * plane_normal
        noise_inplane = np.random.normal(0, noise_std/2, (num_points, 3)) # General noise
        # Project inplane noise to be actually inplane (optional, small noise might be fine)
        # For simplicity, just add general small noise to coordinates
        points += noise_normal
        points += np.random.normal(0, noise_std, (num_points, 3)) * 0.1 # Smaller general noise

    timestamps = np.sort(np.random.rand(num_points) * 1.0) # Timestamps within a 1s window
    return points, timestamps

def test_extract_surfels_synthetic_plane():
    plane_normal_gt = np.array([0.0,0.0,1.0])
    test_voxel_size = 5.0

    # Center points well within a single voxel
    points, timestamps = generate_planar_points(
        num_points=500,
        plane_normal=plane_normal_gt,
        plane_center=[test_voxel_size * 0.5, test_voxel_size * 0.5, test_voxel_size * 0.5],
        size=1.0, # Point spread (size) is much smaller than voxel_size
        noise_std=0.005
    )

    surfels = extract_surfels(points, timestamps, voxel_size=test_voxel_size, planarity_thresh=0.8, min_points_per_surfel=10)

    if len(surfels) != 1:
        voxel_indices_dbg = np.floor(points / test_voxel_size).astype(int)
        unique_voxels_dbg = np.unique(voxel_indices_dbg, axis=0)
        print(f"Debug test_extract_surfels_synthetic_plane: Expected 1 surfel, got {len(surfels)}. Unique voxels: {unique_voxels_dbg}")

    assert len(surfels) == 1, f"Expected 1 surfel, got {len(surfels)}."
    surfel = surfels[0]

    # Check mean position (should be close to plane_center if points are centered, or their actual mean)
    expected_mean = np.mean(points, axis=0)
    assert np.allclose(surfel['mean'], expected_mean, atol=0.1), f"Mean position mismatch: {surfel['mean']} vs {expected_mean}"

    # Check normal (should be close to plane_normal_gt or its negative)
    # Dot product should be close to 1 or -1
    dot_product = np.dot(surfel['normal'], plane_normal_gt)
    assert np.isclose(np.abs(dot_product), 1.0, atol=0.05), \
        f"Normal orientation mismatch. Surfel normal: {surfel['normal']}, Expected: {plane_normal_gt}, Dot: {dot_product}"

    # Check planarity score (should be high for a good plane)
    assert surfel['score'] > 0.8, f"Planarity score too low: {surfel['score']}" # Matches threshold

    # Check resolution
    assert np.isclose(surfel['resolution'], test_voxel_size)

    # Check timestamp
    assert np.isclose(surfel['timestamp_mean'], np.mean(timestamps), atol=0.1)


def test_planarity_rejection_spherical_points():
    # Generate points on a sphere - these should have low planarity
    num_sphere_points = 500
    phi = np.linspace(0, np.pi, int(np.sqrt(num_sphere_points/2)))
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(num_sphere_points/2))*2)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    sphere_points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T * 2.0 # Radius 2
    sphere_points += np.random.normal(0, 0.01, sphere_points.shape) # Small noise

    timestamps = np.sort(np.random.rand(len(sphere_points)) * 0.5)

    # Use a voxel size that captures a significant portion of the sphere
    # If voxel size is too small, tiny patches might appear planar.
    voxel_s = 5.0

    # High planarity threshold to reject non-planar structures
    high_planarity_thresh = 0.8
    surfels = extract_surfels(sphere_points, timestamps, voxel_size=voxel_s,
                              planarity_thresh=high_planarity_thresh, min_points_per_surfel=20)

    # Expect no surfels, or if any are found, their score should have been low (but filtered by thresh)
    assert len(surfels) == 0, \
        f"Expected 0 surfels for spherical points with high planarity threshold, got {len(surfels)}. " \
        f"If surfels found, scores: {[s['score'] for s in surfels]}"

    # Test with low planarity threshold, expect surfels but with low scores
    low_planarity_thresh = 0.01
    surfels_low_thresh = extract_surfels(sphere_points, timestamps, voxel_size=voxel_s,
                                         planarity_thresh=low_planarity_thresh, min_points_per_surfel=20)
    if len(surfels_low_thresh) > 0:
        # If any surfels are extracted from a sphere, their planarity score should be low.
        # This depends on the voxel size relative to sphere curvature.
        # For a large voxel encompassing many sphere points, the overall structure is not planar.
        assert surfels_low_thresh[0]['score'] < 0.5, \
            f"Planarity score for spherical points should be low, got {surfels_low_thresh[0]['score']}"
    # else:
        # print("Warning: No surfels found for sphere even with low threshold. Might be due to min_points or voxelization.")


def test_voxel_clustering_counts():
    voxel_s = 1.0
    # Create points that will fall into distinct voxels, forming planar patches in each

    # Voxel 1: Centered around (0.5, 0.5, 0.5)
    points_v1, ts_v1 = generate_planar_points(50, plane_center=np.array([0.25,0.25,0.25]), size=0.4, noise_std=0.001)

    # Voxel 2: Centered around (1.5, 0.5, 0.5)
    points_v2, ts_v2 = generate_planar_points(50, plane_center=np.array([1.25,0.25,0.25]), size=0.4, noise_std=0.001)

    # Voxel 3 (non-planar, should be rejected or low score)
    # These points are arranged linearly, should have low planarity
    points_v3 = np.array([[2.1,0.1,0.1], [2.2,0.1,0.1], [2.3,0.1,0.1], [2.4,0.1,0.1], [2.5,0.1,0.1], [2.6,0.1,0.1]]) * voxel_s
    ts_v3 = np.linspace(0, 0.1, len(points_v3)) + np.max(ts_v2) + 0.1

    all_points = np.vstack((points_v1, points_v2, points_v3))
    all_timestamps = np.concatenate((ts_v1, ts_v2, ts_v3))

    # Expect 2 surfels with high planarity from v1 and v2
    # The points in v3 should form a cluster, but its planarity score should be low.
    surfels = extract_surfels(all_points, all_timestamps, voxel_size=voxel_s,
                              planarity_thresh=0.7, min_points_per_surfel=5)

    assert len(surfels) == 2, f"Expected 2 planar surfels, got {len(surfels)}"

    # Check means to identify which surfel corresponds to which voxel's points
    means = np.array([s['mean'] for s in surfels])

    # Expected means for the planar patches
    mean_v1_exp = np.mean(points_v1, axis=0)
    mean_v2_exp = np.mean(points_v2, axis=0)

    # Check if one surfel corresponds to points_v1 and another to points_v2
    found_v1 = any(np.allclose(m, mean_v1_exp, atol=0.1) for m in means)
    found_v2 = any(np.allclose(m, mean_v2_exp, atol=0.1) for m in means)

    assert found_v1, "Surfel corresponding to voxel 1 not found or mean is off."
    assert found_v2, "Surfel corresponding to voxel 2 not found or mean is off."

    for surfel in surfels:
        assert surfel['score'] >= 0.7

def test_min_points_rejection():
    points, timestamps = generate_planar_points(num_points=5, size=1.0, noise_std=0.001) # Only 5 points

    # min_points_per_surfel = 10, should reject
    surfels = extract_surfels(points, timestamps, voxel_size=2.0, planarity_thresh=0.1, min_points_per_surfel=10)
    assert len(surfels) == 0, "Should reject surfel due to insufficient points (5 < 10)"

    # min_points_per_surfel = 5. Let's use num_points=10 and min_points_per_surfel=6
    # to ensure a more robust plane for testing acceptance.
    num_test_points = 10
    min_req = 6
    test_voxel_size = 4.0
    # Center points well within a single voxel
    # Points spread over size=0.5, centered at [test_voxel_size/2, ...] ensures they are in voxel [0,0,0]
    points_accept, timestamps_accept = generate_planar_points(
        num_points=num_test_points,
        plane_center=[test_voxel_size * 0.5, test_voxel_size * 0.5, test_voxel_size * 0.5],
        size=0.5, # Small spread relative to voxel_size
        noise_std=0.01 # Moderate noise
    )

    # Threshold should be high enough to accept a decent plane
    test_thresh = 0.5
    surfels_accept = extract_surfels(
        points_accept, timestamps_accept,
        voxel_size=test_voxel_size,
        planarity_thresh=test_thresh,
        min_points_per_surfel=min_req
    )

    if len(surfels_accept) != 1:
        # Debug: Check voxel distribution if assertion fails
        voxel_indices_dbg = np.floor(points_accept / test_voxel_size).astype(int)
        unique_voxels_dbg = np.unique(voxel_indices_dbg, axis=0)
        print(f"Debug test_min_points_rejection: Points generated for acceptance resulted in {len(unique_voxels_dbg)} unique voxels: {unique_voxels_dbg}")
        if len(unique_voxels_dbg) == 1 and len(points_accept) >= min_req :
             cov_matrix_dbg = np.cov(points_accept, rowvar=False)
             print(f"  Debug Covariance Matrix:\n{cov_matrix_dbg}")
             if not (np.isnan(cov_matrix_dbg).any() or np.isinf(cov_matrix_dbg).any()):
                try:
                    eigenvalues_dbg, _ = np.linalg.eigh(cov_matrix_dbg)
                    print(f"  Debug Eigenvalues: {eigenvalues_dbg}")
                    eig_s, eig_m, eig_l = np.sort(eigenvalues_dbg)
                    eig_s = max(eig_s, 1e-12) # clamp before sqrt
                    eig_m = max(eig_m, 1e-12)
                    eig_l = max(eig_l, 1e-12)
                    score_dbg = (np.sqrt(eig_m) - np.sqrt(eig_s)) / np.sqrt(eig_l)
                    print(f"  Debug Calculated Score: {score_dbg}")
                except np.linalg.LinAlgError as e:
                    print(f"  Debug: LinAlgError during eigh: {e}")


    assert len(surfels_accept) == 1, f"Should form surfel with {num_test_points} points (min_req={min_req}). Got {len(surfels_accept)}."


def test_empty_input():
    points = np.empty((0,3))
    timestamps = np.empty((0,))
    surfels = extract_surfels(points, timestamps, voxel_size=1.0, planarity_thresh=0.1, min_points_per_surfel=5)
    assert len(surfels) == 0
    assert surfels.dtype is not None # Should return empty structured array

def test_collinear_points_low_planarity():
    # Collinear points should have low planarity score. Eigenvalues e.g., [0, small, large]
    # The planarity score (sqrt(e_mid) - sqrt(e_small)) / sqrt(e_large)
    # If e_small is ~0, this becomes sqrt(e_mid / e_large).
    # For collinear points, e_mid should also be ~0. So score ~0.
    points = np.array([
        [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0], [6,0,0]
    ])
    timestamps = np.arange(len(points)) * 0.1

    # Use a low threshold to actually extract the surfel and check its score
    surfels = extract_surfels(points, timestamps, voxel_size=10.0, planarity_thresh=0.001, min_points_per_surfel=3)

    # With planarity_thresh = 0.001, a score of 0 (expected for collinear) should be rejected.
    assert len(surfels) == 0, f"Expected 0 surfels for collinear points with thresh=0.001, got {len(surfels)}. Score was likely 0."

    # If we wanted to test the score itself, we'd use planarity_thresh=0.0
    surfels_check_score = extract_surfels(points, timestamps, voxel_size=10.0, planarity_thresh=0.0, min_points_per_surfel=3)
    assert len(surfels_check_score) == 1, "Expected 1 surfel with planarity_thresh=0.0"
    assert np.isclose(surfels_check_score[0]['score'], 0.0), \
        f"Planarity score for collinear points should be 0, got {surfels_check_score[0]['score']}"


def test_coplanar_points_degenerate_covariance():
    # Test with points that are coplanar but might lead to singular covariance in some dimensions
    # e.g., all points on x-y plane, z=0. Covariance along z will be 0.
    # Smallest eigenvalue will be 0.
    points = np.array([
        [0,0,0], [1,0,0], [0,1,0], [1,1,0],
        [0.5,0.5,0], [0.2,0.7,0], [0.8,0.3,0]
    ])
    timestamps = np.arange(len(points)) * 0.1
    surfels = extract_surfels(points, timestamps, voxel_size=2.0, planarity_thresh=0.5, min_points_per_surfel=3)
    assert len(surfels) == 1
    assert surfels[0]['score'] > 0.5 # Should be highly planar
    # Normal should be [0,0,1] or [0,0,-1]
    assert np.allclose(np.abs(surfels[0]['normal']), [0,0,1], atol=1e-6)

def test_identical_points_in_voxel():
    # All points in the voxel are identical. Covariance matrix will be all zeros.
    # Eigenvalues will be all zeros. Planarity score should be handled (e.g. 0).
    points = np.array([
        [1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1]
    ])
    timestamps = np.arange(len(points)) * 0.1
    surfels = extract_surfels(points, timestamps, voxel_size=2.0, planarity_thresh=0.01, min_points_per_surfel=3)

    # Current implementation might skip this due to cov_matrix being all zeros or eigh issues.
    # If a surfel is formed, its score should be low or zero as it's not a plane.
    # The `extract_surfels` has a check: `if eig_l == 0: planarity_score = 0`
    # And `if np.isnan(cov_matrix).any()` check.
    # For identical points, cov_matrix is all zeros. eig_s,m,l are all zeros.
    # sqrt_eig_l will be zero, so planarity_score becomes 0.
    # So it should be rejected by any reasonable planarity_thresh > 0.
    if len(surfels) > 0:
         assert surfels[0]['score'] < 0.01, f"Score for identical points should be 0 or very low, got {surfels[0]['score']}"
    else:
        # This is also acceptable if the surfel creation was skipped due to degenerate covariance
        pass
    # With planarity_thresh = 0.01, a score of 0 should lead to rejection.
    assert len(surfels) == 0, "Identical points should not form a planar surfel above threshold 0.01"
