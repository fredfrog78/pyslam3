import pytest
import numpy as np
from wildcat_slam.pose_graph import Submap, SubmapCollector, GraphBuilder, PoseGraphEdge
from wildcat_slam.geometry import so3_exp, so3_hat, se3_exp, se3_hat, se3_log, se3_vee

# --- Submap Tests ---
def test_submap_creation():
    submap_id = 0
    timestamp = 100.0
    odometry_pose = np.eye(4)
    odometry_pose[0,3] = 1.0 # tx=1

    # Dummy surfels structured array
    surfel_dtype = [('mean', '3f8'), ('normal', '3f8'), ('score', 'f8'),
                    ('timestamp_mean', 'f8'), ('resolution', 'f8')]
    surfels_data = np.array([([0,0,0], [0,0,1], 0.9, 100.0, 0.5)], dtype=surfel_dtype)

    gravity_vec = np.array([0,0,-9.81])

    sm = Submap(submap_id, timestamp, odometry_pose, surfels_data, gravity_vec)

    assert sm.id == submap_id
    assert sm.timestamp == timestamp
    assert np.allclose(sm.odometry_pose, odometry_pose)
    assert len(sm.surfels) == 1
    assert np.allclose(sm.surfels[0]['mean'], [0,0,0])
    assert np.allclose(sm.gravity_vector_local, gravity_vec)

    with pytest.raises(ValueError): # Invalid pose shape
        Submap(1, 101.0, np.eye(3), surfels_data)


# --- SubmapCollector Tests ---
@pytest.fixture
def surfel_array_fixture():
    surfel_dtype = [('mean', '3f8'), ('normal', '3f8'), ('score', 'f8'),
                    ('timestamp_mean', 'f8'), ('resolution', 'f8')]
    return np.array([([0.1,0.2,0.3], [0,0,1], 0.95, 100.1, 0.2)], dtype=surfel_dtype)

def test_submap_collector_add_submap(surfel_array_fixture):
    collector = SubmapCollector()
    assert collector.next_submap_id == 0
    assert len(collector.submaps) == 0

    ts1 = 10.0
    pose1 = np.eye(4)
    pose1[0,3] = 1.0
    sm1 = collector.add_submap(ts1, pose1, surfel_array_fixture)

    assert collector.next_submap_id == 1
    assert len(collector.submaps) == 1
    assert sm1.id == 0
    assert sm1.timestamp == ts1
    assert np.allclose(sm1.odometry_pose, pose1)
    # Compare structured arrays field by field
    assert len(sm1.surfels) == len(surfel_array_fixture)
    for field_name in surfel_array_fixture.dtype.names:
        if isinstance(sm1.surfels[0][field_name], np.ndarray):
            assert np.allclose(sm1.surfels[field_name], surfel_array_fixture[field_name]), f"Field {field_name} mismatch"
        else:
            assert np.array_equal(sm1.surfels[field_name], surfel_array_fixture[field_name]), f"Field {field_name} mismatch"
    assert collector.get_submap_by_id(0) == sm1

    ts2 = 20.0
    pose2 = np.eye(4)
    pose2[0,3] = 2.0
    sm2 = collector.add_submap(ts2, pose2, surfel_array_fixture, gravity_vector_local=np.array([0,0.1, -9.8]))

    assert collector.next_submap_id == 2
    assert len(collector.submaps) == 2
    assert sm2.id == 1
    assert np.allclose(sm2.gravity_vector_local, [0,0.1,-9.8])
    assert collector.get_submap_by_id(1) == sm2
    assert collector.get_submap_by_id(99) is None


# --- GraphBuilder Tests ---
@pytest.fixture
def sample_submaps(surfel_array_fixture):
    sm0 = Submap(0, 10.0, np.eye(4), surfel_array_fixture)

    T1_se3 = se3_exp(se3_hat(np.array([1.0, 0.0, 0.0, 0, 0, np.pi/4]))) # dx=1, rot_z=45deg
    sm1 = Submap(1, 20.0, T1_se3, surfel_array_fixture)

    # T2 relative to T1: dx=1 (local frame of T1), rot_y=30deg (local frame of T1)
    # T_1_2_rel = se3_exp(se3_hat(np.array([1.0, 0, 0, 0, np.pi/6, 0])))
    # T2_world = T1_se3 @ T_1_2_rel
    # For simplicity, define T2 directly in world for now
    T2_se3_rot_part = so3_exp(so3_hat(np.array([0, np.pi/6, np.pi/4]))) # Combined rotation
    T2_se3 = np.eye(4)
    T2_se3[:3,:3] = T2_se3_rot_part
    T2_se3[:3,3] = np.array([2.0, 0.5, 0.1]) # Arbitrary translation
    sm2 = Submap(2, 30.0, T2_se3, surfel_array_fixture)

    return [sm0, sm1, sm2]

def test_graph_builder_add_node(sample_submaps):
    builder = GraphBuilder()
    sm0, sm1, _ = sample_submaps

    node_id0 = builder.add_node(sm0)
    assert node_id0 == sm0.id
    assert node_id0 in builder.nodes
    assert np.allclose(builder.get_node_pose(node_id0), sm0.odometry_pose)
    assert builder.submap_id_to_node_id[sm0.id] == node_id0

    node_id1 = builder.add_node(sm1)
    assert node_id1 == sm1.id
    assert np.allclose(builder.get_node_pose(node_id1), sm1.odometry_pose)

    with pytest.raises(TypeError):
        builder.add_node("not_a_submap")

def test_graph_builder_add_odometry_edge(sample_submaps):
    builder = GraphBuilder()
    sm0, sm1, sm2 = sample_submaps

    builder.add_node(sm0)
    builder.add_node(sm1)
    builder.add_node(sm2)

    # Edge 0 -> 1
    edge01 = builder.add_odometry_edge(sm0.id, sm1.id)
    assert len(builder.edges) == 1
    assert edge01.from_node_id == sm0.id
    assert edge01.to_node_id == sm1.id
    assert edge01.type == 'odom'

    # Expected relative pose T_0_1 = T0_w^-1 * T1_w
    T0_w_inv = np.linalg.inv(sm0.odometry_pose)
    T1_w = sm1.odometry_pose
    expected_rel_pose01 = T0_w_inv @ T1_w
    assert np.allclose(edge01.relative_pose_se3, expected_rel_pose01)
    assert np.allclose(edge01.information_matrix, np.eye(6)) # Default info matrix

    # Edge 1 -> 2
    custom_info_matrix = np.diag([1,2,3,4,5,6])
    edge12 = builder.add_odometry_edge(sm1.id, sm2.id, information_matrix=custom_info_matrix)
    assert len(builder.edges) == 2
    T1_w_inv = np.linalg.inv(sm1.odometry_pose)
    T2_w = sm2.odometry_pose
    expected_rel_pose12 = T1_w_inv @ T2_w
    assert np.allclose(edge12.relative_pose_se3, expected_rel_pose12)
    assert np.allclose(edge12.information_matrix, custom_info_matrix)

    with pytest.raises(ValueError, match="not found in graph nodes"):
        builder.add_odometry_edge(sm0.id, 99) # Node 99 does not exist

def test_graph_builder_placeholders_callable(sample_submaps):
    """Tests that placeholder methods can be called without error."""
    builder = GraphBuilder()
    sm0, sm1, _ = sample_submaps
    builder.add_node(sm0)
    builder.add_node(sm1)

    collector = SubmapCollector() # merge_redundant_nodes needs this
    collector.add_submap(sm0.timestamp, sm0.odometry_pose, sm0.surfels)
    collector.add_submap(sm1.timestamp, sm1.odometry_pose, sm1.surfels)


    builder.optimize_graph() # Should print placeholder message
    builder.merge_redundant_nodes(sm0.id, sm1.id, collector) # Should print placeholder message
    # No assertions needed other than them not crashing.
    # Actual functionality will be tested in later iterations.

def test_find_loop_closures(surfel_array_fixture):
    builder = GraphBuilder()
    collector = SubmapCollector()

    # Node 0: (0,0,0) at t=0
    sm0_pose = np.eye(4)
    sm0 = collector.add_submap(0.0, sm0_pose, surfel_array_fixture)
    builder.add_node(sm0)

    # Node 1: (10,0,0) at t=10 (too far for LC with node 0 if radius is small)
    sm1_pose = np.eye(4); sm1_pose[0,3] = 10.0
    sm1 = collector.add_submap(10.0, sm1_pose, surfel_array_fixture)
    builder.add_node(sm1)

    # Node 2: (0.5,0,0) at t=20 (close to node 0, good time diff) -> Potential LC
    sm2_pose = np.eye(4); sm2_pose[0,3] = 0.5
    sm2 = collector.add_submap(20.0, sm2_pose, surfel_array_fixture)
    builder.add_node(sm2)

    # Node 3: (0.6,0,0) at t=1 (close to node 0, but too close in time) -> No LC due to time
    sm3_pose = np.eye(4); sm3_pose[0,3] = 0.6
    sm3 = collector.add_submap(1.0, sm3_pose, surfel_array_fixture) # Time 1.0
    builder.add_node(sm3)

    # Node 4: (20,0,0) at t=30 (far from node 0)
    sm4_pose = np.eye(4); sm4_pose[0,3] = 20.0
    sm4 = collector.add_submap(30.0, sm4_pose, surfel_array_fixture)
    builder.add_node(sm4)

    # Find LCs for sm2 (node_id 2)
    # Radius should catch sm0. min_time_diff should exclude sm3 if we were checking against sm0.
    # Here, we check LCs *for* sm2.
    # sm2 is at (0.5,0,0), t=20.
    # Candidates to check against sm2:
    # - sm0: (0,0,0), t=0. Dist=0.5. TimeDiff=20. Potential.
    # - sm1: (10,0,0), t=10. Dist=9.5. TimeDiff=10. (Potentially too far, or too small time diff if min_time_diff was higher)
    # - sm3: (0.6,0,0), t=1. Dist=0.1. TimeDiff=19. Potential.
    # - sm4: (20,0,0), t=30. Dist=19.5. TimeDiff=10.

    # Test 1: Find LCs for Node 2 (sm2)
    # Radius 1.0m, min_time_diff 5s
    # Expected: (2,0) because dist(sm2,sm0)=0.5 < 1.0 and abs(20-0)=20 > 5
    #           (2,3) because dist(sm2,sm3)=0.1 < 1.0 and abs(20-1)=19 > 5
    candidates_for_sm2 = builder.find_loop_closures(sm2.id, collector, radius_m=1.0, min_time_diff_s=5.0)
    candidate_pairs_sm2 = sorted([(c[0], c[1]) for c in candidates_for_sm2]) # Sort for consistent comparison

    expected_pairs_sm2 = sorted([(sm2.id, sm0.id), (sm2.id, sm3.id)])
    assert candidate_pairs_sm2 == expected_pairs_sm2, f"LCs for sm2: Expected {expected_pairs_sm2}, Got {candidate_pairs_sm2}"


    # Test 2: Find LCs for Node 0 (sm0)
    # Radius 1.0m, min_time_diff 5s
    # Expected: (0,2) because dist(sm0,sm2)=0.5 < 1.0 and abs(0-20)=20 > 5
    #           sm3 is too close in time (abs(0-1)=1 < 5)
    candidates_for_sm0 = builder.find_loop_closures(sm0.id, collector, radius_m=1.0, min_time_diff_s=5.0)
    candidate_pairs_sm0 = sorted([(c[0], c[1]) for c in candidates_for_sm0])
    expected_pairs_sm0 = sorted([(sm0.id, sm2.id)])
    assert candidate_pairs_sm0 == expected_pairs_sm0, f"LCs for sm0: Expected {expected_pairs_sm0}, Got {candidate_pairs_sm0}"

    # Test 3: No candidates if current_submap_id doesn't exist
    assert builder.find_loop_closures(999, collector, 1.0, 5.0) == []

    # Test 4: No candidates if all other nodes are too close in time
    # Find LCs for sm0, but with large min_time_diff_s
    candidates_for_sm0_large_tdiff = builder.find_loop_closures(sm0.id, collector, radius_m=1.0, min_time_diff_s=25.0)
    assert candidates_for_sm0_large_tdiff == [], f"Expected no LCs for sm0 with large time diff, got {candidates_for_sm0_large_tdiff}"


def test_icp_point2plane_placeholder(surfel_array_fixture):
    builder = GraphBuilder()
    surfel_dtype = surfel_array_fixture.dtype

    # Create at least 6 source and target surfels for ICP to attempt a 6DoF solve
    src_surfels_list = [
        ([0.0,0.0,0.0], [0,0,1], 0.9, 1.0, 0.1), ([0.1,0.0,0.0], [0,0,1], 0.9, 1.0, 0.1),
        ([0.0,0.1,0.0], [0,0,1], 0.9, 1.0, 0.1), ([0.1,0.1,0.0], [0,0,1], 0.9, 1.0, 0.1),
        ([0.0,0.0,0.1], [0,0,1], 0.9, 1.0, 0.1), ([0.1,0.0,0.1], [0,0,1], 0.9, 1.0, 0.1),
    ]
    src_surfels = np.array(src_surfels_list, dtype=surfel_dtype)

    tgt_surfels_list = [
        ([0.05,0.0,0.0], [0,0,1], 0.9, 1.0, 0.1), ([0.15,0.0,0.0], [0,0,1], 0.9, 1.0, 0.1),
        ([0.05,0.1,0.0], [0,0,1], 0.9, 1.0, 0.1), ([0.15,0.1,0.0], [0,0,1], 0.9, 1.0, 0.1),
        ([0.05,0.0,0.1], [0,0,1], 0.9, 1.0, 0.1), ([0.15,0.0,0.1], [0,0,1], 0.9, 1.0, 0.1),
    ]
    tgt_surfels = np.array(tgt_surfels_list, dtype=surfel_dtype)

    # Test with no initial guess - it might not converge well, but should run
    pose, cov, success = builder.icp_point2plane(src_surfels, tgt_surfels, max_iterations=5)
    # We don't assert success is True for placeholder as convergence isn't guaranteed with dummy data
    # Just check that it ran and returned expected shapes
    assert pose.shape == (4,4)
    assert cov.shape == (6,6)
    # Success can be True or False depending on convergence with this data.

    # Test with an initial guess
    initial_guess = np.eye(4)
    initial_guess[0,3] = 0.01 # Small initial offset
    pose_guess, cov_guess, success_guess = builder.icp_point2plane(src_surfels, tgt_surfels, initial_guess, max_iterations=5)
    assert pose_guess.shape == (4,4)
    assert cov_guess.shape == (6,6)
    # Again, success_guess can be True or False.


def test_add_loop_closure_edge(sample_submaps):
    builder = GraphBuilder()
    sm0, _, sm2 = sample_submaps # Using sm0 and sm2 which are not directly consecutive

    builder.add_node(sm0)
    builder.add_node(sm2)

    # Dummy relative pose and info matrix for LC
    T_0_2_lc = se3_exp(se3_hat(np.array([-0.5, 0.1, -0.2, 0.05, -0.03, 0.1])))
    info_matrix_lc = np.eye(6) * 100

    edge = builder.add_loop_closure_edge(sm0.id, sm2.id, T_0_2_lc, info_matrix_lc)

    assert len(builder.edges) == 1
    lc_edge = builder.edges[0]
    assert lc_edge.type == 'loop_closure'
    assert lc_edge.from_node_id == sm0.id
    assert lc_edge.to_node_id == sm2.id
    assert np.allclose(lc_edge.relative_pose_se3, T_0_2_lc)
    assert np.allclose(lc_edge.information_matrix, info_matrix_lc)


# --- ICP Tests ---
def test_icp_recovers_known_transform(surfel_array_fixture): # Use surfel_array_fixture for dtype
    builder = GraphBuilder()
    surfel_dtype = surfel_array_fixture.dtype # Get dtype from fixture

    # 1. Create target surfels (e.g., a small plane)
    target_surfels_list = [
        # Plane 1 (XY plane, normal Z)
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, 1.0, 0.1),
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, 1.0, 0.1),
        # Plane 2 (XZ plane, normal Y)
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.9, 1.0, 0.1),
        ([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.9, 1.0, 0.1),
        # Plane 3 (YZ plane, normal X)
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.9, 1.0, 0.1),
        ([0.0, 1.0, 1.0], [1.0, 0.0, 0.0], 0.9, 1.0, 0.1),
    ]
    target_surfels = np.array(target_surfels_list, dtype=surfel_dtype)

    # 2. Define a known SE(3) transform (T_target_source)
    #    This is the transform ICP should find. It takes points from source to target frame.
    #    e.g., translate by (0.5, -0.5, 0.1), rotate 30 deg around Z axis
    true_dx = np.array([0.5, -0.5, 0.1])
    true_rot_axis = np.array([0, 0, 1.0])
    true_rot_angle = np.deg2rad(30)
    true_omega = true_rot_axis * true_rot_angle

    true_transform_T_target_source = se3_exp(se3_hat(np.hstack((true_dx, true_omega))))

    # 3. Create source surfels by applying the *inverse* of true_transform to target surfels' means.
    #    P_source = inv(T_target_source) @ P_target
    #    Normals of source surfels can be transformed by inv(R_target_source).
    inv_true_transform = np.linalg.inv(true_transform_T_target_source)
    inv_R_true = inv_true_transform[:3,:3]

    source_surfels_list = []
    for t_surf in target_surfels:
        t_mean_homo = np.append(t_surf['mean'], 1)
        s_mean_homo = inv_true_transform @ t_mean_homo
        s_mean = s_mean_homo[:3]

        s_normal = inv_R_true @ t_surf['normal']
        # Ensure normal is unit vector
        s_normal = s_normal / (np.linalg.norm(s_normal) + 1e-9)

        source_surfels_list.append(
            (s_mean, s_normal, t_surf['score'], t_surf['timestamp_mean'], t_surf['resolution'])
        )
    source_surfels = np.array(source_surfels_list, dtype=surfel_dtype)

    # 4. Provide an initial pose estimate
    initial_pose_estimate = np.eye(4) # Start from identity

    # 5. Call icp_point2plane
    estimated_pose, covariance, success = builder.icp_point2plane(
        source_surfels, target_surfels, initial_relative_guess_se3=initial_pose_estimate,
        max_iterations=100, tolerance=1e-5, max_correspondence_dist=1.0 # Reduced max_corr_dist
    )

    assert success, "ICP did not converge successfully"

    # 6. Assert that estimated_pose is close to true_transform_T_target_source
    #    Can check this by seeing if inv(estimated_pose) @ true_transform is close to Identity.
    error_transform = np.linalg.inv(estimated_pose) @ true_transform_T_target_source
    error_twist = se3_vee(se3_log(error_transform)) # Get the 6-vector error

    translation_error = np.linalg.norm(error_twist[:3])
    rotation_error_rad = np.linalg.norm(error_twist[3:])

    # Define tolerances for translation (meters) and rotation (radians)
    translation_tol = 1e-3 # 1 mm
    rotation_tol = np.deg2rad(0.1) # 0.1 degrees in radians

    assert translation_error < translation_tol, f"ICP translation error too high: {translation_error} m"
    assert rotation_error_rad < rotation_tol, f"ICP rotation error too high: {np.rad2deg(rotation_error_rad)} degrees"

    # Also check that the returned covariance matrix has the right shape (6x6)
    assert covariance.shape == (6,6), "Covariance matrix shape is incorrect"
    # And that it's positive semi-definite (all eigenvalues >= 0, or small negative due to numerics)
    eigenvalues_cov = np.linalg.eigvalsh(covariance) # eigvalsh for symmetric matrices
    assert np.all(eigenvalues_cov >= -1e-9), "Covariance matrix is not positive semi-definite"


def test_icp_rotation_only(surfel_array_fixture):
    builder = GraphBuilder()
    surfel_dtype = surfel_array_fixture.dtype

    target_surfels_list = [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, 1.0, 0.1), ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, 1.0, 0.1),
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.9, 1.0, 0.1), ([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.9, 1.0, 0.1),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.9, 1.0, 0.1), ([0.0, 1.0, 1.0], [1.0, 0.0, 0.0], 0.9, 1.0, 0.1),
    ]
    target_surfels = np.array(target_surfels_list, dtype=surfel_dtype)

    true_dx = np.array([0.0, 0.0, 0.0]) # Pure rotation
    true_rot_axis = np.array([0, 0, 1.0])
    true_rot_angle = np.deg2rad(30)
    true_omega = true_rot_axis * true_rot_angle
    true_transform_T_target_source = se3_exp(se3_hat(np.hstack((true_dx, true_omega))))

    inv_true_transform = np.linalg.inv(true_transform_T_target_source)
    inv_R_true = inv_true_transform[:3,:3]
    source_surfels_list = []
    for t_surf in target_surfels:
        t_mean_homo = np.append(t_surf['mean'], 1)
        s_mean_homo = inv_true_transform @ t_mean_homo
        s_mean = s_mean_homo[:3]
        s_normal = inv_R_true @ t_surf['normal']
        s_normal = s_normal / (np.linalg.norm(s_normal) + 1e-9)
        source_surfels_list.append(
            (s_mean, s_normal, t_surf['score'], t_surf['timestamp_mean'], t_surf['resolution'])
        )
    source_surfels = np.array(source_surfels_list, dtype=surfel_dtype)

    initial_pose_estimate = np.eye(4)
    estimated_pose, _, success = builder.icp_point2plane(
        source_surfels, target_surfels, initial_relative_guess_se3=initial_pose_estimate,
        max_iterations=100, tolerance=1e-5, max_correspondence_dist=1.0
    )

    assert success, "ICP (rotation-only) did not converge successfully"

    error_transform = np.linalg.inv(estimated_pose) @ true_transform_T_target_source
    error_twist = se3_vee(se3_log(error_transform))
    translation_error = np.linalg.norm(error_twist[:3])
    rotation_error_rad = np.linalg.norm(error_twist[3:])

    translation_tol = 1e-3
    rotation_tol = np.deg2rad(0.1)

    assert translation_error < translation_tol, f"ICP (rotation-only) translation error too high: {translation_error} m"
    assert rotation_error_rad < rotation_tol, f"ICP (rotation-only) rotation error too high: {np.rad2deg(rotation_error_rad)} degrees"


def test_icp_translation_only(surfel_array_fixture):
    builder = GraphBuilder()
    surfel_dtype = surfel_array_fixture.dtype

    target_surfels_list = [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, 1.0, 0.1), ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, 1.0, 0.1),
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.9, 1.0, 0.1), ([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.9, 1.0, 0.1),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.9, 1.0, 0.1), ([0.0, 1.0, 1.0], [1.0, 0.0, 0.0], 0.9, 1.0, 0.1),
    ]
    target_surfels = np.array(target_surfels_list, dtype=surfel_dtype)

    true_dx = np.array([0.2, -0.1, 0.3]) # Pure translation
    true_omega = np.array([0.0, 0.0, 0.0])
    true_transform_T_target_source = se3_exp(se3_hat(np.hstack((true_dx, true_omega))))

    inv_true_transform = np.linalg.inv(true_transform_T_target_source)
    inv_R_true = inv_true_transform[:3,:3] # Should be identity
    source_surfels_list = []
    for t_surf in target_surfels:
        t_mean_homo = np.append(t_surf['mean'], 1)
        s_mean_homo = inv_true_transform @ t_mean_homo
        s_mean = s_mean_homo[:3]
        # For pure translation, source normals are same as target normals
        s_normal = t_surf['normal'] # More direct: inv_R_true @ t_surf['normal']
        source_surfels_list.append(
            (s_mean, s_normal, t_surf['score'], t_surf['timestamp_mean'], t_surf['resolution'])
        )
    source_surfels = np.array(source_surfels_list, dtype=surfel_dtype)

    initial_pose_estimate = np.eye(4)
    estimated_pose, _, success = builder.icp_point2plane(
        source_surfels, target_surfels, initial_relative_guess_se3=initial_pose_estimate,
        max_iterations=100, tolerance=1e-5, max_correspondence_dist=1.0
    )

    assert success, "ICP (translation-only) did not converge successfully"

    error_transform = np.linalg.inv(estimated_pose) @ true_transform_T_target_source
    error_twist = se3_vee(se3_log(error_transform))
    translation_error = np.linalg.norm(error_twist[:3])
    rotation_error_rad = np.linalg.norm(error_twist[3:])

    translation_tol = 1e-3
    rotation_tol = np.deg2rad(0.1)

    assert translation_error < translation_tol, f"ICP (translation-only) translation error too high: {translation_error} m"
    assert rotation_error_rad < rotation_tol, f"ICP (translation-only) rotation error too high: {np.rad2deg(rotation_error_rad)} degrees"


def test_icp_identity_transform(surfel_array_fixture):
    builder = GraphBuilder()
    surfel_dtype = surfel_array_fixture.dtype

    target_surfels_list = [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, 1.0, 0.1), ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, 1.0, 0.1),
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.9, 1.0, 0.1), ([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.9, 1.0, 0.1),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.9, 1.0, 0.1), ([0.0, 1.0, 1.0], [1.0, 0.0, 0.0], 0.9, 1.0, 0.1),
    ]
    target_surfels = np.array(target_surfels_list, dtype=surfel_dtype)

    # Source surfels are identical to target surfels
    source_surfels = np.copy(target_surfels)
    true_transform_T_target_source = np.eye(4) # Identity transform

    initial_pose_estimate = np.eye(4)
    # For identity, ICP should converge very quickly, possibly in 0 iterations if checked first.
    # Let's give it a few iterations to ensure it doesn't diverge.
    estimated_pose, _, success = builder.icp_point2plane(
        source_surfels, target_surfels, initial_relative_guess_se3=initial_pose_estimate,
        max_iterations=10, tolerance=1e-7, max_correspondence_dist=0.1 # Tighter tolerance for identity
    )

    assert success, "ICP (identity) did not report convergence (success=True)"

    error_transform = np.linalg.inv(estimated_pose) @ true_transform_T_target_source
    error_twist = se3_vee(se3_log(error_transform))
    translation_error = np.linalg.norm(error_twist[:3])
    rotation_error_rad = np.linalg.norm(error_twist[3:])

    # Expect very small errors for identity
    translation_tol = 1e-6
    rotation_tol = np.deg2rad(1e-4) # Very small rotation tolerance

    assert translation_error < translation_tol, f"ICP (identity) translation error too high: {translation_error} m"
    assert rotation_error_rad < rotation_tol, f"ICP (identity) rotation error too high: {np.rad2deg(rotation_error_rad)} degrees"


def test_icp_robustness_to_noise(surfel_array_fixture):
    builder = GraphBuilder()
    surfel_dtype = surfel_array_fixture.dtype

    # 1. Create target surfels (same as in the noise-free test)
    target_surfels_list = [
        # Plane 1 (XY plane, normal Z)
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, 1.0, 0.1),
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, 1.0, 0.1),
        # Plane 2 (XZ plane, normal Y)
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.9, 1.0, 0.1),
        ([1.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.9, 1.0, 0.1),
        # Plane 3 (YZ plane, normal X)
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.9, 1.0, 0.1),
        ([0.0, 1.0, 1.0], [1.0, 0.0, 0.0], 0.9, 1.0, 0.1),
    ]
    target_surfels = np.array(target_surfels_list, dtype=surfel_dtype)

    # 2. Define the true transform (same as before)
    true_dx = np.array([0.5, -0.5, 0.1])
    true_omega = np.array([0, 0, np.deg2rad(30)])
    true_transform_T_target_source = se3_exp(se3_hat(np.hstack((true_dx, true_omega))))
    inv_true_transform = np.linalg.inv(true_transform_T_target_source)
    inv_R_true = inv_true_transform[:3,:3]

    # 3. Create source surfels and add noise
    source_surfels_list_clean = []
    for t_surf in target_surfels:
        t_mean_homo = np.append(t_surf['mean'], 1)
        s_mean_homo = inv_true_transform @ t_mean_homo
        s_mean_clean = s_mean_homo[:3]
        s_normal_clean = inv_R_true @ t_surf['normal']
        s_normal_clean = s_normal_clean / (np.linalg.norm(s_normal_clean) + 1e-9)
        source_surfels_list_clean.append(
            (s_mean_clean, s_normal_clean, t_surf['score'], t_surf['timestamp_mean'], t_surf['resolution'])
        )
    source_surfels_clean = np.array(source_surfels_list_clean, dtype=surfel_dtype)

    # Add Gaussian noise to source surfel mean positions
    noise_std_dev = 0.01 # 1 cm noise
    noisy_source_surfels = np.copy(source_surfels_clean)
    # Generate noise for all means at once
    noise_vectors = np.random.normal(scale=noise_std_dev, size=(len(noisy_source_surfels), 3))
    noisy_source_surfels['mean'] += noise_vectors
    # Normals are kept from the clean transformed versions for this test,
    # as point-to-plane relies on good target normals, source normals aren't directly used by this ICP variant.

    # 4. Initial pose estimate
    initial_pose_estimate = np.eye(4) # Start from identity

    # 5. Call ICP
    estimated_pose, _, success = builder.icp_point2plane(
        noisy_source_surfels, target_surfels, initial_relative_guess_se3=initial_pose_estimate,
        max_iterations=100, tolerance=1e-5, max_correspondence_dist=1.0 # Reduced max_corr_dist
    )

    assert success, "ICP did not converge successfully with noisy data"

    # 6. Assert that estimated_pose is still close to true_transform
    error_transform = np.linalg.inv(estimated_pose) @ true_transform_T_target_source
    error_twist = se3_vee(se3_log(error_transform))

    translation_error = np.linalg.norm(error_twist[:3])
    rotation_error_rad = np.linalg.norm(error_twist[3:])

    # Tolerances might need to be slightly larger due to noise
    translation_tol_noisy = 5 * noise_std_dev # e.g., 5cm if noise is 1cm
    rotation_tol_noisy = np.deg2rad(1.0)  # e.g., 1 degree

    assert translation_error < translation_tol_noisy, \
        f"ICP noisy translation error too high: {translation_error} m (noise std: {noise_std_dev} m)"
    assert rotation_error_rad < rotation_tol_noisy, \
        f"ICP noisy rotation error too high: {np.rad2deg(rotation_error_rad)} degrees"
