import numpy as np
from collections import namedtuple # Or dataclasses if Python 3.7+ is assumed and preferred

# Define the Submap structure
# Using namedtuple for simplicity, can be upgraded to a class or dataclass later if more methods are needed.
# A simple class might be better for type hinting and potential methods later.
class Submap:
    def __init__(self, submap_id, timestamp, odometry_pose_se3, surfels, gravity_vector_local=None):
        """
        Represents a submap.

        Args:
            submap_id (int): A unique identifier for this submap.
            timestamp (float): Timestamp associated with the submap (e.g., end time of its window).
            odometry_pose_se3 (np.ndarray): The SE(3) pose (4x4 matrix) of this submap's origin
                                            in the global/world frame, as estimated by odometry.
            surfels (np.ndarray): Structured array of surfels belonging to this submap.
                                  (dtype should match surfel definition from surfel.py)
            gravity_vector_local (np.ndarray, optional): (3,) vector representing gravity in the submap's local frame.
        """
        if not isinstance(odometry_pose_se3, np.ndarray) or odometry_pose_se3.shape != (4,4):
            raise ValueError("odometry_pose_se3 must be a 4x4 numpy array.")
        # TODO: Add check for surfels dtype and structure if possible/performant

        self.id = submap_id
        self.timestamp = timestamp
        self.odometry_pose = np.copy(odometry_pose_se3) # Pose of the submap frame in world
        self.surfels = np.copy(surfels) # Local coordinates relative to submap frame origin

        # The "local coordinate frame" of a submap is defined by its odometry_pose.
        # Surfel positions within self.surfels are typically relative to this submap frame.

        self.gravity_vector_local = np.copy(gravity_vector_local) if gravity_vector_local is not None else None


class SubmapCollector:
    def __init__(self):
        self.submaps = [] # List to store Submap objects
        self.next_submap_id = 0

    def add_submap(self, timestamp, odometry_pose_se3, surfels, gravity_vector_local=None):
        """
        Creates a new Submap object and adds it to the collection.

        Args:
            timestamp (float): Timestamp for the new submap.
            odometry_pose_se3 (np.ndarray): SE(3) odometry pose of the submap.
            surfels (np.ndarray): Array of surfels for the submap.
            gravity_vector_local (np.ndarray, optional): Gravity vector in submap's local frame.

        Returns:
            Submap: The newly created and added Submap object.
        """
        new_submap = Submap(
            submap_id=self.next_submap_id,
            timestamp=timestamp,
            odometry_pose_se3=odometry_pose_se3,
            surfels=surfels,
            gravity_vector_local=gravity_vector_local
        )
        self.submaps.append(new_submap)
        self.next_submap_id += 1
        return new_submap

    def get_submap_by_id(self, submap_id):
        for sm in self.submaps:
            if sm.id == submap_id:
                return sm
        return None

# Define Edge structure (placeholder for now, might need more fields like covariance)
# An edge connects two node IDs (submap IDs) and stores the relative transform.
PoseGraphEdge = namedtuple('PoseGraphEdge', ['from_node_id', 'to_node_id', 'relative_pose_se3', 'information_matrix', 'type'])
# type can be 'odom' or 'loop_closure' or 'imu_gravity' etc.

class GraphBuilder:
    def __init__(self):
        # Nodes in the graph: keys are submap_ids (or a unified node_id after merging)
        # Values could be current estimated SE(3) poses of these nodes in the world frame.
        # Initially, node poses can be the odometry_poses of the submaps.
        self.nodes = {} # {node_id: current_estimated_pose_se3 (4x4)}

        self.edges = [] # List of PoseGraphEdge namedtuples

        self.submap_id_to_node_id = {} # If merging happens, submap_id might map to a merged node_id
                                      # For now, node_id will be same as submap_id.
        self._next_node_id_internal = 0 # If we need node IDs separate from submap IDs

    def add_node(self, submap):
        """
        Adds a new node to the pose graph based on a submap.
        The node's initial pose is the submap's odometry pose.
        """
        if not isinstance(submap, Submap):
            raise TypeError("Input must be a Submap object.")

        node_id = submap.id # For now, node_id is the submap_id
        if node_id in self.nodes:
            # print(f"Warning: Node {node_id} already exists. Updating its pose/data might be needed or this indicates an issue.")
            # For now, let's assume we don't add duplicates this way.
            # If submaps are continually generated, this will be called for each new one.
            pass

        self.nodes[node_id] = np.copy(submap.odometry_pose)
        self.submap_id_to_node_id[submap.id] = node_id
        return node_id

    def add_odometry_edge(self, from_submap_id, to_submap_id, information_matrix=None):
        """
        Adds an odometry edge between two nodes (submaps).
        The relative pose is calculated from their current estimated poses in the graph (or their odom poses).
        """
        from_node_id = self.submap_id_to_node_id.get(from_submap_id)
        to_node_id = self.submap_id_to_node_id.get(to_submap_id)

        if from_node_id is None or to_node_id is None:
            raise ValueError("One or both submap IDs not found in graph nodes.")

        pose_from_world = self.nodes[from_node_id]
        pose_to_world = self.nodes[to_node_id]

        # Relative pose: T_from_to = T_from_world^-1 * T_to_world
        relative_pose = np.linalg.inv(pose_from_world) @ pose_to_world

        if information_matrix is None:
            # Default information matrix (e.g., identity, or based on odometry uncertainty)
            # For SE(3), this is a 6x6 matrix (for dx,dy,dz,droll,dpitch,dyaw)
            information_matrix = np.eye(6)

        edge = PoseGraphEdge(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relative_pose_se3=relative_pose,
            information_matrix=information_matrix,
            type='odom'
        )
        self.edges.append(edge)
        return edge

    def get_node_pose(self, node_id):
        return self.nodes.get(node_id)

    def optimize_graph(self):
        """
        Placeholder for global pose graph optimization.
        This will involve:
        - Constructing a non-linear least squares problem from all nodes and edges.
        - Using a solver (e.g., scipy.sparse.linalg, or g2o/ceres if allowed by deps)
          to update self.nodes poses.
        """
        # For Iteration 6, this is just a placeholder. Full implementation in Iteration 7.
        # This will involve building a sparse linear system (Ax=b) from all residuals and Jacobians
        # and solving it.
        # Residuals include: odometry edges, loop closure edges, gravity alignment term.
        # Variables are corrections to current node poses (self.nodes).
        print("GraphBuilder.optimize_graph() called - (Placeholder, no optimization performed yet)")

        # Example of what might happen here:
        # 1. Parameterize node poses (e.g., SE(3) -> 6-vector delta).
        # 2. For each edge, compute residual and Jacobian w.r.t. involved node pose deltas.
        #    - Odometry edge: err = log(T_ij_measured^-1 * (T_i_world^-1 * T_j_world))
        #    - Loop closure edge: err = log(T_kl_icp^-1 * (T_k_world^-1 * T_l_world))
        # 3. For each node with gravity info, compute gravity alignment residual and Jacobian.
        #    - err_up = R_i * u_hat_i - w_u (from Eq. 16)
        # 4. Form Hx = -g (or A^T A dx = A^T r) system using information matrices as weights.
        # 5. Solve for dx (pose corrections).
        # 6. Update self.nodes poses.
        # 7. Iterate if needed.
        pass

    def merge_redundant_nodes(self, submap_A_id, submap_B_id, submap_collector):
        """
        Placeholder for merging redundant nodes (submaps).
        Actual implementation would involve:
        - Checking for overlap (e.g. based on surfel proximity or bounding boxes of submaps from collector).
        - Checking Mahalanobis distance between their poses.
        - If criteria met:
            - Choose one node to keep (or create a new merged node).
            - Update submap_id_to_node_id mapping for the merged submap.
            - Re-wire edges connected to the removed node to point to the kept/merged node.
            - Potentially update the kept/merged node's pose and surfel map.
        """
        # For Iteration 6, this is just a placeholder.
        print(f"GraphBuilder.merge_redundant_nodes({submap_A_id}, {submap_B_id}) called - (Placeholder)")
        pass

    def find_loop_closures(self, current_submap_id, submap_collector, radius_m, min_time_diff_s=30.0):
        """
        Finds potential loop closure candidates for a given current submap.
        Uses KD-tree on node positions for spatial search.
        Further filtering by time difference and possibly Mahalanobis distance.

        Args:
            current_submap_id (int): The ID of the submap to find closures for.
            submap_collector (SubmapCollector): Used to access submap timestamps and other data.
            radius_m (float): Search radius in meters for nearby nodes.
            min_time_diff_s (float): Minimum time difference to consider a distinct loop closure.

        Returns:
            list: List of (current_submap_id, candidate_submap_id) tuples.
        """
        if current_submap_id not in self.nodes:
            return []

        current_node_pose = self.nodes[current_submap_id]
        current_node_position = current_node_pose[:3, 3]
        current_submap = submap_collector.get_submap_by_id(current_submap_id)
        if not current_submap: return [] # Should not happen if node exists
        current_timestamp = current_submap.timestamp

        candidate_nodes_positions = []
        candidate_nodes_ids = []

        for node_id, pose_se3 in self.nodes.items():
            if node_id == current_submap_id:
                continue # Don't compare with self

            # Time difference check
            candidate_submap = submap_collector.get_submap_by_id(node_id)
            if not candidate_submap: continue
            if abs(candidate_submap.timestamp - current_timestamp) < min_time_diff_s:
                continue

            candidate_nodes_positions.append(pose_se3[:3, 3])
            candidate_nodes_ids.append(node_id)

        if not candidate_nodes_positions:
            return []

        from scipy.spatial import KDTree # Import here as it's a specific dependency

        # Build KD-tree from positions of other nodes
        kdtree = KDTree(np.array(candidate_nodes_positions))

        # Query for neighbors within radius
        # indices_in_candidates will be indices into candidate_nodes_positions/ids
        indices_in_candidates = kdtree.query_ball_point(current_node_position, r=radius_m)

        loop_closure_candidates = []
        for idx in indices_in_candidates:
            candidate_node_id = candidate_nodes_ids[idx]
            # Here, one might add Mahalanobis distance check if pose covariances were available.
            # For now, proximity and time difference are the main filters.
            loop_closure_candidates.append((current_submap_id, candidate_node_id))

        return loop_closure_candidates

    def icp_point2plane(self, src_surfels, tgt_surfels, initial_relative_guess_se3=None, max_iterations=20, tolerance=1e-4):
        """
        Performs point-to-plane ICP between two sets of surfels.
        Args:
            src_surfels (np.ndarray): Source surfels (will be transformed).
            tgt_surfels (np.ndarray): Target surfels (fixed).
            initial_relative_guess_se3 (np.ndarray, optional): Initial guess for T_tgt_src (transform from src to tgt).
                                                              Defaults to identity.
            max_iterations (int): Maximum ICP iterations.
            tolerance (float): Convergence tolerance for change in transform.
        Returns:
            tuple: (optimized_relative_pose_se3 (4x4), covariance_matrix (6x6), success_flag (bool))
                   optimized_relative_pose_se3 is T_tgt_src.
        """
        # This is a complex function. Placeholder for now.
        # Steps would involve:
        # 1. Initialize current_T_tgt_src with initial_relative_guess_se3 or Identity.
        # 2. Loop for max_iterations:
        #    a. Transform src_surfels_means using current_T_tgt_src.
        #    b. For each transformed src_surfel_mean, find closest point in tgt_surfels_means (or its normal projection).
        #       This requires a KDTree on tgt_surfels['mean'].
        #    c. Form point-to-plane error residuals: dot( (T_src_mean - tgt_mean), tgt_normal ).
        #    d. Compute Jacobian of these residuals w.r.t. small perturbation of current_T_tgt_src (a 6-vector xi).
        #    e. Solve linear system (J^T J) * xi = -J^T * residuals.
        #    f. Update current_T_tgt_src using se3_exp(se3_hat(xi)) @ current_T_tgt_src.
        #    g. Check for convergence (e.g., norm of xi < tolerance).
        # 3. Estimate covariance of the resulting transform (e.g., from (J^T J)^-1).

        print(f"GraphBuilder.icp_point2plane called - (Placeholder)")

        # Return a dummy successful result for now
        final_pose = np.eye(4) if initial_relative_guess_se3 is None else np.copy(initial_relative_guess_se3)
        # Simulate a small adjustment if there was an initial guess
        if initial_relative_guess_se3 is not None:
             final_pose[0,3] += 0.01 # Tiny change

        dummy_covariance = np.eye(6) * 0.1 # Dummy covariance
        return final_pose, dummy_covariance, True

    def add_loop_closure_edge(self, from_node_id, to_node_id, relative_pose_se3, information_matrix):
        """Adds a loop closure edge to the graph."""
        edge = PoseGraphEdge(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relative_pose_se3=relative_pose_se3,
            information_matrix=information_matrix,
            type='loop_closure'
        )
        self.edges.append(edge)
        return edge
