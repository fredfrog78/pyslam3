import argparse
from .io import LVX2Reader # Placeholder for future use

def main():
    parser = argparse.ArgumentParser(description="Wildcat SLAM: Online Continuous-Time 3D Lidar-Inertial SLAM.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input LVX2 file.")
    parser.add_argument("--output_map", type=str, required=True, help="Path to save the output map (PLY format).")
    parser.add_argument("--output_traj", type=str, required=True, help="Path to save the output trajectory (TXT format).")

    # Iteration 8+ arguments
    parser.add_argument("--voxel_size", type=float, default=0.5, help="Voxel size for surfel extraction.")
    parser.add_argument("--odometry_window_duration", type=float, default=2.0, help="Duration of the odometry sliding window in seconds.")
    parser.add_argument("--odometry_imu_frequency", type=float, default=100.0, help="IMU frequency for odometry.")
    parser.add_argument("--odometry_num_samples", type=int, default=10, help="Number of sample poses in odometry window.")
    # parser.add_argument("--iteration_count", type=int, default=5, help="Number of optimization iterations.") # For odometry window
    # parser.add_argument("--frame_rate", type=int, default=20, help="Processing frame rate (Hz) for lvx.") # LVX specific, usually derived

    # Iteration 9 arguments
    parser.add_argument("--profile", action="store_true", help="Enable cProfile for performance profiling.")


    args = parser.parse_args()

    print("Wildcat SLAM CLI")
    print(f"Input LVX2 file: {args.input}")
    print(f"Output map PLY file: {args.output_map}")
    print(f"Output trajectory TXT file: {args.output_traj}")
    print(f"Voxel Size: {args.voxel_size}")
    print(f"Odometry Window Duration: {args.odometry_window_duration}s")
    # Add other args if needed

    profiler = None
    if args.profile:
        import cProfile, pstats
        profiler = cProfile.Profile()
        print("\nPerformance profiling enabled. Running main logic under profiler...")
        profiler.enable()

    # Call the main logic
    actual_main_operation(args)

    if args.profile and profiler:
        profiler.disable()
        print("\n--- Performance Profile ---")
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(20)
        # stats.dump_stats("wildcat_profile.prof")
        # print("Profile data saved to wildcat_profile.prof")

def actual_main_operation(args): # Renamed from main_logic to avoid confusion with main()
    """Encapsulates the main operational logic of the CLI for profiling."""
    # --- Initialize components ---
    from .io import LVX2Reader, PLYWriter, TrajWriter
    from .odometry import OdometryWindow # IMUIntegrator might be used internally or by a runner
    from .surfel import extract_surfels
    from .pose_graph import SubmapCollector, GraphBuilder
    import numpy as np # For dummy data

    try:
        # 1. LVX2 Reader
        # For now, we'll just read headers. Full frame reading is complex.
        print(f"\nAttempting to read LVX2 file: {args.input}")
        with LVX2Reader(args.input) as reader:
            print("LVX2 public header:", reader.public_header)
            print("LVX2 private header:", reader.private_header)
            print("LVX2 device info:", reader.device_info_list)
            # In a real scenario, reader would yield frames/packets
            num_frames_to_simulate = 10 # Simulate processing a few "frames"

        # 2. Odometry (Conceptual - OdometryWindow is more of a component than a runner)
        #    An "OdometryRunner" would manage OdometryWindow instances, feed IMU/Lidar data.
        #    For now, simulate its output: poses and surfels over time.
        print("\nSimulating Odometry Processing...")
        # odometry_window = OdometryWindow(
        #     args.odometry_window_duration,
        #     args.odometry_imu_frequency,
        #     args.odometry_num_samples
        # )

        # Simulated odometry outputs
        simulated_odometry_poses = []
        simulated_surfel_maps = [] # List of surfel arrays, one per submap
        current_sim_pose = np.eye(4)
        sim_timestamp = 0.0

        # Create some dummy surfels for the map
        dummy_surfel_dtype = [('mean', '3f8'), ('normal', '3f8'), ('score', 'f8'),
                              ('timestamp_mean', 'f8'), ('resolution', 'f8')]

        for i in range(num_frames_to_simulate): # Simulate a new submap being generated periodically
            sim_timestamp += 1.0 # e.g., 1 second per submap

            # Simulate motion: small translation and rotation
            delta_trans = np.array([0.1 * i, 0.05 * i, 0])
            delta_rot_vec = np.array([0, 0, 0.02 * i]) # Small rotation around Z
            from .geometry import se3_exp, so3_hat # Local import for se3_exp, so3_hat
            delta_pose = se3_exp(np.array([
                [0, -delta_rot_vec[2], delta_rot_vec[1], delta_trans[0]],
                [delta_rot_vec[2], 0, -delta_rot_vec[0], delta_trans[1]],
                [-delta_rot_vec[1], delta_rot_vec[0], 0, delta_trans[2]],
                [0,0,0,0]
            ]))
            current_sim_pose = current_sim_pose @ delta_pose
            simulated_odometry_poses.append(np.copy(current_sim_pose))

            # Simulate some surfels for this submap
            num_sim_surfels = np.random.randint(5, 15)
            sim_surfels_mean = np.random.rand(num_sim_surfels, 3) * args.voxel_size * 5 # Spread out
            sim_surfels_normal = np.zeros((num_sim_surfels, 3))
            sim_surfels_normal[:,2] = 1.0 # All normals point up
            sim_surfels_score = np.random.rand(num_sim_surfels) * 0.5 + 0.5 # Scores > 0.5
            sim_surfels_ts = np.full(num_sim_surfels, sim_timestamp)
            sim_surfels_res = np.full(num_sim_surfels, args.voxel_size)

            current_map_surfels = np.zeros(num_sim_surfels, dtype=dummy_surfel_dtype)
            current_map_surfels['mean'] = sim_surfels_mean
            current_map_surfels['normal'] = sim_surfels_normal
            current_map_surfels['score'] = sim_surfels_score
            current_map_surfels['timestamp_mean'] = sim_surfels_ts
            current_map_surfels['resolution'] = sim_surfels_res
            simulated_surfel_maps.append(current_map_surfels)

        # 3. Pose Graph SLAM
        print("\nSimulating Pose Graph SLAM Processing...")
        submap_collector = SubmapCollector()
        graph_builder = GraphBuilder()

        # Simulate processing odometry outputs
        for i in range(len(simulated_odometry_poses)):
            odom_pose = simulated_odometry_poses[i]
            submap_surfels = simulated_surfel_maps[i]
            # Timestamp for submap could be start/mid/end of its collection window
            submap_ts = i * 1.0 # Using the sim_timestamp directly

            new_submap = submap_collector.add_submap(submap_ts, odom_pose, submap_surfels)
            graph_builder.add_node(new_submap)

            if i > 0: # Add odometry edge from previous submap
                prev_submap_id = new_submap.id - 1
                graph_builder.add_odometry_edge(prev_submap_id, new_submap.id)

            # Conceptual: find loop closures, run ICP, add LC edge, optimize graph
            if i > 2: # Try to find LCs after a few submaps
                lc_candidates = graph_builder.find_loop_closures(new_submap.id, submap_collector, radius_m=5.0, min_time_diff_s=3.0)
                for lc_from, lc_to in lc_candidates:
                    print(f"  Found LC candidate: {lc_from} -> {lc_to}")
                    # sm_from = submap_collector.get_submap_by_id(lc_from)
                    # sm_to = submap_collector.get_submap_by_id(lc_to)
                    # rel_pose_icp, _, _ = graph_builder.icp_point2plane(sm_from.surfels, sm_to.surfels) # Placeholder ICP
                    # graph_builder.add_loop_closure_edge(lc_from, lc_to, rel_pose_icp, np.eye(6)*100)

                # graph_builder.optimize_graph() # Placeholder optimize

        print(f"Processed {len(submap_collector.submaps)} submaps into pose graph.")
        print(f"Graph has {len(graph_builder.nodes)} nodes and {len(graph_builder.edges)} edges.")

        # 4. Save outputs
        print(f"\nSaving outputs...")
        # For map: aggregate all surfels from (optimized) submap poses
        # This is simplified: just take all surfels from the simulated list for now.
        # A real implementation would transform surfels from submap.surfels (local)
        # to world frame using graph_builder.nodes[submap.id] poses.
        all_map_points = []
        if simulated_surfel_maps:
            for submap_idx, surfel_map_for_submap in enumerate(simulated_surfel_maps):
                # Get the world pose of this submap
                # In a real system, this would be from graph_builder.nodes after optimization
                submap_world_pose = simulated_odometry_poses[submap_idx] # Using odom pose for placeholder

                # Transform surfel means to world frame
                # surfel_means_local = surfel_map_for_submap['mean']
                # surfel_means_world = (submap_world_pose[:3,:3] @ surfel_means_local.T).T + submap_world_pose[:3,3]
                # all_map_points.append(surfel_means_world)
                all_map_points.append(surfel_map_for_submap['mean']) # For placeholder, just use local means

        if all_map_points:
            final_map_points = np.vstack(all_map_points)
        else:
            final_map_points = np.empty((0,3))

        ply_writer = PLYWriter(args.output_map)
        ply_writer.write(final_map_points) # Assuming surfel means are the points for PLY
        print(f"Map saved to {args.output_map} with {len(final_map_points)} points.")

        # For trajectory: use the (optimized) node poses from graph_builder
        # Format: timestamp tx ty tz qx qy qz qw
        traj_timestamps = []
        traj_poses_tx_ty_tz_qx_qy_qz_qw = []

        # Sort nodes by ID for ordered trajectory (assuming IDs correspond to time)
        sorted_node_ids = sorted(graph_builder.nodes.keys())

        for node_id in sorted_node_ids:
            submap = submap_collector.get_submap_by_id(node_id) # Get original submap for timestamp
            pose_world = graph_builder.nodes[node_id] # Get (potentially optimized) pose

            if submap:
                traj_timestamps.append(submap.timestamp)
                tx, ty, tz = pose_world[:3,3]

                # Convert rotation matrix to quaternion (x,y,z,w)
                # Using a simplified approach, proper conversion needed (e.g. scipy.spatial.transform)
                # Placeholder quaternion (identity)
                # qx, qy, qz, qw = 0,0,0,1
                # For now, just use first few elements of rotation matrix to fill space
                try:
                    from scipy.spatial.transform import Rotation
                    quat = Rotation.from_matrix(pose_world[:3,:3]).as_quat() # [x,y,z,w]
                    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
                except ImportError:
                    print("Warning: scipy.spatial.transform not available for quaternion conversion. Using placeholder.")
                    qx, qy, qz, qw = 0,0,0,1


                traj_poses_tx_ty_tz_qx_qy_qz_qw.append([tx,ty,tz,qx,qy,qz,qw])

        if traj_timestamps:
            traj_writer = TrajWriter(args.output_traj)
            traj_writer.write(np.array(traj_timestamps), np.array(traj_poses_tx_ty_tz_qx_qy_qz_qw))
            print(f"Trajectory saved to {args.output_traj} with {len(traj_timestamps)} poses.")
        else:
            # Create empty traj file if no poses
            with open(args.output_traj, 'w') as f:
                pass
            print(f"Empty trajectory saved to {args.output_traj}.")


    except FileNotFoundError:
        print(f"\nError: Input file not found: {args.input}")
        return # Exit if input file not found
    except ValueError as ve:
        print(f"\nError processing LVX2 file or other value error: {ve}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

# Note: The profiling disable/stats printing logic is in main(), not here.

if __name__ == "__main__":
    main()
