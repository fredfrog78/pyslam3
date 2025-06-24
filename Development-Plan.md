Below is a test‐driven development (TDD) plan for a Python‐only, NumPy/SciPy‐centric implementation of Wildcat SLAM. It is structured into incremental “iterations.” Each iteration introduces a small, testable piece of functionality, provides a suite of unit‐ and integration‐tests (using pytest and numpy.testing), and culminates in a working prototype of that piece. Over the course of the iterations you will arrive at a complete, real‐time‐capable lidar–inertial SLAM pipeline that reads LVX2 files and emits a PLY and a trajectory TXT, driven entirely via command‐line options.
Directory layout (suggestion)
──────────────────────────────────────────────────
wildcat_slam/
 src/
  wildcat_slam/
   init.py
   io.py # LVX2 reader, PLY writer, TXT writer
   geometry.py # SO(3), SE(3) helpers, interpolation
   surfel.py # voxel‐grid clustering & surfel extraction
   odometry.py # CT odometry sliding‐window
   pose_graph.py # submap, graph builder + optimiser
   cli.py # command‐line interface (argparse)
 tests/
  data/
   indoor_sample.lvx # provided 60‐frame LVX2 file
   synthetic_imu.npy # small synthetic IMU timeseries
   synthetic_lidar.npy # synthetic 2‐ or 3‐frame pointclouds
  test_io.py
  test_geometry.py
  test_surfel.py
  test_odometry.py
  test_pose_graph.py
  test_end_to_end.py
setup.py
readme.md
Common tooling
────────────
• pytest for tests
• numpy.testing for numerical assertions (assert_allclose, etc.)
• flake8 / black for linting and style
Iteration 1: LVX2 I/O and CLI stub
────────────────────────────────
Goal
• Parse a minimal LVX2 file header & device info
• CLI stub that accepts input/output paths and prints summary
Tasks
io.py::LVX2Reader class with
– read_public_header(): verify signature & magic code, expose version
– read_private_header(): expose frame_duration & device_count
– read_device_info(): collect list of device dicts
io.py::PLYWriter.write(points: (N×3) float, colors: optional)
io.py::TrajWriter.write(timestamps, poses)
cli.py with argparse flags: –input, –output_map, –output_traj
Tests (tests/test_io.py)
• test_read_public_header(): use a handcrafted 24‐byte header (bytes
start with b"livox_tech\0\0\0\0\0\0", magic=0xAC0EA767). assert Reader.version.
• test_read_private_header(): create a MemoryFile with known frame_dur(50), device_count(2).
• test_read_device_info(): append two 63‐byte structs with known SNs & extrinsics. assert dict fields.
• test_ply_traj_writer(): round‐trip: write a small temp file then read back with a simple ASCII check.
Iteration 2: Geometry primitives & Continuous‐time splines
────────────────────────────────────────────────────────
Goal
• Implement small SE(3) utility functions and CT interpolation
• Ensure fast, batch‐wise routines (no Python for‐loops in core)
Tasks
geometry.py:
– so3_hat, so3_vee
– so3_exp, so3_log (using scipy.linalg.expm/logm or Rodrigues)
– se3_exp, se3_log
– rotation interpolation: RotInterpolate(r1_vec3, r2_vec3, α)
– linear interpolation of translations
– Cubic‐B‐spline class: given n+1 control‐poses at times, evaluate pose(t)
use vectorised matrix‐operations wherever possible
Tests (tests/test_geometry.py)
• test_hat_vee_roundtrip(): for random ω ∈R3 assert vee(hat(ω)) ≈ ω
• test_so3_exp_log(): for random ω small magnitude, assert log(exp(hat(ω))) ≈ hat(ω)
• test_rot_interpolate(): α=0,1 case, and α=0.5 halfway quaternion check
• test_spline_endpoints(): spline(t0) == ctrl_pose0, spline(tn) == ctrl_osen0
• test_spline_midpoint(): consistency with manual de Boor
Iteration 3: Surfel extraction
────────────────────────────
Goal
• From one “frame” of points (N×3 float + per‐point timestamp) build surfels
• Grid‐based clustering + eigen‐analysis to yield (pos, normal, score)
Tasks
surfel.py::extract_surfels(points: (N×3), times: (N,), voxel_size, time_window, planarity_thresh)
– assign each point to voxel: integer division (vectorised)
– sort/segment‐by‐voxel+time chunk, aggregate each cluster via np.mean/cov
– eigen‐decomp of covariance (np.linalg.eigh) → compute planarity score
– keep surfels where score > thresh
– return structured array surfels[(x,y,z),(nx,ny,nz),score,timestamp_mean,resolution]
Tests (tests/test_surfel.py)
• synthetic cube: generate 1000 points on a plane + noise; timestamps random in one frame
– assert surfels ≈ plane params, normals within small angle
• test_planarity_rejection(): cluster of random sphere pts → planarity_score near 0 → rejected
• test_voxel_clustering_counts(): known points at corners of voxels → known number of clusters
Iteration 4: IMU‐only pose init & surfel correspondence
────────────────────────────────────────────────────
Goal
• From IMU timeseries, build initial discrete poses by pre‐integrating (basic Euler)
• For surfel maps in sliding window, interpolate initial poses for each surfel timestamp
• Build kNN correspondences (e.g. sklearn.neighbors or KDTree from SciPy)
Tasks
odometry.py::IMUIntegrator:
– feed imu samples (t_i, acc_i, angvel_i), integrate forward at 100 Hz to produce {R(t), t(t)}
– simple Euler integration; tests only small gravity removal
odometry.py::initial_pose_interpolation(imu_poses, query_times) → list of poses
– using geometry.RotInterpolate + LinInterpolate
odometry.py::match_surfels(surfels_current_window, k=1, time_gap_thresh) → list of (i,j)
– kd‐tree on 7‐D descriptors (x,y,z,nx,ny,nz,res), mutual nearest‐neighbours; reject close‐time pairs
Tests (tests/test_odometry.py)
• synthetic IMU: generate stationary imu samples with known g; integrator→ final pos ≈ 0, orientation≈Id
• test_initial_interpolation(): two known imu poses t0,t1 and mid‐timestamp query yields average pose
• synthetic surfels: two sets of identical surfel arrays offset by a known translation; match→ correct pairs
Iteration 5: Local CT odometry optimisation
────────────────────────────────────
Goal
• Implement sliding‐window CT‐optimisation: form GN linear system of eqn (2) with residuals (6),(9),(10),(11), optimise using scipy.sparse and scipy.sparse.linalg.solve
Tasks
odometry.py::OdometryWindow class:
– holds imu_poses, surfels, matches M, sample‐poses rcor_i,tcor_i (init zeros)
– assemble Jacobians + residuals for IMU cost fτ_imu and surfel cost f_s,s’
– build least‐squares normal‐equations (dense or sparse)
– IRLS weight update (Cauchy M‐estimator)
– update sample‐poses, then project onto IMU timeline via CT spline (Eqn 3)
Integrate iterative surfel‐pose loop (5–10 iterations) until convergence
Tests (tests/test_odometry.py)
• tiny window with 3 imu‐poses, 2 surfel matches under known rigid transform; optimise→ recovers transform
• robust to outliers: add one bad match → Cauchy‐based IRLS downweights it
• missing lidar frames: call optimise() with zero surfels → IMU‐only smoothing still runs
Iteration 6: Submap generation & pose‐graph
────────────────────────────────────────
Goal
• After every K seconds, generate submap: bundle of poses + surfels
• Retain submaps, build odometry edges (between consecutive submaps)
Tasks
pose_graph.py::SubmapCollector: take (time, odom_pose, surfels) → new Submap object
pose_graph.py::GraphBuilder:
– nodes = submap IDs → SE(3) poses
– edges: odom edges from odom relative‐pose between submap frames
Tests (tests/test_pose_graph.py)
• feed 3 submaps with known odometry → graph with 2 edges; relative‐pose matches ground‐truth
• test_merge_redundant(): if two submaps overlap heavily and Mahalanobis distance small, they merge→ one node
Iteration 7: Loop‐closure & global optimisation
──────────────────────────────────────────────
Goal
• Detect candidate LC via Mahalanobis distance on node poses
• For selected pair, compute relative pose via point‐to‐plane ICP on surfel maps
• Add LC edge, solve global pose‐graph with gravity‐alignment term (16) via GN
Tasks
pose_graph.py::find_loop_closures(threshold) using KD‐tree on node‐positions
pose_graph.py::icp_point2plane(src_surfels, tgt_surfels) →∂‐pose + covariance
pose_graph.py::optimize_graph(): assemble linear system for all edges + up‐term, solve via scipy.sparse
Tests (tests/test_pose_graph.py)
• toy square trajectory: revisit start → Mahalanobis picks up LC, ICP recovers known loop‐closure transform
• optimize_graph(): after adding small random drift to nodes and LC edge, optimize→ reduces drift
Iteration 8: End‐to‐end integration & CLI wiring
────────────────────────────────────────────
Goal
• Wire together io→ odometry→ submap→ pose_graph→ outputs
• Command‐line parameters: voxel_size, window_size, iteration_count, frame_rate, output file paths
Tasks
cli.py::main(): parse args, instantiate LVX2Reader, OdometryRunner, PoseGraphRunner, run loop
Periodically write intermediate PLY/trajectory if requested
Tests (tests/test_end_to_end.py)
• using tests/data/indoor_sample.lvx (60 frames), run wildcat_slam –input … –out_map tmp.ply –out_traj tmp.txt
– assert tmp.ply exists, has ~60×surfel_count lines (header OK)
– assert tmp.txt exists, lines = number_of_submaps, each line format matches regex for time + 7‐quat fields
• synthetic too: very small synthetic LVX2 constructed from tests/data/synthetic_*.npy → pipeline runs without crash
Iteration 9: Performance profiling & final tweaks
────────────────────────────────────────
Goal
• Profile hot‐spots via cProfile; confirm <5 ms per frame for odometry on 60 k points
• Ensure <2 G memory for 60‐frame run
Tasks
add optional –profile flag to CLI to print timing summary
vectorise any remaining for‐loops in surfel & odometry maths
finalize dependency list: numpy, scipy, pytest, optionally scikit‐sparse if needed
Deliverable
──────────
By the end of Iteration 9 you will have:
• A full Python package “wildcat_slam” with minimal deps: numpy & scipy and pytest for dev
• A CLI tool wildcat_slam that processes LVX2 → PLY + TXT in real time (<20 Hz)
• A comprehensive pytest suite covering every component and end‐to‐end tests

