# Wildcat SLAM Implementation

This project is a Python-based implementation of the Wildcat SLAM algorithm, following the test-driven development plan outlined in `Development-Plan.md`. It focuses on creating a real-time capable lidar-inertial SLAM pipeline that processes LVX2 files and outputs a PLY map and a trajectory TXT file.

## Overview of the Algorithm

Wildcat SLAM combines a robust real-time lidar-inertial odometry module using a continuous-time trajectory representation with an efficient pose-graph optimization module. This implementation aims to replicate its core concepts, including surfel-based mapping, continuous-time trajectory optimization for odometry, and global pose-graph SLAM for large-scale consistency.

Key components include:
- **LVX2 Parsing**: Reading lidar and IMU data from LVX2 files.
- **Geometry Primitives**: SO(3)/SE(3) operations and trajectory interpolation (e.g., B-splines).
- **Surfel Extraction**: Clustering point clouds into planar surfel features.
- **Lidar-Inertial Odometry**: Estimating ego-motion in a sliding window by optimizing a continuous-time trajectory against IMU measurements and surfel correspondences.
- **Pose Graph SLAM**: Building a graph of submaps, detecting loop closures, and performing global optimization to maintain a consistent map and trajectory.

## Current Status & Implemented Features

This implementation has progressed through the initial 9 iterations of the development plan. The foundational structure and many key components are in place, though some core algorithmic parts (especially within odometry optimization and pose graph optimization) are currently placeholders.

**Implemented Iterations:**
-   **Iteration 1: LVX2 I/O and CLI stub**
    -   LVX2 file header parsing (`LVX2Reader`).
    -   PLY and TXT output writers (`PLYWriter`, `TrajWriter`).
    -   Basic Command Line Interface (`cli.py`) with input/output arguments.
-   **Iteration 2: Geometry primitives & Continuous-time splines**
    -   SO(3) and SE(3) Lie algebra helper functions (`so3_hat`/`vee`, `so3_exp`/`log`, etc.).
    -   Rotation and translation interpolation functions.
    -   Placeholder `CubicBSpline` class structure.
-   **Iteration 3: Surfel extraction**
    -   `extract_surfels` function implementing voxel-based clustering, mean/covariance calculation, eigen-decomposition for normal and planarity score calculation, and filtering.
-   **Iteration 4: IMU-only pose init & surfel correspondence**
    -   `IMUIntegrator` class for basic Euler integration of IMU measurements.
    -   `initial_pose_interpolation` function using implemented geometry primitives.
    -   Placeholder `match_surfels` function.
-   **Iteration 5: Local CT odometry optimisation**
    -   `OdometryWindow` class structure.
    -   Placeholder `optimize_window` method including structure for Gauss-Newton iterations and IRLS, but using dummy Jacobians and residuals.
    -   Cauchy M-estimator for weights.
    -   Placeholder update of estimated IMU trajectory from sample poses.
-   **Iteration 6: Submap generation & pose-graph**
    -   `Submap` class to store submap data (ID, pose, surfels, gravity).
    -   `SubmapCollector` to manage and create submaps.
    -   `GraphBuilder` class with methods to add nodes (from submaps) and odometry edges (calculating relative poses). Placeholders for `optimize_graph` and `merge_redundant_nodes`.
-   **Iteration 7: Loop-closure & global optimisation**
    -   `GraphBuilder.find_loop_closures()` implemented using KD-Tree for spatial search and filtering by time difference.
    -   Placeholder `GraphBuilder.icp_point2plane()` method.
    -   `GraphBuilder.add_loop_closure_edge()` method.
-   **Iteration 8: End-to-end integration & CLI wiring**
    -   Updated `cli.py` to include more parameters (voxel_size, odometry settings).
    -   Simulated end-to-end data flow: LVX2 reading (headers) -> Simulated Odometry (poses, surfels) -> Submap Creation -> Pose Graph Building (nodes, odom edges, LC candidates).
    -   CLI produces output PLY map and TXT trajectory files based on this simulated run.
    -   Improved robustness of `LVX2Reader` for ASCII decoding.
-   **Iteration 9: Performance profiling & final tweaks**
    -   Added `--profile` flag to `cli.py` for enabling `cProfile` and printing performance statistics.
    -   Reviewed code for obvious vectorization opportunities (most numerical parts already use NumPy).
    -   Finalized core dependencies (`numpy`, `scipy`) in `setup.py`.

## Directory Structure

```
wildcat_slam/
├── setup.py
├── readme.md
├── src/
│   └── wildcat_slam/
│       ├── __init__.py
│       ├── cli.py          # Command-line interface
│       ├── geometry.py     # SO(3), SE(3) helpers, interpolation, splines
│       ├── io.py           # LVX2 reader, PLY writer, TXT writer
│       ├── odometry.py     # CT odometry sliding-window, IMU integration
│       ├── pose_graph.py   # Submap, graph builder + optimizer
│       └── surfel.py       # Voxel-grid clustering & surfel extraction
└── tests/
    ├── data/
    │   └── indoor_sample.lvx # Sample LVX2 file for testing
    ├── test_end_to_end.py
    ├── test_geometry.py
    ├── test_io.py
    ├── test_odometry.py
    ├── test_pose_graph.py
    └── test_surfel.py
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd wildcat_slam_project_directory
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install the package and its dependencies:**
    For regular use:
    ```bash
    pip install ./wildcat_slam
    ```
    For development (editable install):
    ```bash
    pip install -e ./wildcat_slam[dev]
    ```
    The `[dev]` option includes `pytest`, `flake8`, and `black` for testing and linting.

## Usage

The primary way to run the Wildcat SLAM implementation is through its command-line interface.

```bash
wildcat_slam --input path/to/your/file.lvx2 --output_map path/to/save/map.ply --output_traj path/to/save/trajectory.txt [OPTIONS]
```

**Key Options:**
*   `--input <filepath>`: (Required) Path to the input LVX2 file.
*   `--output_map <filepath>`: (Required) Path to save the output map in PLY format.
*   `--output_traj <filepath>`: (Required) Path to save the output trajectory in TXT format.
*   `--voxel_size <float>`: Voxel size for surfel extraction (default: 0.5).
*   `--odometry_window_duration <float>`: Duration of the odometry sliding window in seconds (default: 2.0).
*   `--odometry_imu_frequency <float>`: IMU frequency for odometry (default: 100.0).
*   `--odometry_num_samples <int>`: Number of sample poses in the odometry window (default: 10).
*   `--profile`: Enable cProfile for performance profiling and print summary.

**Example:**
```bash
wildcat_slam --input ./tests/data/indoor_sample.lvx \
             --output_map ./output/map.ply \
             --output_traj ./output/trajectory.txt \
             --voxel_size 0.4
```
*(Note: The current implementation uses simulated processing beyond LVX2 header reading. Output files will reflect this simulated data.)*

## How to Run Tests

Tests are written using `pytest`. To run the tests:

1.  Ensure you have installed the development dependencies: `pip install -e ./wildcat_slam[dev]`
2.  Navigate to the `wildcat_slam` directory (where `setup.py` is located).
3.  Run pytest:
    ```bash
    python -m pytest
    ```
    Or, to run tests for a specific file:
    ```bash
    python -m pytest tests/test_io.py
    ```

## Future Work / Roadmap

The current implementation provides a structural backbone. Significant work remains to fully implement the core algorithms:

-   **LVX2 Frame Parsing**: Implement full parsing of point cloud data from LVX2 frames in `LVX2Reader`.
-   **OdometryWindow Optimization**:
    -   Implement actual IMU residual and Jacobian calculations.
    -   Implement surfel matching residual and Jacobian calculations.
    -   Refine the `_update_estimated_imu_trajectory_from_samples` method, potentially using a fully functional `CubicBSpline` class.
-   **Surfel Matching**: Implement `match_surfels` in `odometry.py` using KD-trees and mutual nearest-neighbor logic.
-   **ICP Implementation**: Fully implement `GraphBuilder.icp_point2plane()`.
-   **Global Graph Optimization**: Implement the solver logic in `GraphBuilder.optimize_graph()` including odometry, loop closure, and gravity alignment terms.
-   **Node Merging**: Implement `GraphBuilder.merge_redundant_nodes()`.
-   **Performance**: Further profiling and optimization once core algorithms are complete, aiming for real-time performance targets.

This project follows the iterations outlined in `Development-Plan.md`. Refer to it for more detailed tasks for upcoming iterations.

## Dependencies
- numpy
- scipy

Development dependencies:
- pytest
- flake8
- black

*(License and Contribution sections are omitted as they require external decisions.)*
