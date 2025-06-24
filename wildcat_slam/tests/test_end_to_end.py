import pytest
import subprocess
import os
import numpy as np
import struct # For creating dummy lvx2

# Helper to create a minimal valid LVX2 file for testing basic CLI execution
def create_dummy_lvx2_file(filepath, num_device_infos=1, num_frames=1):
    # Public Header (24 bytes)
    signature = b"livox_tech\0\0\0\0\0\0"
    version = bytes([2, 0, 0, 0]) # v2.0.0.0
    magic_code = struct.pack('<I', 0xAC0EA767)
    public_header = signature + version + magic_code

    # Private Header (5 bytes)
    frame_duration_ms = 50
    device_count = num_device_infos
    private_header = struct.pack('<I', frame_duration_ms) + struct.pack('<B', device_count)

    # Device Info Block (63 bytes per device)
    device_info_block = b''
    for i in range(device_count):
        lidar_sn = f"LIDARSN{i:07}".encode('ascii')
        lidar_sn_bytes = lidar_sn + b'\0' * (16 - len(lidar_sn))
        hub_sn = f"HUBSN{i:09}".encode('ascii')
        hub_sn_bytes = hub_sn + b'\0' * (16 - len(hub_sn))
        dev_info = lidar_sn_bytes + \
                     hub_sn_bytes + \
                     struct.pack('<I', i) + \
                     struct.pack('<B', 0) + \
                     struct.pack('<B', 9) + \
                     struct.pack('<B', 0) + \
                     struct.pack('<ffffff', 0,0,0,0,0,0) # roll,pitch,yaw,x,y,z
        device_info_block += dev_info

    # Point Cloud Data Block (Frame Headers + Packages) - very minimal
    # This part is complex to simulate fully. For CLI tests, often just having the headers
    # is enough to check if the reader part initializes.
    # The current CLI main() only reads headers and then simulates the rest.
    # So, a very simple frame structure might be okay.
    # If the CLI starts parsing actual frames, this dummy will need more data.
    frames_block = b''
    current_offset_val = 24 + 5 + (63 * device_count) # Initial offset for first frame

    # Add dummy frame data to make file a bit larger and avoid EOF issues if parser tries to read more
    # Frame Header (24 bytes)
    # For a simple test, one frame header might be enough if the CLI doesn't read point data yet
    if num_frames > 0:
        # This is a simplification; real LVX2 files have chained offsets.
        # For a dummy file where we don't parse frames, this is less critical.
        # The CLI currently only reads headers, so frame data is not strictly needed for it to pass.
        # However, if LVX2Reader.read_frames() was called, it would need valid frame headers.
        # For now, let's make it structurally plausible but minimal.
        # The CLI's simulated loop doesn't use the reader to yield frames yet.
        pass # No actual frame data needed since CLI simulates processing

    with open(filepath, "wb") as f:
        f.write(public_header)
        f.write(private_header)
        f.write(device_info_block)
        # f.write(frames_block) # No frame data written as CLI main doesn't read it yet.
    return filepath


@pytest.fixture
def dummy_lvx_filepath(tmp_path):
    return create_dummy_lvx2_file(tmp_path / "dummy_test.lvx2")

def run_cli_command(args_list):
    """Helper to run the CLI script via subprocess."""
    # Assuming wildcat_slam is installed and in PATH, or use python -m wildcat_slam.cli
    # For hermeticity, using python -m is better.
    cmd = ["python", "-m", "wildcat_slam.cli"] + args_list
    # print(f"Running command: {' '.join(cmd)}") # For debugging test
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # print("STDOUT:\n", result.stdout)
    # print("STDERR:\n", result.stderr)
    return result

def test_cli_runs_with_dummy_lvx(dummy_lvx_filepath, tmp_path):
    """Test if the CLI runs with a dummy LVX2 file and creates output files."""
    output_map_ply = tmp_path / "output_map.ply"
    output_traj_txt = tmp_path / "output_traj.txt"

    cli_args = [
        "--input", str(dummy_lvx_filepath),
        "--output_map", str(output_map_ply),
        "--output_traj", str(output_traj_txt),
        "--voxel_size", "0.6", # Example of overriding default
    ]

    result = run_cli_command(cli_args)
    assert result.returncode == 0, f"CLI script failed with error: {result.stderr}"

    assert output_map_ply.exists(), "Output PLY map file was not created."
    assert output_traj_txt.exists(), "Output TXT trajectory file was not created."

    # Basic check on PLY file content (header)
    with open(output_map_ply, 'r') as f:
        ply_header = [next(f) for _ in range(7)] # Read first few lines
    assert ply_header[0].strip() == "ply"
    assert ply_header[1].strip() == "format ascii 1.0"
    assert "element vertex" in ply_header[2]
    # The number of vertices will depend on the simulation in cli.py
    # For now, the simulated part creates 10 "frames" with 5-15 surfels each.
    # So, between 50 and 150 points.
    num_vertices = int(ply_header[2].strip().split()[-1])
    # Simulated surfels are (num_frames_to_simulate * num_sim_surfels_per_frame)
    # num_frames_to_simulate = 10. num_sim_surfels (random 5-15). Min 50, Max 150.
    # This is a loose check due to randomness in CLI simulation.
    assert 0 <= num_vertices <= 15 * 10 , f"Unexpected number of vertices in PLY: {num_vertices}"


    # Basic check on TXT trajectory file content
    # Expect N lines, where N is num_frames_to_simulate (10)
    # Each line: timestamp tx ty tz qx qy qz qw (8 numbers)
    with open(output_traj_txt, 'r') as f:
        traj_lines = f.readlines()

    num_frames_simulated_in_cli = 10 # As per current cli.py simulation
    if num_frames_simulated_in_cli == 0 : # If simulation results in no poses
         assert len(traj_lines) == 0 or (len(traj_lines) == 1 and traj_lines[0].strip() == "")
    else:
        assert len(traj_lines) == num_frames_simulated_in_cli, \
            f"Expected {num_frames_simulated_in_cli} lines in trajectory, got {len(traj_lines)}"
        if traj_lines: # If not empty
            first_line_parts = traj_lines[0].strip().split()
            assert len(first_line_parts) == 8, \
                f"Trajectory line should have 8 parts (ts + pose), got {len(first_line_parts)}"
            # Try to convert to float to ensure format is numeric
            [float(p) for p in first_line_parts]


# Path to the sample LVX file (assuming it's copied to tests/data)
SAMPLE_LVX_PATH = "wildcat_slam/tests/data/indoor_sample.lvx"

@pytest.mark.skipif(not os.path.exists(SAMPLE_LVX_PATH), reason="indoor_sample.lvx not found in tests/data")
def test_cli_runs_with_indoor_sample_lvx(tmp_path):
    """
    Test if the CLI runs with the provided indoor_sample.lvx.
    This is more of an integration smoke test for the LVX2Reader part of the CLI.
    """
    output_map_ply = tmp_path / "indoor_map.ply"
    output_traj_txt = tmp_path / "indoor_traj.txt"

    cli_args = [
        "--input", SAMPLE_LVX_PATH,
        "--output_map", str(output_map_ply),
        "--output_traj", str(output_traj_txt),
    ]

    result = run_cli_command(cli_args)
    # Check if stdout contains expected prints from LVX2Reader
    assert "LVX2 public header:" in result.stdout
    assert "LVX2 private header:" in result.stdout
    assert "LVX2 device info:" in result.stdout

    # For now, the CLI simulation part runs regardless of actual LVX content past headers.
    # So, output files should still be created.
    assert result.returncode == 0, f"CLI script failed with error: {result.stderr}"
    assert output_map_ply.exists(), "Output PLY map file was not created for indoor_sample."
    assert output_traj_txt.exists(), "Output TXT trajectory file was not created for indoor_sample."

    # Content checks similar to the dummy_lvx test
    with open(output_map_ply, 'r') as f:
        ply_header = [next(f) for _ in range(3)]
    assert ply_header[0].strip() == "ply"
    assert "element vertex" in ply_header[2]
    num_vertices = int(ply_header[2].strip().split()[-1])
    assert 0 <= num_vertices <= 15 * 10

    with open(output_traj_txt, 'r') as f:
        traj_lines = f.readlines()
    num_frames_simulated_in_cli = 10
    if num_frames_simulated_in_cli == 0:
        assert len(traj_lines) == 0 or (len(traj_lines) == 1 and traj_lines[0].strip() == "") # Empty file or just a newline
    else:
        assert len(traj_lines) == num_frames_simulated_in_cli
        if traj_lines: # Ensure not empty before trying to access traj_lines[0]
            assert len(traj_lines[0].strip().split()) == 8

def test_cli_profiling_flag(dummy_lvx_filepath, tmp_path):
    """Test if the CLI runs with --profile and produces profiling output."""
    output_map_ply = tmp_path / "profile_map.ply"
    output_traj_txt = tmp_path / "profile_traj.txt"

    cli_args = [
        "--input", str(dummy_lvx_filepath),
        "--output_map", str(output_map_ply),
        "--output_traj", str(output_traj_txt),
        "--profile"
    ]

    result = run_cli_command(cli_args)
    assert result.returncode == 0, f"CLI script with --profile failed: {result.stderr}"

    # Check if profiling output is present in stdout
    assert "Performance profiling enabled" in result.stdout
    assert "--- Performance Profile ---" in result.stdout
    assert "ncalls" in result.stdout # Typical header from pstats output
    assert "tottime" in result.stdout
    assert "cumtime" in result.stdout

    # Ensure output files are still created
    assert output_map_ply.exists(), "Output PLY map file was not created with --profile."
    assert output_traj_txt.exists(), "Output TXT trajectory file was not created with --profile."
