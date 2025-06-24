import pytest
import struct
import os
import numpy as np
from wildcat_slam.io import LVX2Reader, PLYWriter, TrajWriter

# Helper to create dummy LVX2 file content
def create_lvx2_content(signature="livox_tech", version=(2,0,0,0), magic_code=0xAC0EA767,
                        frame_duration=50, device_count=1, device_infos_override=None):
    sig_bytes = signature.encode('ascii')
    sig_bytes += b'\0' * (16 - len(sig_bytes))
    ver_bytes = bytes(version)
    magic_bytes = struct.pack('<I', magic_code)

    pub_header = sig_bytes + ver_bytes + magic_bytes

    priv_header = struct.pack('<I', frame_duration) + struct.pack('<B', device_count)

    dev_infos_bytes = b''
    actual_device_infos = []
    if device_infos_override is not None:
        actual_device_infos = device_infos_override
    elif device_count > 0: # Create default if no override and count > 0
        for i in range(device_count):
            actual_device_infos.append({
                'lidar_sn_code': f"LIDARSN{i:03}", 'hub_sn_code': f"HUBSN{i:03}", 'lidar_id': i + 1,
                'lidar_type': 0, 'device_type': 9, 'extrinsic_enable': 1,
                'roll': 0.1 * (i + 1), 'pitch': 0.2 * (i + 1), 'yaw': 0.3 * (i + 1),
                'x': 0.4 * (i + 1), 'y': 0.5 * (i + 1), 'z': 0.6 * (i + 1)
            })

    # Ensure device_infos_override matches device_count if provided
    if device_infos_override is not None and len(device_infos_override) != device_count:
        raise ValueError("device_infos_override length must match device_count if provided")

    for dev_info in actual_device_infos:
        # Ensure all keys are present, providing defaults if necessary (e.g. for empty dicts in override)
        sn_str = dev_info.get('lidar_sn_code', f"DEF{actual_device_infos.index(dev_info):03}")
        hub_sn_str = dev_info.get('hub_sn_code', f"DEFHUB{actual_device_infos.index(dev_info):03}")
        lidar_id_val = dev_info.get('lidar_id', actual_device_infos.index(dev_info) + 1)
        lidar_type_val = dev_info.get('lidar_type', 0)
        device_type_val = dev_info.get('device_type', 9)
        extrinsic_enable_val = dev_info.get('extrinsic_enable', 1)
        roll_val = dev_info.get('roll', 0.0)
        pitch_val = dev_info.get('pitch', 0.0)
        yaw_val = dev_info.get('yaw', 0.0)
        x_val = dev_info.get('x', 0.0)
        y_val = dev_info.get('y', 0.0)
        z_val = dev_info.get('z', 0.0)

        sn = sn_str.encode('ascii')
        sn_bytes = sn + b'\0' * (16 - len(sn))
        hub_sn_ = hub_sn_str.encode('ascii')
        hub_sn_bytes = hub_sn_ + b'\0' * (16 - len(hub_sn_))

        dev_infos_bytes += sn_bytes
        dev_infos_bytes += hub_sn_bytes
        dev_infos_bytes += struct.pack('<I', lidar_id_val)
        dev_infos_bytes += struct.pack('<B', lidar_type_val)
        dev_infos_bytes += struct.pack('<B', device_type_val)
        dev_infos_bytes += struct.pack('<B', extrinsic_enable_val)
        dev_infos_bytes += struct.pack('<f', roll_val)
        dev_infos_bytes += struct.pack('<f', pitch_val)
        dev_infos_bytes += struct.pack('<f', yaw_val)
        dev_infos_bytes += struct.pack('<f', x_val)
        dev_infos_bytes += struct.pack('<f', y_val)
        dev_infos_bytes += struct.pack('<f', z_val)

    return pub_header + priv_header + dev_infos_bytes

@pytest.fixture
def temp_lvx2_file(tmp_path):
    def _create_file(content):
        filepath = tmp_path / "test.lvx2"
        with open(filepath, "wb") as f:
            f.write(content)
        return filepath
    return _create_file

def test_read_public_header_valid(temp_lvx2_file):
    content = create_lvx2_content()
    filepath = temp_lvx2_file(content)
    with LVX2Reader(filepath) as reader:
        assert reader.public_header['signature'] == "livox_tech"
        assert reader.public_header['version'] == (2,0,0,0)
        assert reader.public_header['magic_code'] == 0xAC0EA767

def test_read_public_header_invalid_signature(temp_lvx2_file):
    content = create_lvx2_content(signature="invalid_sig")
    filepath = temp_lvx2_file(content)
    with pytest.raises(ValueError, match="Invalid LVX2 file signature"):
        with LVX2Reader(filepath) as reader:
            pass # Exception happens in __enter__

def test_read_public_header_invalid_magic_code(temp_lvx2_file):
    content = create_lvx2_content(magic_code=0x12345678)
    filepath = temp_lvx2_file(content)
    with pytest.raises(ValueError, match="Invalid LVX2 magic code"):
        with LVX2Reader(filepath) as reader:
            pass

def test_read_private_header_valid(temp_lvx2_file):
    frame_dur = 100
    dev_count = 2
    # Now create_lvx2_content will generate default full device_infos if device_infos_override is not given
    # or if it's given as [{}] empty dicts, it will fill them with defaults.
    content = create_lvx2_content(frame_duration=frame_dur, device_count=dev_count, device_infos_override=[{}, {}])
    filepath = temp_lvx2_file(content)
    with LVX2Reader(filepath) as reader:
        assert reader.private_header['frame_duration'] == frame_dur
        assert reader.private_header['device_count'] == dev_count

def test_read_device_info_single(temp_lvx2_file):
    dev_info_data = {
        'lidar_sn_code': "TESTSN01", 'hub_sn_code': "TESTHUB01", 'lidar_id': 7,
        'lidar_type': 1, 'device_type': 10, 'extrinsic_enable': 0,
        'roll': 1.1, 'pitch': 1.2, 'yaw': 1.3,
        'x': 1.4, 'y': 1.5, 'z': 1.6
    }
    content = create_lvx2_content(device_count=1, device_infos_override=[dev_info_data])
    filepath = temp_lvx2_file(content)
    with LVX2Reader(filepath) as reader:
        assert len(reader.device_info_list) == 1
        read_dev_info = reader.device_info_list[0]
        assert read_dev_info['lidar_sn_code'] == dev_info_data['lidar_sn_code']
        assert read_dev_info['hub_sn_code'] == dev_info_data['hub_sn_code']
        assert read_dev_info['lidar_id'] == dev_info_data['lidar_id']
        assert read_dev_info['device_type'] == dev_info_data['device_type']
        assert read_dev_info['extrinsic_enable'] == dev_info_data['extrinsic_enable']
        assert pytest.approx(read_dev_info['roll']) == dev_info_data['roll']
        assert pytest.approx(read_dev_info['pitch']) == dev_info_data['pitch']
        assert pytest.approx(read_dev_info['yaw']) == dev_info_data['yaw']
        assert pytest.approx(read_dev_info['x']) == dev_info_data['x']
        assert pytest.approx(read_dev_info['y']) == dev_info_data['y']
        assert pytest.approx(read_dev_info['z']) == dev_info_data['z']

def test_read_device_info_multiple(temp_lvx2_file):
    dev_infos_data = [
        { 'lidar_sn_code': "SN01", 'hub_sn_code': "HUB01", 'lidar_id': 1, 'lidar_type':0, 'device_type': 9, 'extrinsic_enable': 1, 'roll': .1, 'pitch': .2, 'yaw': .3, 'x': .4, 'y': .5, 'z': .6 },
        { 'lidar_sn_code': "SN02", 'hub_sn_code': "HUB02", 'lidar_id': 2, 'lidar_type':0, 'device_type': 10, 'extrinsic_enable': 0, 'roll': .7, 'pitch': .8, 'yaw': .9, 'x': 1., 'y': 1.1, 'z': 1.2 }
    ]
    content = create_lvx2_content(device_count=2, device_infos_override=dev_infos_data)
    filepath = temp_lvx2_file(content)
    with LVX2Reader(filepath) as reader:
        assert len(reader.device_info_list) == 2
        for i, dev_info_data in enumerate(dev_infos_data):
            read_dev_info = reader.device_info_list[i]
            assert read_dev_info['lidar_sn_code'] == dev_info_data['lidar_sn_code']
            assert read_dev_info['lidar_id'] == dev_info_data['lidar_id']

def test_read_device_info_zero_devices(temp_lvx2_file):
    content = create_lvx2_content(device_count=0, device_infos_override=[])
    filepath = temp_lvx2_file(content)
    with LVX2Reader(filepath) as reader:
        assert len(reader.device_info_list) == 0

def test_ply_writer_basic(tmp_path):
    filepath = tmp_path / "test.ply"
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    writer = PLYWriter(filepath)
    writer.write(points)

    with open(filepath, 'r') as f:
        content = f.readlines()

    assert content[0] == "ply\n"
    assert content[1] == "format ascii 1.0\n"
    assert content[2] == f"element vertex {points.shape[0]}\n"
    assert "property float x\n" in content
    assert "property float y\n" in content
    assert "property float z\n" in content
    assert "end_header\n" in content
    # Check data lines
    data_lines = [l.strip() for l in content if not l.startswith("ply") and not l.startswith("format") and not l.startswith("element") and not l.startswith("property") and not l.startswith("end_header")]
    assert len(data_lines) == points.shape[0]
    assert data_lines[0] == "1.000000 2.000000 3.000000" # Adjusted for formatting
    assert data_lines[1] == "4.000000 5.000000 6.000000" # Adjusted for formatting

def test_ply_writer_with_colors(tmp_path):
    filepath = tmp_path / "test_color.ply"
    points = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    colors = np.array([[100, 150, 200]], dtype=np.uint8)
    writer = PLYWriter(filepath)
    writer.write(points, colors)

    with open(filepath, 'r') as f:
        content = f.readlines()

    assert "property uchar red\n" in content
    assert "property uchar green\n" in content
    assert "property uchar blue\n" in content
    data_lines = [l.strip() for l in content if not l.startswith("ply") and not l.startswith("format") and not l.startswith("element") and not l.startswith("property") and not l.startswith("end_header")]
    assert data_lines[0] == "0.100000 0.200000 0.300000 100 150 200" # Adjusted for formatting

def test_ply_writer_invalid_input(tmp_path):
    filepath = tmp_path / "test_invalid.ply"
    writer = PLYWriter(filepath)
    with pytest.raises(ValueError, match="Points must be an N x 3 numpy array."):
        writer.write(np.array([1,2,3]))
    with pytest.raises(ValueError, match="Colors must be an N x 3 numpy array"):
        writer.write(np.array([[1.,2.,3.]]), colors=np.array([255,0,0]))
    with pytest.raises(ValueError, match="Colors must be uint8"):
        writer.write(np.array([[1.,2.,3.]]), colors=np.array([[255,0,0]], dtype=np.float32))


def test_traj_writer_basic(tmp_path):
    filepath = tmp_path / "test_traj.txt"
    timestamps = np.array([1000.1, 1000.2])
    poses = np.array([
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], # x,y,z, qx,qy,qz,qw
        [4.0, 5.0, 6.0, 0.707, 0.0, 0.0, 0.707]
    ], dtype=np.float64)
    writer = TrajWriter(filepath)
    writer.write(timestamps, poses)

    with open(filepath, 'r') as f:
        content = f.readlines()

    assert len(content) == 2
    assert content[0].strip() == "1000.1 1.0 2.0 3.0 0.0 0.0 0.0 1.0"
    assert content[1].strip() == "1000.2 4.0 5.0 6.0 0.707 0.0 0.0 0.707"

def test_traj_writer_invalid_input(tmp_path):
    filepath = tmp_path / "test_traj_invalid.txt"
    writer = TrajWriter(filepath)
    with pytest.raises(ValueError, match="Timestamps and poses must have the same length."):
        writer.write(np.array([1.0]), np.array([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])) # Mismatched lengths
    with pytest.raises(ValueError, match="Poses must be an N x 7 numpy array."):
        # Correct length (7 elements), but poses is 1D instead of 2D
        writer.write(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), np.array([1,2,3,4,5,6,7]))
    with pytest.raises(ValueError, match="Poses must be an N x 7 numpy array."):
        # Correct length, 2D, but not N x 7
        writer.write(np.array([1.0]), np.array([[1,2,3,4,5,6]])) # N=1, but inner array has 6 elements


# It's good practice to also test file opening errors for LVX2Reader,
# but that might be more involved if not using a real file path.
# For now, focus on content parsing.

# A simple test for the CLI stub to ensure it runs.
# This requires the dummy_sample.lvx2 created by io.py's main block or a similar mechanism
# For isolated testing, it might be better to create a fixture for a dummy lvx2 file.
def test_cli_runs_basic(tmp_path, monkeypatch):
    import sys # Import sys here
    dummy_lvx_content = create_lvx2_content(device_count=0) # Simplest valid file
    input_file = tmp_path / "dummy_cli.lvx2"
    with open(input_file, "wb") as f:
        f.write(dummy_lvx_content)

    output_map = tmp_path / "map.ply"
    output_traj = tmp_path / "traj.txt"

    # Use monkeypatch to simulate command line arguments
    args = ["--input", str(input_file), "--output_map", str(output_map), "--output_traj", str(output_traj)]

    # If cli.main() is directly callable:
    from wildcat_slam.cli import main as cli_main

    # Capture stdout to check output (optional, but good for CLI tests)
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Simulate running `python -m wildcat_slam.src.wildcat_slam.cli --input ...`
    # This is a bit tricky due to how __main__ is handled.
    # A more robust way would be to use subprocess, but for a simple stub,
    # directly calling main after patching sys.argv is common.

    monkeypatch.setattr(sys, "argv", ["cli.py"] + args) # sys.argv[0] is script name

    cli_main() # Call the main function of cli.py

    sys.stdout = sys.__stdout__ # Reset stdout
    output = captured_output.getvalue()

    assert f"Input LVX2 file: {input_file}" in output
    assert f"Output map PLY file: {output_map}" in output
    assert f"Output trajectory TXT file: {output_traj}" in output
    # Check for absence of common error indicators and presence of completion messages
    assert "Error" not in output
    assert "Exception" not in output
    # Check for a positive indication of processing, even if dummy.
    # Based on current cli.py, it will try to read frames and might print about it.
    # If read_frames yields nothing (as it would for a header-only file),
    # it might then print about saving an empty map/trajectory.
    assert "Map PLY saved to" in output or "Trajectory saved to" in output or "No frames found or processed" in output

def test_cli_file_not_found(tmp_path, monkeypatch, capsys):
    import sys # Import sys here
    # capsys is a pytest fixture to capture stdout/stderr
    input_file = tmp_path / "non_existent.lvx2"
    output_map = tmp_path / "map.ply"
    output_traj = tmp_path / "traj.txt"

    args = ["--input", str(input_file), "--output_map", str(output_map), "--output_traj", str(output_traj)]

    from wildcat_slam.cli import main as cli_main
    monkeypatch.setattr(sys, "argv", ["cli.py"] + args)

    cli_main()

    captured = capsys.readouterr()
    assert f"Error: Input file not found: {input_file}" in captured.out

def test_cli_invalid_lvx_file(tmp_path, monkeypatch, capsys):
    import sys # Import sys here
    input_file = tmp_path / "invalid.lvx2"
    with open(input_file, "wb") as f:
        f.write(b"this is not an lvx file") # Invalid content

    output_map = tmp_path / "map.ply"
    output_traj = tmp_path / "traj.txt"

    args = ["--input", str(input_file), "--output_map", str(output_map), "--output_traj", str(output_traj)]

    from wildcat_slam.cli import main as cli_main
    monkeypatch.setattr(sys, "argv", ["cli.py"] + args)

    cli_main()

    captured = capsys.readouterr()
    # The error message from cli.py's except block is "Error processing LVX2 file or other value error: {e}"
    # And the e from LVX2Reader is "Invalid LVX2 file signature: expected b'livox_tech' prefix, got ..."
    expected_error_snippet = "Error processing LVX2 file or other value error: Invalid LVX2 file signature" # Check for the key part.
    assert expected_error_snippet in captured.out

def test_read_frames_from_handcrafted_file(temp_lvx2_file):
    # --- Create Handcrafted LVX2 Content ---
    # 1. Headers
    device_count = 1
    dev_info_data = {
        'lidar_sn_code': "SNFRAME01", 'hub_sn_code': "HUBFRAME01", 'lidar_id': 1,
        'lidar_type': 0, 'device_type': 9, 'extrinsic_enable': 0, # Extrinsic disabled for simplicity
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0
    }
    header_content = create_lvx2_content(device_count=device_count, device_infos_override=[dev_info_data])

    # Calculate offset of first frame
    first_frame_offset = len(header_content) # 24 (public) + 5 (private) + 63*1 (device_info)

    # --- Frame 1 ---
    frame1_index = 0
    frame1_pkg1_ts = 1000000000  # ns
    frame1_pkg1_points_mm = [
        (1000, 2000, 3000, 101), # x,y,z (mm), refl
        (1500, 2500, 3500, 102)
    ]
    frame1_pkg1_data_type1 = b''
    for p in frame1_pkg1_points_mm:
        frame1_pkg1_data_type1 += struct.pack('<iiiB', p[0], p[1], p[2], p[3]) + b'\x00' # Add tag byte

    # Package Header: Version(B), LiDAR ID(I), LiDAR_Type(B), Timestamp Type(B),
    # Timestamp(Q), Udp Counter(H), Data Type(B), Length(I), Frame_Counter(B)
    # These are 9 fields, total 1+4+1+1+8+2+1+4+1 = 23 bytes.
    # Then, 4 reserved bytes follow. Total 27 bytes.
    frame1_pkg1_header_fields = struct.pack('<BIBBQHBIB',
                                            0,  # Version
                                            1,  # LiDAR ID
                                            0,  # LiDAR Type (Reserved)
                                            0,  # Timestamp Type
                                            frame1_pkg1_ts,  # Timestamp (ns)
                                            0,  # UDP Counter
                                            1,  # Data Type (Type 1: 14 bytes/pt)
                                            len(frame1_pkg1_data_type1),  # Length of point data
                                            0   # Frame Counter (Reserved)
                                            )
    frame1_pkg1_header = frame1_pkg1_header_fields + b'\x00\x00\x00\x00' # Reserved 4 bytes

    frame1_pkg2_ts = 2000000000  # ns
    frame1_pkg2_points_cm = [
        (100, 200, 300, 201) # x,y,z (cm), refl
    ]
    frame1_pkg2_data_type2 = b''
    for p in frame1_pkg2_points_cm:
        frame1_pkg2_data_type2 += struct.pack('<hhhB', p[0], p[1], p[2], p[3]) + b'\x00' # Add tag byte

    frame1_pkg2_header_fields = struct.pack('<BIBBQHBIB',
                                            0,  # Version
                                            1,  # LiDAR ID
                                            0,  # LiDAR Type
                                            0,  # Timestamp Type
                                            frame1_pkg2_ts,  # Timestamp
                                            1,  # UDP Counter
                                            2,  # Data Type (Type 2: 8 bytes/pt)
                                            len(frame1_pkg2_data_type2), # Length
                                            0 # Frame Counter
                                            )
    frame1_pkg2_header = frame1_pkg2_header_fields + b'\x00\x00\x00\x00' # Reserved 4 bytes

    frame1_content = frame1_pkg1_header + frame1_pkg1_data_type1 + \
                     frame1_pkg2_header + frame1_pkg2_data_type2

    # Frame 1 Header
    frame1_header_len = 24
    frame1_data_len = len(frame1_content)
    next_frame_offset_for_f1 = first_frame_offset + frame1_header_len + frame1_data_len # Points to start of Frame 2 (or EOF if no Frame 2)

    # --- Frame 2 (Last Frame) ---
    frame2_index = 1
    frame2_pkg1_ts = 3000000000 # ns
    frame2_pkg1_points_mm = [
        (500, 600, 700, 51)
    ]
    frame2_pkg1_data_type1 = b''
    for p in frame2_pkg1_points_mm:
        frame2_pkg1_data_type1 += struct.pack('<iiiB', p[0], p[1], p[2], p[3]) + b'\x00'

    frame2_pkg1_header_fields = struct.pack('<BIBBQHBIB',
                                            0,  # Version
                                            1,  # LiDAR ID
                                            0,  # LiDAR Type
                                            0,  # Timestamp Type
                                            frame2_pkg1_ts, # Timestamp
                                            0,  # UDP Counter
                                            1,  # Data Type (Type 1)
                                            len(frame2_pkg1_data_type1), # Length
                                            0   # Frame Counter
                                            )
    frame2_pkg1_header = frame2_pkg1_header_fields + b'\x00\x00\x00\x00' # Reserved 4 bytes
    frame2_content = frame2_pkg1_header + frame2_pkg1_data_type1

    # Frame 2 Header
    frame2_header_len = 24
    frame2_data_len = len(frame2_content)
    # For the last frame, next_offset is 0
    frame1_header = struct.pack('<QQQ', first_frame_offset, next_frame_offset_for_f1, frame1_index)
    frame2_header = struct.pack('<QQQ', next_frame_offset_for_f1, 0, frame2_index)


    # --- Assemble Full File ---
    full_content = header_content + frame1_header + frame1_content + frame2_header + frame2_content
    filepath = temp_lvx2_file(full_content)

    # --- Test LVX2Reader ---
    frames_read = []
    with LVX2Reader(filepath) as reader:
        for frame_data in reader.read_frames():
            frames_read.append(frame_data)

    assert len(frames_read) == 2

    # --- Validate Frame 1 ---
    frame1 = frames_read[0]
    assert frame1['frame_index'] == frame1_index

    expected_f1_points = np.array([
        [1.0, 2.0, 3.0], # from pkg1
        [1.5, 2.5, 3.5], # from pkg1
        [1.0, 2.0, 3.0]  # from pkg2 (100cm, 200cm, 300cm)
    ], dtype=np.float32)
    expected_f1_ts = np.array([frame1_pkg1_ts, frame1_pkg1_ts, frame1_pkg2_ts], dtype=np.uint64)
    expected_f1_refl = np.array([101, 102, 201], dtype=np.uint8)

    assert frame1['point_cloud'].shape == (3, 3)
    np.testing.assert_array_almost_equal(frame1['point_cloud'], expected_f1_points)
    np.testing.assert_array_equal(frame1['timestamps'], expected_f1_ts)
    np.testing.assert_array_equal(frame1['reflectivity'], expected_f1_refl)

    # --- Validate Frame 2 ---
    frame2 = frames_read[1]
    assert frame2['frame_index'] == frame2_index

    expected_f2_points = np.array([
        [0.5, 0.6, 0.7] # from pkg1
    ], dtype=np.float32)
    expected_f2_ts = np.array([frame2_pkg1_ts], dtype=np.uint64)
    expected_f2_refl = np.array([51], dtype=np.uint8)

    assert frame2['point_cloud'].shape == (1, 3)
    np.testing.assert_array_almost_equal(frame2['point_cloud'], expected_f2_points)
    np.testing.assert_array_equal(frame2['timestamps'], expected_f2_ts)
    np.testing.assert_array_equal(frame2['reflectivity'], expected_f2_refl)

def test_read_frames_from_indoor_test_file(tmp_path): # tmp_path is not strictly needed but good for consistency
    # Assuming 'indoor_sample.lvx' is in 'tests/data/' relative to the repo root
    # Construct path to the test data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, "data", "indoor_sample.lvx")

    if not os.path.exists(data_file_path):
        pytest.skip(f"Test data file not found: {data_file_path}")

    frames_read_count = 0
    total_points_count = 0

    with LVX2Reader(data_file_path) as reader:
        # Basic check: Ensure headers are read without error
        assert reader.public_header is not None
        assert reader.private_header is not None
        assert reader.private_header.get('device_count', 0) > 0 # Expect at least one device
        assert len(reader.device_info_list) == reader.private_header['device_count']

        for i, frame_data in enumerate(reader.read_frames()):
            frames_read_count += 1

            assert isinstance(frame_data, dict)
            assert 'frame_index' in frame_data
            assert 'point_cloud' in frame_data
            assert 'timestamps' in frame_data
            assert 'reflectivity' in frame_data

            assert isinstance(frame_data['frame_index'], (int, np.integer))
            # assert frame_data['frame_index'] == i # Check if frame indices are sequential as expected

            pc = frame_data['point_cloud']
            ts = frame_data['timestamps']
            refl = frame_data['reflectivity']

            assert isinstance(pc, np.ndarray)
            if pc.size > 0: # Only check shape if points exist
                assert pc.ndim == 2
                assert pc.shape[1] == 3
                assert pc.dtype == np.float32
                # Basic sanity check for coordinate values (e.g. not excessively large, not all zero)
                # This depends on the typical scale of "indoor_sample.lvx"
                # For now, just checking they are finite.
                assert np.all(np.isfinite(pc))
                total_points_count += pc.shape[0]


            assert isinstance(ts, np.ndarray)
            assert ts.dtype == np.uint64

            assert isinstance(refl, np.ndarray)
            assert refl.dtype == np.uint8

            if pc.size > 0:
                 assert len(ts) == pc.shape[0]
                 assert len(refl) == pc.shape[0]
            else: # if no points, timestamps and reflectivity should also be empty
                 assert len(ts) == 0
                 assert len(refl) == 0


    assert frames_read_count > 0, "No frames were read from the LVX file."
    # Depending on the file, we might also assert total_points_count > some_threshold
    # For now, just ensuring it runs through is the primary goal.
    # print(f"Successfully read {frames_read_count} frames with a total of {total_points_count} points from {data_file_path}")
