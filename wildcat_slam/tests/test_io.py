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
    assert "Successfully opened LVX2 file." in output # Check if it tried to parse

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
    assert "Error processing LVX2 file: Invalid LVX2 file signature" in captured.out
