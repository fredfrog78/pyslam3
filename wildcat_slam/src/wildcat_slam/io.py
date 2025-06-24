import struct
import numpy as np

class LVX2Reader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.public_header = {}
        self.private_header = {}
        self.device_info_list = []

    def __enter__(self):
        self.file = open(self.filepath, 'rb')
        self._read_public_header()
        self._read_private_header()
        self._read_device_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def _read_public_header(self):
        """
        Reads the public header block.
        File Signature (“livox_tech”) char[16] 16 bytes
        Version-A char 1 bytes
        Version-B char 1 bytes
        Version-C char 1 bytes
        Version-D char 1 bytes
        Magic Code unsigned int 4 bytes
        """
        if not self.file:
            raise IOError("File not open")

        self.file.seek(0)
        signature_bytes = self.file.read(16)
        # Check the critical part of the signature directly as bytes
        if signature_bytes[:10] != b"livox_tech":
            # Try to decode for a more informative error message if it's not total garbage
            try:
                decoded_sig_start = signature_bytes[:10].decode('ascii', errors='replace')
            except Exception: # Should not happen with errors='replace' for ascii
                decoded_sig_start = str(signature_bytes[:10]) # Raw bytes if decode fails
            raise ValueError(f"Invalid LVX2 file signature: expected b'livox_tech' prefix, got {decoded_sig_start}")

        # For storage/reporting, decode the full signature safely, handling potential non-ASCII in padding
        signature_str = signature_bytes.decode('ascii', errors='ignore').rstrip('\x00')

        version_a = struct.unpack('<B', self.file.read(1))[0]
        version_b = struct.unpack('<B', self.file.read(1))[0]
        version_c = struct.unpack('<B', self.file.read(1))[0]
        version_d = struct.unpack('<B', self.file.read(1))[0]
        self.public_header['version'] = (version_a, version_b, version_c, version_d)

        magic_code = struct.unpack('<I', self.file.read(4))[0]
        if magic_code != 0xAC0EA767:
            raise ValueError(f"Invalid LVX2 magic code: {hex(magic_code)}")
        self.public_header['magic_code'] = magic_code
        self.public_header['signature'] = signature_str # Store the cleaned string
        return self.public_header

    def _read_private_header(self):
        """
        Reads the private header block.
        Frame Duration unsigned int 4 bytes
        Device Count unsigned char 1 byte
        """
        if not self.file:
            raise IOError("File not open")
        # Public header is 24 bytes
        self.file.seek(24)
        frame_duration = struct.unpack('<I', self.file.read(4))[0]
        device_count = struct.unpack('<B', self.file.read(1))[0]

        self.private_header['frame_duration'] = frame_duration
        self.private_header['device_count'] = device_count
        return self.private_header

    def _read_device_info(self):
        """
        Reads the device info block.
        Device Info 0 struct 63 bytes
        ......
        Device Info N struct 63 bytes
        N = Device Count - 1
        """
        if not self.file:
            raise IOError("File not open")
        # Public header (24 bytes) + Private header (5 bytes)
        self.file.seek(29)
        device_count = self.private_header.get('device_count', 0)
        self.device_info_list = []

        for _ in range(device_count):
            dev_info = {}
            dev_info['lidar_sn_code'] = self.file.read(16).decode('ascii', errors='ignore').rstrip('\x00')
            dev_info['hub_sn_code'] = self.file.read(16).decode('ascii', errors='ignore').rstrip('\x00')
            dev_info['lidar_id'] = struct.unpack('<I', self.file.read(4))[0]
            dev_info['lidar_type'] = struct.unpack('<B', self.file.read(1))[0] # Reserved
            dev_info['device_type'] = struct.unpack('<B', self.file.read(1))[0]
            dev_info['extrinsic_enable'] = struct.unpack('<B', self.file.read(1))[0]
            dev_info['roll'] = struct.unpack('<f', self.file.read(4))[0]
            dev_info['pitch'] = struct.unpack('<f', self.file.read(4))[0]
            dev_info['yaw'] = struct.unpack('<f', self.file.read(4))[0]
            dev_info['x'] = struct.unpack('<f', self.file.read(4))[0]
            dev_info['y'] = struct.unpack('<f', self.file.read(4))[0]
            dev_info['z'] = struct.unpack('<f', self.file.read(4))[0]
            self.device_info_list.append(dev_info)
        return self.device_info_list

    def read_frames(self):
        # This will be implemented in a later iteration
        # For now, it's a placeholder
        pass

class PLYWriter:
    def __init__(self, filepath):
        self.filepath = filepath

    def write(self, points, colors=None):
        """
        Writes points (N x 3 float) and optional colors to a PLY file.
        """
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be an N x 3 numpy array.")
        if colors is not None and (not isinstance(colors, np.ndarray) or colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] != points.shape[0]):
            raise ValueError("Colors must be an N x 3 numpy array, matching the number of points.")
        if colors is not None and not (np.all(colors >= 0) and np.all(colors <= 255) and colors.dtype == np.uint8):
            raise ValueError("Colors must be uint8 and in the range [0, 255].")

        with open(self.filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")

            for i in range(points.shape[0]):
                # Format floats to typically 6 decimal places for PLY
                if colors is not None:
                    f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} {colors[i,0]} {colors[i,1]} {colors[i,2]}\n")
                else:
                    f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}\n")

class TrajWriter:
    def __init__(self, filepath):
        self.filepath = filepath

    def write(self, timestamps, poses):
        """
        Writes timestamps and poses (SE(3) or N x 7 [tx,ty,tz,qx,qy,qz,qw]) to a TXT file.
        Each line: timestamp tx ty tz qx qy qz qw
        """
        if len(timestamps) != len(poses):
            raise ValueError("Timestamps and poses must have the same length.")
        if not isinstance(poses, np.ndarray) or poses.ndim != 2 or poses.shape[1] != 7:
            # Assuming poses are N x 7 (t_x, t_y, t_z, q_x, q_y, q_z, q_w)
             raise ValueError("Poses must be an N x 7 numpy array.")

        with open(self.filepath, 'w') as f:
            for ts, pose in zip(timestamps, poses):
                # Format: timestamp tx ty tz qx qy qz qw
                f.write(f"{ts} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}\n")

if __name__ == '__main__':
    # Example Usage (requires a sample.lvx2 file)
    # This part is for basic manual testing and will be removed or moved to tests later.
    # Create a dummy lvx2 file for testing
    # Public Header (24 bytes)
    # Signature (16) + Version (4) + Magic Code (4)
    signature = b"livox_tech\0\0\0\0\0\0"
    version = bytes([2, 0, 0, 0])
    magic_code = struct.pack('<I', 0xAC0EA767)
    # Private Header (5 bytes)
    # Frame Duration (4) + Device Count (1)
    frame_duration = struct.pack('<I', 50) # 50 ms
    device_count = struct.pack('<B', 1)    # 1 device
    # Device Info (63 bytes per device)
    # SN (16) + Hub SN (16) + ID (4) + LiDAR Type (1) + Device Type (1) + Extrinsic (1) + r,p,y,x,y,z (6*4=24)
    lidar_sn = b"LIDARSN123456789" + b'\0' * (16 - len(b"LIDARSN123456789"))
    hub_sn = b"HUBSN1234567890" + b'\0' * (16 - len(b"HUBSN1234567890"))
    lidar_id = struct.pack('<I', 1)
    lidar_type_reserved = struct.pack('<B', 0)
    device_type = struct.pack('<B', 9) # Mid
    extrinsic_enable = struct.pack('<B', 1)
    roll = struct.pack('<f', 0.1)
    pitch = struct.pack('<f', 0.2)
    yaw = struct.pack('<f', 0.3)
    x = struct.pack('<f', 0.4)
    y = struct.pack('<f', 0.5)
    z = struct.pack('<f', 0.6)

    dummy_lvx2_content = signature + version + magic_code + \
                         frame_duration + device_count + \
                         lidar_sn + hub_sn + lidar_id + lidar_type_reserved + device_type + \
                         extrinsic_enable + roll + pitch + yaw + x + y + z

    # Add some dummy frame data to make file a bit larger and avoid EOF issues if parser tries to read more
    # Frame Header (24 bytes)
    current_offset = struct.pack('<Q', 29 + 63) # Start of this frame
    next_offset = struct.pack('<Q', 29 + 63 + 24 + 100) # Dummy next offset
    frame_index = struct.pack('<Q', 0)
    dummy_frame_header = current_offset + next_offset + frame_index
    # Dummy package data (simplistic)
    dummy_package_data = b'\0' * 100 # Placeholder for package data

    dummy_lvx2_content += dummy_frame_header + dummy_package_data


    with open("dummy_sample.lvx2", "wb") as f:
        f.write(dummy_lvx2_content)

    print("Dummy dummy_sample.lvx2 created.")

    try:
        with LVX2Reader("dummy_sample.lvx2") as reader:
            print("Public Header:", reader.public_header)
            print("Private Header:", reader.private_header)
            print("Device Info:", reader.device_info_list)
    except Exception as e:
        print(f"Error reading dummy LVX2 file: {e}")

    # Test PLYWriter
    points_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    colors_data = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    ply_writer = PLYWriter("dummy_points.ply")
    ply_writer.write(points_data, colors_data)
    print("dummy_points.ply created.")
    ply_writer_no_color = PLYWriter("dummy_points_no_color.ply")
    ply_writer_no_color.write(points_data)
    print("dummy_points_no_color.ply created.")


    # Test TrajWriter
    timestamps_data = np.array([1678886400.123, 1678886400.223, 1678886400.323])
    # tx, ty, tz, qx, qy, qz, qw
    poses_data = np.array([
        [1.1, 1.2, 1.3, 0.1, 0.2, 0.3, 0.9],
        [2.1, 2.2, 2.3, 0.4, 0.5, 0.6, 0.8],
        [3.1, 3.2, 3.3, 0.7, 0.8, 0.9, 0.1]
    ], dtype=np.float64)
    traj_writer = TrajWriter("dummy_traj.txt")
    traj_writer.write(timestamps_data, poses_data)
    print("dummy_traj.txt created.")
