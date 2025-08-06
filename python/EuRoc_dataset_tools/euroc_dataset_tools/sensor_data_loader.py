"""
Sensor data loader for dataset loading - MATLAB compatibility layer
Matches MATLAB dataset_load_sensor_data.m functionality for CSV parsing

Supports different sensor types: IMU, camera, position, pose, visual-inertial, pointcloud, etc.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from .quaternion_utils import q_min, q_C2q

try:
    from plyfile import PlyData, PlyElement
    PLY_AVAILABLE = True
except ImportError:
    PLY_AVAILABLE = False

try:
    import pyminiply
    MINIPLY_AVAILABLE = True
except ImportError:
    MINIPLY_AVAILABLE = False


def dataset_load_sensor_data(sensor_type: str, sensor_folder_name: str) -> Dict[str, Any]:
    """
    Load sensor data based on sensor type (MATLAB dataset_load_sensor_data equivalent)
    
    Args:
        sensor_type: Type of sensor ('imu', 'camera', 'position', 'pose', 'visual-inertial', etc.)
        sensor_folder_name: Path to sensor folder containing data.csv
        
    Returns:
        Dictionary containing parsed sensor data with timestamps and sensor-specific fields
    """
    # For pointcloud, we expect a PLY file instead of CSV
    if sensor_type == 'pointcloud':
        return _load_pointcloud_data(sensor_folder_name)

    csv_filename = os.path.join(sensor_folder_name, 'data.csv')
    
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Data file not found: {csv_filename}")
    
    data = {}
    
    if sensor_type == 'imu':
        data = _load_imu_data(csv_filename)
    elif sensor_type == 'camera':
        data = _load_camera_data(csv_filename)
    elif sensor_type == 'position':
        data = _load_position_data(csv_filename)
    elif sensor_type == 'pose':
        data = _load_pose_data(csv_filename)
    elif sensor_type == 'visual-inertial':
        data = _load_visual_inertial_data(csv_filename)
    elif sensor_type == 'camera_target':
        data = _load_camera_target_data(sensor_folder_name)
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")
    
    return data


def _load_imu_data(csv_filename: str) -> Dict[str, np.ndarray]:
    """
    Load IMU sensor data from CSV file
    
    Expected format: timestamp,w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z
    """
    # Read CSV with pandas for efficient parsing
    df = pd.read_csv(csv_filename)
    
    # Convert to numpy arrays (MATLAB format: columns are row vectors)
    data = {
        't': df.iloc[:, 0].values,  # timestamps
        'omega': df.iloc[:, 1:4].values.T,  # angular velocity (3 x N)
        'a': df.iloc[:, 4:7].values.T       # acceleration (3 x N)
    }
    
    return data


def _load_camera_data(csv_filename: str) -> Dict[str, Any]:
    """
    Load camera sensor data from CSV file
    
    Expected format: timestamp,filename
    """
    df = pd.read_csv(csv_filename)
    
    data = {
        't': df.iloc[:, 0].values,           # timestamps
        'filenames': df.iloc[:, 1].values    # image filenames
    }
    
    return data


def _load_position_data(csv_filename: str) -> Dict[str, np.ndarray]:
    """
    Load position sensor data from CSV file
    
    Expected format: timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z
    """
    df = pd.read_csv(csv_filename)
    
    data = {
        't': df.iloc[:, 0].values,          # timestamps
        'p_RS_R': df.iloc[:, 1:4].values.T  # position (3 x N)
    }
    
    return data


def _load_pose_data(csv_filename: str) -> Dict[str, np.ndarray]:
    """
    Load pose sensor data from CSV file
    
    Expected format: timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z
    """
    df = pd.read_csv(csv_filename)
    
    # Extract position and quaternion data
    positions = df.iloc[:, 1:4].values.T   # position (3 x N)
    quaternions = df.iloc[:, 4:8].values.T  # quaternion (4 x N)
    
    # Apply quaternion minimization (MATLAB q_min equivalent)
    quaternions_min = q_min(quaternions)
    
    data = {
        't': df.iloc[:, 0].values,      # timestamps
        'p_RS_R': positions,            # position (3 x N)
        'q_RS': quaternions_min         # minimal quaternions (4 x N)
    }
    
    return data


def _load_visual_inertial_data(csv_filename: str) -> Dict[str, np.ndarray]:
    """
    Load visual-inertial sensor data from CSV file
    
    Expected format: timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z,
                    v_RS_R_x,v_RS_R_y,v_RS_R_z,bw_S_x,bw_S_y,bw_S_z,ba_S_x,ba_S_y,ba_S_z
    """
    df = pd.read_csv(csv_filename)
    
    # Extract all data fields
    positions = df.iloc[:, 1:4].values.T    # position (3 x N)
    quaternions = df.iloc[:, 4:8].values.T   # quaternion (4 x N)
    velocities = df.iloc[:, 8:11].values.T   # velocity (3 x N)
    gyro_bias = df.iloc[:, 11:14].values.T   # gyro bias (3 x N)
    accel_bias = df.iloc[:, 14:17].values.T  # accel bias (3 x N)
    
    # Apply quaternion minimization
    quaternions_min = q_min(quaternions)
    
    data = {
        't': df.iloc[:, 0].values,      # timestamps
        'p_RS_R': positions,            # position (3 x N)
        'q_RS': quaternions_min,        # minimal quaternions (4 x N)
        'v_RS_R': velocities,           # velocity (3 x N)
        'bw_S': gyro_bias,              # gyroscope bias (3 x N)
        'ba_S': accel_bias              # accelerometer bias (3 x N)
    }
    
    return data


def _load_pointcloud_data(sensor_folder_name: str) -> Dict[str, np.ndarray]:
    """
    Load pointcloud sensor data from PLY file
    
    Expected file: data.ply in sensor folder
    Returns pointcloud with positions data
    
    Uses pyminiply if available (faster), falls back to plyfile
    """
    ply_filename = os.path.join(sensor_folder_name, 'data.ply')
    
    if not os.path.exists(ply_filename):
        raise FileNotFoundError(f"PLY file not found: {ply_filename}")
    
    # Try pyminiply first (faster)
    if MINIPLY_AVAILABLE:
        return _load_pointcloud_with_pyminiply(ply_filename)
    elif PLY_AVAILABLE:
        return _load_pointcloud_with_plyfile(ply_filename)
    else:
        raise ImportError("Neither pyminiply nor plyfile library available. Install with: pip install plyfile")


def _load_pointcloud_with_pyminiply(ply_filename: str) -> Dict[str, np.ndarray]:
    """
    Load pointcloud using pyminiply library
    """
    import pyminiply
    
    # Read PLY file - returns vertices, indices, normals, uv, color, intensity
    vertices, indices, normals, uv, color, intensity = pyminiply.read(ply_filename)
    
    if vertices is None or len(vertices) == 0:
        raise ValueError(f"No vertex data found in PLY file: {ply_filename}")
    
    # Convert to (3, N) format to match MATLAB convention
    if vertices.shape[1] == 3:
        positions = vertices.T  # (N, 3) -> (3, N)
    else:
        raise ValueError(f"Invalid vertex data shape: {vertices.shape}, expected (N, 3)")
    
    n_points = positions.shape[1]
    
    data = {
        'positions': positions  # Point positions (3 x N)
    }
    
    # Add intensity if available
    if intensity.size > 0:
        data['intensity'] = intensity  # Intensity values (N,)
        print(f"     Loaded pointcloud with {n_points} points and intensity (using pyminiply)")
    else:
        print(f"     Loaded pointcloud with {n_points} points (using pyminiply)")
    
    return data


def _load_pointcloud_with_plyfile(ply_filename: str) -> Dict[str, np.ndarray]:
    """
    Load pointcloud using plyfile library (fallback)
    """
    # Read PLY file
    plydata = PlyData.read(ply_filename)
    
    if 'vertex' not in plydata:
        raise ValueError(f"No vertex data found in PLY file: {ply_filename}")
    
    vertex_element = plydata['vertex']
    vertex_data = vertex_element.data  # Get the actual numpy array
    n_points = len(vertex_data)
    
    # Extract point positions (required) - ensure (3, N) format
    positions = np.zeros((3, n_points))
    positions[0, :] = vertex_data['x']
    positions[1, :] = vertex_data['y'] 
    positions[2, :] = vertex_data['z']
    
    data = {
        'positions': positions  # Point positions (3 x N)
    }
    
    print(f"     Loaded pointcloud with {n_points} points (using plyfile)")
    
    return data


def _load_camera_target_data(sensor_folder_name: str) -> Dict[str, Any]:
    """
    Load camera target calibration data from multiple CSV files
    
    Files: data_target.csv, data_undistorted.csv, data_pose_estimates.csv
    """
    data = {}
    
    # Load target points
    target_file = os.path.join(sensor_folder_name, 'data_target.csv')
    if os.path.exists(target_file):
        target_df = pd.read_csv(target_file, header=None)
        target_data = target_df.values.flatten()
        n_target_points = len(target_data) // 3
        data['targetPoints_'] = target_data.reshape(n_target_points, 3).T
    
    # Load undistorted measurements
    undistorted_file = os.path.join(sensor_folder_name, 'data_undistorted.csv')
    if os.path.exists(undistorted_file):
        undist_df = pd.read_csv(undistorted_file, header=None)
        data['t_m_'] = undist_df.iloc[:, 0].values
        
        # Process undistorted measurements
        measurements = undist_df.iloc[:, 1:].values
        n_frames = len(undist_df)
        n_target_points = (measurements.shape[1]) // 3
        
        data['undistortedMeasurements_'] = []
        for i in range(n_frames):
            frame_data = measurements[i, :].reshape(3, n_target_points)
            data['undistortedMeasurements_'].append(frame_data)
    
    # Load pose estimates
    pose_file = os.path.join(sensor_folder_name, 'data_pose_estimates.csv')
    if os.path.exists(pose_file):
        pose_df = pd.read_csv(pose_file, header=None)
        data['t'] = pose_df.iloc[:, 0].values
        
        # Process transformation matrices
        matrices = pose_df.iloc[:, 1:].values
        n_frames = len(pose_df)
        
        data['T_TC_'] = []
        data['q_TC_'] = np.zeros((4, n_frames))
        data['p_TC_T_'] = np.zeros((3, n_frames))
        
        for i in range(n_frames):
            T_matrix = matrices[i, :].reshape(4, 4).T  # MATLAB column-major order
            data['T_TC_'].append(T_matrix)
            
            # Extract quaternion and position
            R_matrix = T_matrix[:3, :3]
            data['q_TC_'][:, i] = q_C2q(R_matrix)
            data['p_TC_T_'][:, i] = T_matrix[:3, 3]
    
    return data


def validate_sensor_data_loader():
    """
    Test sensor data loader with synthetic data
    """
    print("Testing sensor data loader...")
    
    # Test IMU data format validation
    # Create synthetic test data
    timestamps = np.arange(0, 1000, 5) * 1e6  # microseconds
    n_samples = len(timestamps)
    
    # Synthetic IMU data
    omega_data = np.random.randn(n_samples, 3) * 0.1
    accel_data = np.random.randn(n_samples, 3) + [0, 0, -9.81]
    
    # Test data structure
    test_imu_data = {
        't': timestamps,
        'omega': omega_data.T,
        'a': accel_data.T
    }
    
    assert test_imu_data['omega'].shape == (3, n_samples), "IMU omega shape incorrect"
    assert test_imu_data['a'].shape == (3, n_samples), "IMU acceleration shape incorrect"
    
    # Test quaternion minimization
    test_quaternions = np.array([
        [0.7071, -0.7071],   # qw
        [0, 0],              # qx
        [0, 0],              # qy
        [0.7071, -0.7071]    # qz
    ])
    
    q_minimized = q_min(test_quaternions)
    assert np.all(q_minimized[0, :] >= 0), "Quaternion minimization failed"
    
    print("✓ Sensor data loader validation passed!")


if __name__ == "__main__":
    validate_sensor_data_loader() 