#!/usr/bin/env python3
"""
Dataset plotting using rerun visualization - MATLAB compatibility layer
Equivalent to MATLAB dataset_plot.m functionality with rerun 3D visualization

Provides interactive 3D visualization of:
- Sensor configurations and body frames
- 3D trajectories with pose markers  
- IMU time series (gyroscope/accelerometer)
- Camera target observations
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from typing import Dict, List, Any, Optional
from dataset_loader import get_sensor_by_type, get_sensor_by_name
from quaternion_utils import q_q2C


def dataset_plot(dataset: Dict[str, List[Dict[str, Any]]], 
                 recording_name: str = "EuRoC_Dataset",
                 spawn_viewer: bool = True,
                 max_time_sec: float = float('inf')):
    """
    Plot dataset using rerun visualization (MATLAB dataset_plot equivalent)
    
    Args:
        dataset: Loaded dataset dictionary
        recording_name: Name for the rerun recording
        spawn_viewer: Whether to spawn the rerun viewer automatically
        max_time_sec: Maximum time duration to visualize (for performance)
    """
    # Setup rerun blueprint layout
    blueprint = create_dataset_blueprint()
    
    # Initialize rerun
    rr.init(recording_name, spawn=spawn_viewer, default_blueprint=blueprint)
    
    print(' >> plotting bodies with rerun')
    
    n_bodies = len(dataset['body'])
    for i, body in enumerate(dataset['body']):
        body_name = body['name']
        print(f'   plotting body [{body_name}]')
        
        # Create namespace for this body
        body_prefix = f"/body_{body_name}"
        
        # Plot sensor configuration (static)
        plot_body_sensor_setup(body, body_prefix)
        
        # Plot trajectory with poses
        plot_body_trajectory(body, body_prefix, max_time_sec)
        
        # Plot IMU time series
        plot_inertial_sensor_measurements(body, body_prefix, max_time_sec)
        
        # Plot camera target observations
        plot_target_observations(body, body_prefix, max_time_sec)
    
    print(' >> rerun visualization complete')
    print(f' >> View at: http://localhost:9876 or rerun app')


def create_dataset_blueprint() -> rrb.Blueprint:
    """
    Create rerun blueprint layout for dataset visualization
    
    Returns:
        Configured blueprint with 3D view and time series panels
    """
    return rrb.Horizontal(
        # Left panel: 3D spatial view
        rrb.Vertical(
            rrb.Spatial3DView(
                origin="/",
                name="3D Scene",
                background={"Kind": "GradientDark"},
            ),
            name="3D Visualization",
            row_shares=[1.0],
        ),
        # Right panel: Time series plots
        rrb.Vertical(
            rrb.TimeSeriesView(
                origin="/body_*/sensors/imu*/gyroscope",
                name="Gyroscope [rad/s]",
                overrides={
                    "gyroscope": rr.SeriesLines.from_fields(
                        names=["x", "y", "z"], 
                        colors=[[231, 76, 60], [39, 174, 96], [52, 120, 219]]
                    ),
                },
            ),
            rrb.TimeSeriesView(
                origin="/body_*/sensors/imu*/accelerometer", 
                name="Accelerometer [m/s²]",
                overrides={
                    "accelerometer": rr.SeriesLines.from_fields(
                        names=["x", "y", "z"],
                        colors=[[231, 76, 60], [39, 174, 96], [52, 120, 219]]
                    ),
                },
            ),
            name="Sensor Data",
            row_shares=[0.5, 0.5],
        ),
        column_shares=[0.6, 0.4],
    )


def plot_body_sensor_setup(body: Dict[str, Any], body_prefix: str) -> None:
    """
    Plot sensor configuration relative to body frame (MATLAB dataset_plot_body equivalent)
    
    Args:
        body: Body dictionary with sensor configurations
        body_prefix: Rerun path prefix for this body
    """
    body_name = body['name']
    sensors = body.get('sensor', [])
    
    # Plot body coordinate frame at origin
    rr.log(
        f"{body_prefix}/body_frame",
        rr.Transform3D(
            translation=[0, 0, 0],
            quaternion=[1, 0, 0, 0],  # Identity quaternion [w, x, y, z]
        ),
        static=True,
    )
    
    # Add body frame axes visualization
    rr.log(
        f"{body_prefix}/body_frame/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # RGB = XYZ
            labels=["X", "Y", "Z"],
        ),
        static=True,
    )
    
    # Plot sensors
    for sensor in sensors:
        sensor_name = sensor['name']
        sensor_prefix = f"{body_prefix}/sensors/{sensor_name}"
        
        if 'T_BS' in sensor:
            T_BS = sensor['T_BS']
            
            # Extract position and rotation
            position = T_BS[:3, 3]
            rotation_matrix = T_BS[:3, :3]
            
            # Convert to quaternion (transformations library format: [w, x, y, z])
            from quaternion_utils import q_C2q
            quaternion = q_C2q(rotation_matrix)
            
            # Plot sensor coordinate frame
            rr.log(
                f"{sensor_prefix}/frame",
                rr.Transform3D(
                    translation=position,
                    quaternion=quaternion,
                ),
                static=True,
            )
            
            # Add sensor frame axes (smaller than body frame)
            rr.log(
                f"{sensor_prefix}/frame/axes",
                rr.Arrows3D(
                    origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    vectors=[[0.015, 0, 0], [0, 0.015, 0], [0, 0, 0.015]],
                    colors=[[255, 128, 128], [128, 255, 128], [128, 128, 255]],  # Lighter RGB
                    labels=[f"{sensor_name}_X", f"{sensor_name}_Y", f"{sensor_name}_Z"],
                ),
                static=True,
            )
            
            # Draw line from body origin to sensor
            rr.log(
                f"{sensor_prefix}/connection",
                rr.LineStrips3D(
                    strips=[[[0, 0, 0], position.tolist()]],
                    colors=[[128, 128, 128]],
                    radii=[0.001],
                ),
                static=True,
            )
            
            print(f'     plotting sensor [{sensor_name}]')
        else:
            print(f'     detected data without body-sensor extrinsics [{sensor_name}]')


def plot_body_trajectory(body: Dict[str, Any], body_prefix: str, max_time_sec: float = float('inf')) -> None:
    """
    Plot 3D trajectory with poses (MATLAB dataset_plot_body_trajectory equivalent)
    
    Args:
        body: Body dictionary with sensor data
        body_prefix: Rerun path prefix for this body  
        max_time_sec: Maximum time duration to plot
    """
    body_name = body['name']
    sensors = body.get('sensor', [])
    
    # Find trajectory data from visual-inertial sensor
    trajectory_data = None
    trajectory_sensor = None
    
    for sensor in sensors:
        if sensor.get('sensor_type') == 'visual-inertial':
            data = sensor.get('data', {})
            if 't' in data and 'p_RS_R' in data and 'q_RS' in data:
                trajectory_data = data
                trajectory_sensor = sensor
                break
    
    if trajectory_data is None:
        print(f'     no trajectory data found for body [{body_name}]')
        return
    
    timestamps = trajectory_data['t']
    positions = trajectory_data['p_RS_R']  # (3, N)
    quaternions = trajectory_data['q_RS']  # (4, N) [qw, qx, qy, qz]
    
    # Filter by time if specified
    if max_time_sec < float('inf'):
        time_mask = (timestamps - timestamps[0]) / 1e9 <= max_time_sec
        timestamps = timestamps[time_mask]
        positions = positions[:, time_mask]
        quaternions = quaternions[:, time_mask]
    
    n_poses = len(timestamps)
    if n_poses == 0:
        return
    
    print(f'     plotting trajectory with {n_poses} poses')
    
    # Convert timestamps to rerun time format
    timestamps_ns = timestamps.astype('datetime64[ns]')
    times = rr.TimeColumn("timestamp", timestamp=timestamps_ns)
    
    # Log trajectory as 3D line
    rr.log(
        f"{body_prefix}/trajectory/path",
        rr.LineStrips3D(
            strips=[positions.T.tolist()],  # Convert (3, N) to (N, 3)
            colors=[[0, 255, 255]],  # Cyan
            radii=[0.002],
        ),
        static=True,
    )
    
    # Log poses along trajectory (subsampled for performance)
    subsample_factor = max(1, n_poses // 200)  # Aim for ~200 poses max
    
    pose_positions = []
    pose_quaternions = []
    pose_times = []
    
    for i in range(0, n_poses, subsample_factor):
        pose_positions.append(positions[:, i].tolist())
        # Rerun expects [x, y, z, w] quaternion format
        q = quaternions[:, i]  # [qw, qx, qy, qz]
        pose_quaternions.append([q[1], q[2], q[3], q[0]])  # Convert to [qx, qy, qz, qw]
        pose_times.append(timestamps_ns[i])
    
    # Send poses as time-indexed transforms
    if pose_positions:
        times_subsampled = rr.TimeColumn("timestamp", timestamp=pose_times)
        rr.send_columns(
            f"{body_prefix}/trajectory/poses",
            indexes=[times_subsampled],
            columns=rr.Transform3D.columns(
                translation=pose_positions,
                quaternion=pose_quaternions,
            ),
        )
    
    # Add coordinate frame visualization for poses (static, smaller)
    rr.log(
        f"{body_prefix}/trajectory/poses/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["X", "Y", "Z"],
        ),
        static=True,
    )


def plot_inertial_sensor_measurements(body: Dict[str, Any], body_prefix: str, max_time_sec: float = float('inf')) -> None:
    """
    Plot IMU time series data (MATLAB dataset_plot_inertial_sensor_measurements equivalent)
    
    Args:
        body: Body dictionary with sensor data
        body_prefix: Rerun path prefix for this body
        max_time_sec: Maximum time duration to plot
    """
    body_name = body['name']
    imu_sensors = []
    
    # Find all IMU sensors
    for sensor in body.get('sensor', []):
        if sensor.get('sensor_type') == 'imu':
            imu_sensors.append(sensor)
    
    if not imu_sensors:
        return
    
    for imu_sensor in imu_sensors:
        sensor_name = imu_sensor['name']
        data = imu_sensor.get('data', {})
        
        if 't' not in data or 'omega' not in data or 'a' not in data:
            continue
        
        timestamps = data['t']
        omega = data['omega']  # (3, N) rad/s
        accel = data['a']     # (3, N) m/s²
        
        # Filter by time if specified
        if max_time_sec < float('inf'):
            time_mask = (timestamps - timestamps[0]) / 1e9 <= max_time_sec
            timestamps = timestamps[time_mask]
            omega = omega[:, time_mask]
            accel = accel[:, time_mask]
        
        if len(timestamps) == 0:
            continue
        
        print(f'     plotting IMU data [{sensor_name}] with {len(timestamps)} samples')
        
        # Convert timestamps to rerun format
        timestamps_ns = timestamps.astype('datetime64[ns]')
        times = rr.TimeColumn("timestamp", timestamp=timestamps_ns)
        
        # Log gyroscope data
        gyro_path = f"{body_prefix}/sensors/{sensor_name}/gyroscope"
        rr.send_columns(
            gyro_path,
            indexes=[times],
            columns=rr.Scalars.columns(scalars=omega.T),  # Convert (3, N) to (N, 3)
        )
        
        # Log accelerometer data  
        accel_path = f"{body_prefix}/sensors/{sensor_name}/accelerometer"
        rr.send_columns(
            accel_path,
            indexes=[times],
            columns=rr.Scalars.columns(scalars=accel.T),  # Convert (3, N) to (N, 3)
        )


def plot_target_observations(body: Dict[str, Any], body_prefix: str, max_time_sec: float = float('inf')) -> None:
    """
    Plot camera target observations (MATLAB dataset_plot_target_observations equivalent)
    
    Args:
        body: Body dictionary with sensor data
        body_prefix: Rerun path prefix for this body
        max_time_sec: Maximum time duration to plot
    """
    body_name = body['name']
    
    # Find camera_target sensors
    for sensor in body.get('sensor', []):
        if sensor.get('sensor_type') == 'camera_target':
            sensor_name = sensor['name']
            data = sensor.get('data', {})
            
            if 'targetPoints_' not in data:
                continue
            
            target_points = data['targetPoints_']  # (3, N_points)
            
            # Plot target points
            rr.log(
                f"{body_prefix}/calibration/{sensor_name}/target_points",
                rr.Points3D(
                    positions=target_points.T,  # Convert (3, N) to (N, 3)
                    colors=[[0, 0, 0]],  # Black points
                    radii=[0.01],
                ),
                static=True,
            )
            
            # Plot camera poses if available
            if 'T_TC_' in data and 'q_TC_' in data and 'p_TC_T_' in data:
                positions = data['p_TC_T_']  # (3, N)
                quaternions = data['q_TC_']  # (4, N)
                
                n_poses = positions.shape[1]
                subsample_factor = max(1, n_poses // 50)  # Subsample for performance
                
                # Plot camera trajectory
                rr.log(
                    f"{body_prefix}/calibration/{sensor_name}/camera_path",
                    rr.LineStrips3D(
                        strips=[positions.T.tolist()],
                        colors=[[0, 255, 0]],  # Green
                        radii=[0.001],
                    ),
                    static=True,
                )
                
                # Plot subsampled camera poses
                for i in range(0, n_poses, subsample_factor):
                    pos = positions[:, i]
                    q = quaternions[:, i]  # [qw, qx, qy, qz]
                    
                    # Convert quaternion format for rerun
                    quat_rerun = [q[1], q[2], q[3], q[0]]  # [qx, qy, qz, qw]
                    
                    rr.log(
                        f"{body_prefix}/calibration/{sensor_name}/poses/pose_{i}",
                        rr.Transform3D(
                            translation=pos,
                            quaternion=quat_rerun,
                        ),
                        static=True,
                    )
                
                # Add camera frame visualization
                rr.log(
                    f"{body_prefix}/calibration/{sensor_name}/poses/axes",
                    rr.Arrows3D(
                        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        vectors=[[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                        labels=["X", "Y", "Z"],
                    ),
                    static=True,
                )
                
            # Plot some observation rays if available
            if ('undistortedMeasurements_' in data and 
                len(data['undistortedMeasurements_']) > 0 and
                'p_TC_T_' in data and 'q_TC_' in data):
                
                # Plot observation rays for first camera pose
                pos = data['p_TC_T_'][:, 0]
                q = data['q_TC_'][:, 0]
                measurements = data['undistortedMeasurements_'][0]  # (3, N_corners)
                
                observation_scale = 2.0
                R_TC = q_q2C(q)  # Convert quaternion to rotation matrix
                
                rays = []
                for i in range(measurements.shape[1]):
                    ray_end = pos + R_TC @ (observation_scale * measurements[:, i])
                    rays.append([pos.tolist(), ray_end.tolist()])
                
                if rays:
                    rr.log(
                        f"{body_prefix}/calibration/{sensor_name}/observation_rays",
                        rr.LineStrips3D(
                            strips=rays,
                            colors=[[0, 255, 0]],  # Green rays
                            radii=[0.001],
                        ),
                        static=True,
                    )
            
            print(f'     plotting calibration data [{sensor_name}]')


# Convenience function for quick testing
def plot_euroc_dataset(dataset_path: str, max_time_sec: float = 30.0) -> None:
    """
    Quick plotting function for EuRoC datasets
    
    Args:
        dataset_path: Path to EuRoC dataset
        max_time_sec: Maximum time to visualize
    """
    from dataset_loader import dataset_load
    
    print(f"Loading and plotting EuRoC dataset: {dataset_path}")
    dataset = dataset_load(dataset_path)
    dataset_plot(dataset, recording_name=f"EuRoC_{dataset_path.split('/')[-1]}", max_time_sec=max_time_sec)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        max_time = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        plot_euroc_dataset(dataset_path, max_time)
    else:
        # Default test with EuRoC data
        plot_euroc_dataset("../../EuRoc_ASL/MH_01_easy", 30.0) 