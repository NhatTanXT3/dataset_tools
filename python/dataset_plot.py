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
import os
import rerun as rr
import rerun.blueprint as rrb
from typing import Dict, List, Any, Optional
from dataset_loader import get_sensor_by_type, get_sensor_by_name
from quaternion_utils import q_q2C


def dataset_plot(dataset: Dict[str, List[Dict[str, Any]]], 
                 recording_name: str = "",
                 dataset_path: str = "",
                 spawn_viewer: bool = True,
                 max_time_sec: float = float('inf'),
                 blueprint_path: str = ""):
    """
    Plot dataset using rerun visualization (MATLAB dataset_plot equivalent)
    
    Args:
        dataset: Loaded dataset dictionary
        recording_name: Name for the rerun recording
        dataset_path: Path to the dataset
        spawn_viewer: Whether to spawn the rerun viewer automatically
        max_time_sec: Maximum time duration to visualize (for performance)
        blueprint_path: Path to custom rerun blueprint file (.rbl), empty string uses default
    """
    # Setup rerun blueprint layout
    if blueprint_path and os.path.exists(blueprint_path):
        print(f' >> Warning: Blueprint file loading from {blueprint_path} is not supported in this version')
        print(' >> Using default blueprint instead')
        blueprint = create_dataset_blueprint()
    else:
        if blueprint_path:
            print(f' >> Warning: Blueprint file not found: {blueprint_path}')
            print(' >> Using default blueprint instead')
        blueprint = create_dataset_blueprint()
    
    print(f' recording name: {recording_name}, dataset path: {dataset_path}, blueprint path: {blueprint_path}')
        
    if recording_name == "":
        recording_name = dataset_path.split('/')[-1]
    
    # Initialize rerun
    rr.init(recording_name, spawn=spawn_viewer, default_blueprint=blueprint)
    
    print(' >> plotting bodies with rerun')
    
    n_bodies = len(dataset['body'])
    for i, body in enumerate(dataset['body']):
        body_name = body['name']
        print(f'   plotting body [{body_name}]')
        
        # Create namespace for this body
        reference_prefix = f"/ref_{body_name}"

        # Plot sensor configuration (static)
        plot_body_sensor_setup(body, reference_prefix)
        
        # Plot ground truth trajectory (progressive over time)
        plot_position_ground_truth(body, reference_prefix, max_time_sec)

        plot_estimated_ground_truth(body, reference_prefix, max_time_sec)
        
        # Plot IMU time series
        plot_inertial_sensor_measurements(body, reference_prefix, max_time_sec)
        
        # Plot camera images
        plot_camera_images(body, dataset_path, reference_prefix, max_time_sec)
        
        # # Plot camera target observations
        # plot_target_observations(body, reference_prefix, max_time_sec)
    
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


def plot_body_sensor_setup(body: Dict[str, Any], reference_prefix: str) -> None:
    """
    Plot sensor configuration relative to body frame (MATLAB dataset_plot_body equivalent)
    
    Args:
        body: Body dictionary with sensor configurations
        reference_prefix: Rerun path prefix for this body
    """
    body_name = body['name']
    sensors = body.get('sensor', [])

    
    # Plot sensors
    for sensor in sensors:
        sensor_name = sensor['name']
        sensor_type = sensor['sensor_type']
        sensor_prefix = f"{reference_prefix}/{body_name}/{sensor_name}"
        if sensor_type == 'camera' and 'camera_model' in sensor and sensor['camera_model'] == 'pinhole':
            print(f'     plotting camera [{sensor_name}]')
            intrinsics = sensor['intrinsics']
            resolution = sensor['resolution']
            intrinsics_rerun = rr.datatypes.Mat3x3([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
            print(intrinsics)
            rr.log(
                f"{sensor_prefix}/images",
                rr.Pinhole(
                    image_from_camera=intrinsics_rerun,
                    resolution=resolution,
                ),
                static=True,
            )

        if 'T_BS' in sensor:
            T_BS = sensor['T_BS']
            
            # Extract position and rotation
            position = T_BS[:3, 3]
            rotation_matrix = T_BS[:3, :3]

            rr.log(
                f"{sensor_prefix}",
                rr.Transform3D(
                    translation=position,
                    mat3x3=rotation_matrix,
                ),
                static=True,
            )
            
            # Add sensor frame axes (smaller than body frame)
            rr.log(
                f"{sensor_prefix}/axes",
                rr.Arrows3D(
                    origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    vectors=[[0.015, 0, 0], [0, 0.015, 0], [0, 0, 0.015]],
                    colors=[[255, 128, 128], [128, 255, 128], [128, 128, 255]],  # Lighter RGB
                    labels=[f"{sensor_name}_X", f"{sensor_name}_Y", f"{sensor_name}_Z"],
                    show_labels=False
                ),
                static=True,
            )
            
            # Draw line from body origin to sensor
            # rr.log(
            #     f"{sensor_prefix}/connection",
            #     rr.LineStrips3D(
            #         strips=[[[0, 0, 0], position.tolist()]],
            #         colors=[[128, 128, 128]],
            #         radii=[0.001],
            #     ),
            #     static=True,
            # )
            
            print(f'     plotting sensor [{sensor_name}]')
        elif 'T_WR' in sensor:
            print(f'     detected data with world-reference extrinsics [{sensor_name}]')
        else:
            print(f'     detected data without body-sensor extrinsics [{sensor_name}]')

def plot_position_ground_truth(body: Dict[str, Any], reference_prefix: str, max_time_sec: float = float('inf')) -> None:
    """
    Plot ground truth trajectory with poses (MATLAB dataset_plot_ground_truth_trajectory equivalent)
    
    Args:
        body: Body dictionary with sensor data
        reference_prefix: Rerun path prefix for this body
        max_time_sec: Maximum time duration to plot
    """
    body_name = body['name']
    sensors = body.get('sensor', [])

    # Find trajectory data from position sensor
    position_data = None
    position_sensor = None
    
    for sensor in sensors:
        sensor_type = sensor['sensor_type']
        if sensor_type == 'position':
            data = sensor.get('data', {})
            if 't' in data and 'p_RS_R' in data:
                position_data = data
                position_sensor = sensor
                break
    
    if position_data is None:
        print(f'     no ground truth trajectory data found for body [{body_name}]')
        return
    
    timestamps = position_data['t']
    positions = position_data['p_RS_R']
    
    # Filter by time if specified
    if max_time_sec < float('inf'):
        time_mask = (timestamps - timestamps[0]) / 1e9 <= max_time_sec
        timestamps = timestamps[time_mask]
        positions = positions[:, time_mask]
    
    print(f'     plotting ground truth trajectory with {len(timestamps)} poses over time')
    
    # Convert timestamps to rerun time format
    timestamps_ns = timestamps.astype('datetime64[ns]')
    
    # Log trajectory positions progressively over time using send_columns pattern
    # This creates a time-indexed trajectory that builds up as time progresses
    times = rr.TimeColumn("timestamp", timestamp=timestamps_ns)
    
    # Log positions as time-indexed Points3D (progressive trajectory)
    rr.send_columns(
        f"{reference_prefix}/position_ground_truth/points",
        indexes=[times],
        columns=rr.Points3D.columns(
            positions=positions.T.tolist(),  # Convert (3, N) to (N, 3)
        ),
    )

    rr.log(
        f"{reference_prefix}/position_ground_truth",
        rr.LineStrips3D(
            strips=[positions.T.tolist()],  # Convert (3, N) to (N, 3)
            colors=[[0, 255, 255]],
            radii=[0.002],
        ),
        static=True,
    )

def plot_estimated_ground_truth(body: Dict[str, Any], reference_prefix: str, max_time_sec: float = float('inf')) -> None:
    """
    Plot estimated ground truth trajectory with poses (MATLAB dataset_plot_estimated_ground_truth equivalent)
    
    Args:
        body: Body dictionary with sensor data
        reference_prefix: Rerun path prefix for this body
        max_time_sec: Maximum time duration to plot
    """
    body_name = body['name']
    sensors = body.get('sensor', [])

    # Find trajectory data from position sensor
    trajectory_data = None
    trajectory_sensor = None

    for sensor in sensors:
        sensor_type = sensor['sensor_type']
        if sensor_type == 'visual-inertial':
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
    
    print(f'     plotting trajectory with {len(timestamps)} poses, from sensor [{trajectory_sensor["name"]}]')
    # Convert timestamps to rerun time format
    timestamps_ns = timestamps.astype('datetime64[ns]')
    times = rr.TimeColumn("timestamp", timestamp=timestamps_ns)

    quaternions_rerun = []
    positions_rerun = []
    for q in quaternions.T:
        quaternions_rerun.append([q[1], q[2], q[3], q[0]])  # Convert to [qx, qy, qz, qw]
    positions_rerun = positions.T.tolist()

    rr.send_columns(
        f"{reference_prefix}/{body_name}",
        indexes=[times],
        columns=rr.Transform3D.columns(
            translation=positions_rerun,
            quaternion=quaternions_rerun,
        ),
    )

    # Log trajectory as 3D line
    rr.log(
        f"{reference_prefix}/estimated_ground_truth/path",
        rr.LineStrips3D(
            strips=[positions_rerun],  # Convert (3, N) to (N, 3)
            colors=[[0, 255, 0, 100]],  # Cyan
            radii=[0.002],
        ),
        static=True,
    )

    
def plot_body_trajectory(body: Dict[str, Any], reference_prefix: str, max_time_sec: float = float('inf')) -> None:
    """
    Plot 3D trajectory with poses (MATLAB dataset_plot_body_trajectory equivalent)
    
    Args:
        body: Body dictionary with sensor data
        reference_prefix: Rerun path prefix for this body  
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
        f"{reference_prefix}/trajectory/path",
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
            f"{reference_prefix}/trajectory/poses",
            indexes=[times_subsampled],
            columns=rr.Transform3D.columns(
                translation=pose_positions,
                quaternion=pose_quaternions,
            ),
        )
    
    # Add coordinate frame visualization for poses (static, smaller)
    rr.log(
        f"{reference_prefix}/trajectory/poses/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["X", "Y", "Z"],
        ),
        static=True,
    )


def plot_inertial_sensor_measurements(body: Dict[str, Any], reference_prefix: str, max_time_sec: float = float('inf')) -> None:
    """
    Plot IMU time series data (MATLAB dataset_plot_inertial_sensor_measurements equivalent)
    
    Args:
        body: Body dictionary with sensor data
        reference_prefix: Rerun path prefix for this body
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
        sensor_prefix = f"{reference_prefix}/{body_name}/{sensor_name}"
        
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
        gyro_path = f"{sensor_prefix}/gyroscope"
        rr.send_columns(
            gyro_path,
            indexes=[times],
            columns=rr.Scalars.columns(scalars=omega.T),  # Convert (3, N) to (N, 3)
        )
        
        # Log accelerometer data  
        accel_path = f"{sensor_prefix}/accelerometer"
        rr.send_columns(
            accel_path,
            indexes=[times],
            columns=rr.Scalars.columns(scalars=accel.T),  # Convert (3, N) to (N, 3)
        )


def plot_target_observations(body: Dict[str, Any], reference_prefix: str, max_time_sec: float = float('inf')) -> None:
    """
    Plot camera target observations (MATLAB dataset_plot_target_observations equivalent)
    
    Args:
        body: Body dictionary with sensor data
        reference_prefix: Rerun path prefix for this body
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
                f"{reference_prefix}/calibration/{sensor_name}/target_points",
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
                    f"{reference_prefix}/calibration/{sensor_name}/camera_path",
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
                        f"{reference_prefix}/calibration/{sensor_name}/poses/pose_{i}",
                        rr.Transform3D(
                            translation=pos,
                            quaternion=quat_rerun,
                        ),
                        static=True,
                    )
                
                # Add camera frame visualization
                rr.log(
                    f"{reference_prefix}/calibration/{sensor_name}/poses/axes",
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
                        f"{reference_prefix}/calibration/{sensor_name}/observation_rays",
                        rr.LineStrips3D(
                            strips=rays,
                            colors=[[0, 255, 0]],  # Green rays
                            radii=[0.001],
                        ),
                        static=True,
                    )
            
            print(f'     plotting calibration data [{sensor_name}]')


def plot_camera_images(body: Dict[str, Any], dataset_path: str, reference_prefix: str, max_time_sec: float = float('inf')) -> None:
    """
    Plot camera images over time
    
    Args:
        body: Body dictionary with sensor data
        reference_prefix: Rerun path prefix for this body
        max_time_sec: Maximum time duration to plot
    """
    import os
    body_name = body['name']
    sensors = body.get('sensor', [])
    
    # Find camera sensors
    camera_sensors = []
    for sensor in sensors:
        if sensor.get('sensor_type') == 'camera':
            data = sensor.get('data', {})
            if 't' in data and 'filenames' in data:
                camera_sensors.append(sensor)
    
    if not camera_sensors:
        print(f'     no camera data found for body [{body_name}]')
        return
    
    for sensor in camera_sensors:
        sensor_name = sensor['name']
        data = sensor.get('data', {})
        
        timestamps = data['t']
        filenames = data['filenames']
        
        # Filter by time if specified
        if max_time_sec < float('inf'):
            time_mask = (timestamps - timestamps[0]) / 1e9 <= max_time_sec
            timestamps = timestamps[time_mask]
            filenames = [filenames[i] for i in range(len(filenames)) if time_mask[i]]
        
        print(f'     plotting camera images [{sensor_name}] with {len(timestamps)} images')
        
        # Convert timestamps to rerun time format
        timestamps_ns = timestamps.astype('datetime64[ns]')
        
        # Log images individually with timestamps (more efficient for large binary data)
        sensor_prefix = f"{reference_prefix}/{body_name}/{sensor_name}"
        
        for i, (timestamp, filename) in enumerate(zip(timestamps_ns, filenames)):
            # Set current timestamp
            rr.set_time("timestamp", timestamp=timestamp)
            
            # Log image if file exists
            dataset_path_filename = os.path.join(dataset_path, body_name, sensor_name,'data', filename)
            if os.path.exists(dataset_path_filename):
                try:
                    # Use EncodedImage for better performance with large image files
                    rr.log(
                        f"{sensor_prefix}/images",
                        rr.EncodedImage(path=dataset_path_filename)
                    )
                except Exception as e:
                    print(f'     Warning: Failed to load image {dataset_path_filename}: {e}')
            else:
                print(f'     Warning: Image file not found: {dataset_path_filename}')
            
            # Optional: subsample for performance (log every Nth image)
            # Uncomment the lines below to log every 10th image for better performance
            # if i % 10 != 0:
            #     continue


# Convenience function for quick testing
def plot_euroc_dataset(dataset_path: str, max_time_sec: float = 30.0, blueprint_path: str = "") -> None:
    """
    Quick plotting function for EuRoC datasets
    
    Args:
        dataset_path: Path to EuRoC dataset
        max_time_sec: Maximum time to visualize
        blueprint_path: Optional path to custom blueprint file
    """
    from dataset_loader import dataset_load
    
    print(f"Loading and plotting EuRoC dataset: {dataset_path}")
    dataset = dataset_load(dataset_path)
    dataset_plot(dataset, 
                recording_name=f"EuRoC_{dataset_path.split('/')[-1]}", 
                dataset_path=dataset_path,
                max_time_sec=max_time_sec,
                blueprint_path=blueprint_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        max_time = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        blueprint_path = sys.argv[3] if len(sys.argv) > 3 else ""
        plot_euroc_dataset(dataset_path, max_time, blueprint_path)
    else:
        # Default test with EuRoC data
        plot_euroc_dataset("../../EuRoc_ASL/MH_01_easy", 30.0) 