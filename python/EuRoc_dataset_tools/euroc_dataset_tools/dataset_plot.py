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
from .dataset_loader import get_sensor_by_type, get_sensor_by_name
from .quaternion_utils import q_q2C


def dataset_plot(dataset: Dict[str, List[Dict[str, Any]]], 
                 recording_name: str = "",
                 dataset_path: str = "",
                 spawn_viewer: bool = True,
                 max_time_sec: float = float('inf'),
                 blueprint_path: str = "",
                 undistort_images: bool = False):
    """
    Plot dataset using rerun visualization (MATLAB dataset_plot equivalent)
    
    Args:
        dataset: Loaded dataset dictionary
        recording_name: Name for the rerun recording
        dataset_path: Path to the dataset
        spawn_viewer: Whether to spawn the rerun viewer automatically
        max_time_sec: Maximum time duration to visualize (for performance)
        blueprint_path: Path to custom rerun blueprint file (.rbl), empty string uses default
        undistort_images: Whether to apply camera undistortion to images before visualization
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
        
        # Check if we have pointcloud sensors with T_WR (world reference frame)
        has_pointcloud_with_world_frame = False
        for sensor in body.get('sensor', []):
            if sensor.get('sensor_type') == 'pointcloud' and 'T_WR' in sensor:
                has_pointcloud_with_world_frame = True
                break
        
        # Create namespace for this body (use world frame if pointcloud present)
        if has_pointcloud_with_world_frame:
            reference_prefix = f"/world/ref_{body_name}"
        else:
            reference_prefix = f"/ref_{body_name}"

        # Plot sensor configuration (static)
        plot_body_sensor_setup(body, reference_prefix)
        
        # Plot pointcloud data (static)
        plot_pointcloud(body, reference_prefix)
        
        # Plot ground truth trajectory (progressive over time)
        plot_position_ground_truth(body, reference_prefix, max_time_sec)

        plot_estimated_ground_truth(body, reference_prefix, max_time_sec)
        
        # Plot IMU time series
        plot_inertial_sensor_measurements(body, reference_prefix, max_time_sec)
        
        # Plot camera images
        plot_camera_images(body, dataset_path, reference_prefix, max_time_sec, undistort_images)

    
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

    # Log world-to-reference transformation if we have pointcloud with T_WR
    for sensor in sensors:
        if sensor.get('sensor_type') == 'pointcloud' and 'T_WR' in sensor:
            T_WR = sensor['T_WR']
            
            # Extract position and rotation from T_WR
            position = T_WR[:3, 3]
            rotation_matrix = T_WR[:3, :3]
            
            # Log the world-to-reference transformation at the reference prefix
            rr.log(
                reference_prefix,
                rr.Transform3D(
                    translation=position,
                    mat3x3=rotation_matrix,
                ),
                static=True,
            )
            
            # Add world reference frame axes
            rr.log(
                f"{reference_prefix}/axes",
                rr.Arrows3D(
                    origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    vectors=[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Red, Green, Blue
                    labels=["World_X", "World_Y", "World_Z"],
                    show_labels=False
                ),
                static=True,
            )
            
            print(f'     plotting world-reference frame transformation', T_WR)
            break  # Only need to do this once per body
    
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


def plot_pointcloud(body: Dict[str, Any], reference_prefix: str) -> None:
    """
    Plot pointcloud data (static 3D points) with intensity-based coloring
    
    Args:
        body: Body dictionary with sensor data
        reference_prefix: Rerun path prefix for this body
    
    Features:
        - Uses intensity data for monochrome coloring (dark=low intensity, bright=high intensity)
        - Automatically subsamples large pointclouds (>500K points) for performance
        - Falls back to uniform gray if no intensity data available
    """
    body_name = body['name']
    sensors = body.get('sensor', [])
    
    # Find pointcloud sensors
    for sensor in sensors:
        if sensor.get('sensor_type') == 'pointcloud':
            sensor_name = sensor['name']
            data = sensor.get('data', {})
            
            if 'positions' not in data:
                print(f'     no pointcloud data available for [{sensor_name}]')
                continue
            
            positions = data['positions']  # Shape: (3, N)
            n_points = positions.shape[1]
            
            # Convert to (N, 3) format for rerun
            points_rerun = positions.T  # (3, N) -> (N, 3)
            
            # Extract intensity data if available
            intensity = data.get('intensity', None)
            
            # Use subsampling for very large pointclouds to improve performance
            max_points_display = 50000000  # Limit for visualization performance
            subsample_indices = None
            if n_points > max_points_display:
                # Subsample points uniformly
                step = n_points // max_points_display
                subsample_indices = np.arange(0, n_points, step)
                points_rerun = points_rerun[subsample_indices]
                n_display = len(points_rerun)
                print(f'     plotting pointcloud [{sensor_name}]: {n_display}/{n_points} points (subsampled for performance)')
            else:
                print(f'     plotting pointcloud [{sensor_name}]: {n_points} points')
            
            # Prepare colors based on intensity
            if intensity is not None and len(intensity) > 0:
                # Use intensity for monochrome coloring
                intensity_values = intensity
                if subsample_indices is not None:
                    intensity_values = intensity[subsample_indices]
                
                # Normalize intensity to 0-255 range for grayscale
                intensity_min = np.min(intensity_values)
                intensity_max = np.max(intensity_values)
                if intensity_max > intensity_min:
                    # Normalize to 0-1, then scale to 0-255
                    normalized_intensity = (intensity_values - intensity_min) / (intensity_max - intensity_min)
                    grayscale_values = (normalized_intensity * 255).astype(np.uint8)
                else:
                    # Handle case where all intensities are the same
                    grayscale_values = np.full(len(intensity_values), 128, dtype=np.uint8)
                
                # Create RGB colors from grayscale (R=G=B for monochrome)
                colors = np.column_stack([grayscale_values, grayscale_values, grayscale_values])
                
                print(f'       with intensity coloring: range [{intensity_min:.4f}, {intensity_max:.4f}]')
            else:
                # Fallback to uniform gray color
                colors = [128, 128, 128]
                print(f'       using uniform gray color (no intensity data)')
            
            # Create pointcloud path
            pointcloud_prefix = f"{reference_prefix}/pointclouds/{sensor_name}"
            
            # Log pointcloud points
            rr.log(
                f"{pointcloud_prefix}/points",
                rr.Points3D(
                    positions=points_rerun,
                    colors=colors,
                    radii=0.002,  # Small point size
                ),
                static=True,
            )
            
            # Log pointcloud metadata
            if intensity is not None and len(intensity) > 0:
                metadata_text = f"Pointcloud: {n_points} points total\nIntensity range: [{np.min(intensity):.4f}, {np.max(intensity):.4f}]"
            else:
                metadata_text = f"Pointcloud: {n_points} points total\nNo intensity data"
            
            rr.log(
                f"{pointcloud_prefix}/info",
                rr.TextLog(metadata_text),
                static=True,
            )


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
        if sensor_type == 'position' or sensor_type == 'pose':
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

def plot_camera_images(body: Dict[str, Any], dataset_path: str, reference_prefix: str, max_time_sec: float = float('inf'), undistort_images: bool = False) -> None:
    """
    Plot camera images over time
    
    Args:
        body: Body dictionary with sensor data
        dataset_path: Path to the dataset
        reference_prefix: Rerun path prefix for this body
        max_time_sec: Maximum time duration to plot
        undistort_images: Whether to apply camera undistortion before visualization
    """
    import os
    body_name = body['name']
    sensors = body.get('sensor', [])
    
    # Import image processing for undistortion if needed
    undistorters = {}
    if undistort_images:
        try:
            from .image_processing import create_undistorter_for_sensor
            import cv2
        except ImportError as e:
            print(f'     Warning: Cannot import image processing modules for undistortion: {e}')
            print('     Falling back to original images without undistortion')
            undistort_images = False
    
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
    
    # Create undistorters for camera sensors if undistortion is enabled
    if undistort_images:
        print(f'     setting up image undistortion for {len(camera_sensors)} camera sensors')
        for sensor in camera_sensors:
            sensor_name = sensor['name']
            sensor_prefix = f"{reference_prefix}/{body_name}/{sensor_name}"
            undistorter = create_undistorter_for_sensor(sensor)
            if undistorter is not None:
                undistorters[sensor_name] = undistorter
                print(f'     undistorter created for sensor [{sensor_name}]')
                
                # Update camera intrinsics for Rerun pinhole with undistorted values
                # undistorted_intrinsics = undistorter.get_undistorted_intrinsics()
                # intrinsics_rerun = rr.datatypes.Mat3x3([
                #     [undistorted_intrinsics['intrinsics'][0], 0, undistorted_intrinsics['intrinsics'][2]], 
                #     [0, undistorted_intrinsics['intrinsics'][1], undistorted_intrinsics['intrinsics'][3]], 
                #     [0, 0, 1]
                # ])
                # rr.log(
                #     f"{sensor_prefix}/images",
                #     rr.Pinhole(
                #         image_from_camera=intrinsics_rerun,
                #         resolution=undistorted_intrinsics['resolution'],
                #     ),
                #     static=True,
                # )
            else:
                print(f'     failed to create undistorter for sensor [{sensor_name}], using original images')
                undistorters[sensor_name] = None
    
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
        rectified_prefix = f"{sensor_prefix}/rectified_images"
        
        # If undistortion is enabled and undistorter exists, log rectified pinhole model
        if undistort_images and sensor_name in undistorters and undistorters[sensor_name] is not None:
            undistorter = undistorters[sensor_name]
            undistorted_intrinsics = undistorter.get_undistorted_intrinsics()
            intrinsics_rerun = rr.datatypes.Mat3x3([
                [undistorted_intrinsics['intrinsics'][0], 0, undistorted_intrinsics['intrinsics'][2]],
                [0, undistorted_intrinsics['intrinsics'][1], undistorted_intrinsics['intrinsics'][3]],
                [0, 0, 1]
            ])
            rr.log(
                rectified_prefix,
                rr.Pinhole(
                    image_from_camera=intrinsics_rerun,
                    resolution=undistorted_intrinsics['resolution'],
                ),
                static=True,
            )
        
        for i, (timestamp, filename) in enumerate(zip(timestamps_ns, filenames)):
            # Set current timestamp
            rr.set_time("timestamp", timestamp=timestamp)
            
            # Log image if file exists
            dataset_path_filename = os.path.join(dataset_path, body_name, sensor_name,'data', filename)
            if os.path.exists(dataset_path_filename):
                try:
                    # Always log the raw image
                    rr.log(
                        f"{sensor_prefix}/images",
                        rr.EncodedImage(path=dataset_path_filename)
                    )
                    # If undistortion is enabled, also log the rectified image
                    if undistort_images and sensor_name in undistorters and undistorters[sensor_name] is not None:
                        # Undistort the image
                        undistorter = undistorters[sensor_name]
                        undistorted_image = undistorter.undistort_image(dataset_path_filename)
                        if undistorted_image is not None:
                            # Convert BGR to RGB for Rerun (OpenCV uses BGR, Rerun expects RGB)
                            undistorted_image_rgb = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
                            rr.log(
                                f"{sensor_prefix}/rectified_images",
                                rr.Image(undistorted_image_rgb)
                            )
                        else:
                            print(f'     Warning: Failed to undistort image {dataset_path_filename}')
                except Exception as e:
                    print(f'     Warning: Failed to load image {dataset_path_filename}: {e}')
            else:
                print(f'     Warning: Image file not found: {dataset_path_filename}')
            
            # Optional: subsample for performance (log every Nth image)
            # Uncomment the lines below to log every 10th image for better performance
            # if i % 10 != 0:
            #     continue


# Convenience function for quick testing
def plot_euroc_dataset(dataset_path: str, max_time_sec: float = 30.0, blueprint_path: str = "", undistort_images: bool = False) -> None:
    """
    Quick plotting function for EuRoC datasets
    
    Args:
        dataset_path: Path to EuRoC dataset
        max_time_sec: Maximum time to visualize
        blueprint_path: Optional path to custom blueprint file
        undistort_images: Whether to apply camera undistortion
    """
    from .dataset_loader import dataset_load
    
    print(f"Loading and plotting EuRoC dataset: {dataset_path}")
    if undistort_images:
        print("Image undistortion enabled - this may take longer but will show corrected camera images")
    
    dataset = dataset_load(dataset_path)
    dataset_plot(dataset, 
                recording_name=f"EuRoC_{dataset_path.split('/')[-1]}", 
                dataset_path=dataset_path,
                max_time_sec=max_time_sec,
                blueprint_path=blueprint_path,
                undistort_images=undistort_images)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        max_time = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        blueprint_path = sys.argv[3] if len(sys.argv) > 3 else ""
        undistort = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False
        plot_euroc_dataset(dataset_path, max_time, blueprint_path, undistort)
    else:
        # Default test with EuRoC data
        plot_euroc_dataset("../../EuRoc_ASL/MH_01_easy", 30.0) 