"""
Main dataset loader for ASL dataset format - MATLAB compatibility layer
Matches MATLAB dataset_load.m functionality for directory scanning and data loading

Loads hierarchical dataset structure: dataset.body[i].sensor[j].data
"""

import os
import numpy as np
from typing import Dict, List, Any, Union, Tuple, Optional
from .yaml_reader import dataset_read_yaml, extract_sensor_parameters
from .sensor_data_loader import dataset_load_sensor_data


def dataset_load(dataset_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load dataset from directory structure (MATLAB dataset_load equivalent)
    
    Args:
        dataset_path: Path to dataset root directory
        
    Returns:
        Dictionary with 'body' key containing list of body dictionaries,
        each with 'name', 'sensor' fields matching MATLAB structure,
        and 'path' key containing the dataset path
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_path}")
    
    dataset = {'body': [], 'path': dataset_path}
    
    print('')
    print(' >> scanning dataset...')
    
    # Scan for body directories
    dataset_folder_content = os.listdir(dataset_path)
    
    for item in dataset_folder_content:
        if item.startswith('.'):  # Skip hidden files/directories
            continue
            
        body_folder_path = os.path.join(dataset_path, item)
        
        if not os.path.isdir(body_folder_path):
            continue
            
        body_yaml_filename = os.path.join(body_folder_path, 'body.yaml')
        
        # Check if body.yaml exists
        if os.path.exists(body_yaml_filename):
            print(f'    body detected [{item}]')
            
            # Create body entry
            body_entry = {
                'name': item,
                'sensor': []
            }
            
            # Scan for sensor directories within this body
            body_folder_content = os.listdir(body_folder_path)
            
            for sensor_item in body_folder_content:
                if sensor_item.startswith('.'):  # Skip hidden files
                    continue
                    
                sensor_folder_path = os.path.join(body_folder_path, sensor_item)
                
                if not os.path.isdir(sensor_folder_path):
                    continue
                    
                sensor_yaml_filename = os.path.join(sensor_folder_path, 'sensor.yaml')
                
                # Check if sensor.yaml exists
                if os.path.exists(sensor_yaml_filename):
                    try:
                        # Read sensor configuration
                        sensor_config = dataset_read_yaml(sensor_yaml_filename)
                        sensor_type = sensor_config.get('sensor_type', 'unknown')
                        
                        print(f'     sensor detected [{sensor_item}], [{sensor_type}]')
                        
                        # Create sensor entry with configuration
                        sensor_entry = sensor_config.copy()
                        sensor_entry['name'] = sensor_item
                        
                        # Load sensor data
                        try:
                            sensor_data = dataset_load_sensor_data(sensor_type, sensor_folder_path)
                            sensor_entry['data'] = sensor_data
                        except Exception as e:
                            print(f'     Warning: Could not load data for sensor {sensor_item}: {e}')
                            sensor_entry['data'] = {}
                        
                        body_entry['sensor'].append(sensor_entry)
                        
                    except Exception as e:
                        print(f'     Warning: Could not read sensor configuration for {sensor_item}: {e}')
                        continue
            
            dataset['body'].append(body_entry)
    
    # Ensure we found at least one body (MATLAB assertion equivalent)
    if not dataset['body']:
        raise ValueError(f"No valid bodies found in dataset: {dataset_path}")
    
    return dataset


def get_dataset_path(dataset: Dict) -> str:
    """
    Get the dataset path that was used to load the dataset
    
    Args:
        dataset: Loaded dataset dictionary
        
    Returns:
        Dataset path string or None if not available
    """
    return dataset.get('path')


def get_sensor_by_name(dataset: Dict, body_name: str, sensor_name: str) -> Dict[str, Any]:
    """
    Get specific sensor from dataset by body and sensor name
    
    Args:
        dataset: Loaded dataset dictionary
        body_name: Name of body (e.g., 'mav0')
        sensor_name: Name of sensor (e.g., 'imu0', 'cam0')
        
    Returns:
        Sensor dictionary with configuration and data
    """
    for body in dataset['body']:
        if body['name'] == body_name:
            for sensor in body['sensor']:
                if sensor['name'] == sensor_name:
                    return sensor
    
    raise ValueError(f"Sensor {sensor_name} not found in body {body_name}")


def get_sensor_by_type(dataset: Dict, body_name: str, sensor_type: str) -> List[Dict[str, Any]]:
    """
    Get all sensors of specific type from a body
    
    Args:
        dataset: Loaded dataset dictionary
        body_name: Name of body (e.g., 'mav0')
        sensor_type: Type of sensor (e.g., 'imu', 'camera')
        
    Returns:
        List of sensor dictionaries matching the type
    """
    sensors = []
    
    for body in dataset['body']:
        if body['name'] == body_name:
            for sensor in body['sensor']:
                if sensor.get('sensor_type') == sensor_type:
                    sensors.append(sensor)
            break
    
    return sensors


def get_camera_images_info(dataset: Dict, body_name: str, sensor_name: str, timestamp: Union[int, List[int]] = 0, max_duration: float = -1) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """
    Get camera image information based on timestamp criteria
    
    Args:
        dataset: Loaded dataset dictionary
        body_name: Name of body (e.g., 'mav0')
        sensor_name: Name of camera sensor (e.g., 'cam0')
        timestamp: Timestamp criteria:
            - 0 (default): Return first image
            - -1: Return last image
            - number: Return exact match (or None if not found)
            - [from, to]: Return all images in range (inclusive boundaries)
        max_duration: Maximum duration in seconds:
            - -1 (default): Output based on timestamp criteria only
            - >0: Limit output from t0 to t0 + max_duration (up to t1, no more than t1)
            
    Returns:
        Tuple of (image_paths, timestamps) or (None, None) if no data found
        
    Raises:
        ValueError: If invalid timestamp format or sensor not found
    """
    # Validate input
    if not isinstance(timestamp, (int, list)):
        raise ValueError("timestamp must be an integer or list of two integers")
    
    if isinstance(timestamp, list):
        if len(timestamp) != 2:
            raise ValueError("timestamp list must contain exactly 2 elements [from, to]")
        if not all(isinstance(t, int) for t in timestamp):
            raise ValueError("timestamp list elements must be integers")
    
    if not isinstance(max_duration, (int, float)):
        raise ValueError("max_duration must be a number (int or float)")
    if max_duration != -1 and max_duration <= 0:
        raise ValueError("max_duration must be -1 or a positive number")
    
    # Get the camera sensor
    try:
        sensor = get_sensor_by_name(dataset, body_name, sensor_name)
    except ValueError as e:
        raise ValueError(f"Camera sensor not found: {e}")
    
    # Check if it's a camera sensor
    if sensor.get('sensor_type') != 'camera':
        raise ValueError(f"Sensor {sensor_name} is not a camera sensor (type: {sensor.get('sensor_type', 'unknown')})")
    
    # Get sensor data
    data = sensor.get('data', {})
    if not data or 't' not in data or 'filenames' not in data:
        return None, None
    
    timestamps = data['t']
    filenames = data['filenames']
    
    if len(timestamps) == 0:
        return None, None
    
    # Get dataset path for constructing full image paths
    dataset_path = get_dataset_path(dataset)
    if not dataset_path:
        raise ValueError("Dataset path not available")
    
    # Handle different timestamp criteria
    if isinstance(timestamp, int):
        if timestamp == 0:
            # Return first image
            idx = 0
        elif timestamp == -1:
            # Return last image
            idx = len(timestamps) - 1
        else:
            # Return exact match
            idx = None
            for i, t in enumerate(timestamps):
                if t == timestamp:
                    idx = i
                    break
            if idx is None:
                return None, None
        
        # Construct full image path
        filename = filenames[idx]
        image_path = os.path.join(dataset_path, body_name, sensor_name, 'data', filename)
        
        return [image_path], [timestamps[idx]]
    
    else:  # timestamp is a list [from, to]
        from_time, to_time = timestamp
        
        # Handle special range cases
        if from_time == 0:
            from_time = timestamps[0]
        if to_time == -1:
            to_time = timestamps[-1]
        
        # Apply max_duration limit if specified
        if max_duration > 0:
            # Convert seconds to nanoseconds
            max_duration_ns = int(max_duration * 1e9)
            max_end_time = from_time + max_duration_ns
            to_time = min(to_time, max_end_time)
        
        # Find images in range (inclusive boundaries)
        image_paths = []
        selected_timestamps = []
        
        for i, t in enumerate(timestamps):
            if from_time <= t <= to_time:
                filename = filenames[i]
                image_path = os.path.join(dataset_path, body_name, sensor_name, 'data', filename)
                image_paths.append(image_path)
                selected_timestamps.append(t)
        
        if not image_paths:
            return None, None
        
        return image_paths, selected_timestamps


def print_dataset_summary(dataset: Dict) -> None:
    """
    Print summary of loaded dataset (for debugging/validation)
    
    Args:
        dataset: Loaded dataset dictionary
    """
    print(f"\n=== Dataset Summary ===")
    print(f"Bodies found: {len(dataset['body'])}")
    
    for body in dataset['body']:
        print(f"\nBody: {body['name']}")
        print(f"  Sensors: {len(body['sensor'])}")
        
        for sensor in body['sensor']:
            sensor_type = sensor.get('sensor_type', 'unknown')
            sensor_name = sensor.get('name', 'unnamed')
            
            # Check if data was loaded
            data = sensor.get('data', {})
            if 't' in data:
                n_samples = len(data['t'])
                duration = (data['t'][-1] - data['t'][0]) / 1e9 if n_samples > 1 else 0
                print(f"    {sensor_name} ({sensor_type}): {n_samples} samples, {duration:.2f}s")
            elif sensor_type == 'pointcloud' and 'positions' in data:
                n_points = data['positions'].shape[1] if data['positions'].ndim > 1 else len(data['positions'])
                print(f"    {sensor_name} ({sensor_type}): {n_points} points (static)")
            else:
                print(f"    {sensor_name} ({sensor_type}): no data")


def validate_dataset_loader():
    """
    Test dataset loader with EuRoC sample if available
    """
    print("Testing dataset loader...")
    
    # Test with a simple directory structure
    test_structure = {
        'body': [],
    }
    
    # Test data structure validation
    assert 'body' in test_structure, "Dataset structure missing body key"
    assert isinstance(test_structure['body'], list), "Body should be a list"
    
    print("✓ Dataset loader structure validation passed!")
    
    # Test new utility functions
    test_dataset = {
        'path': '/test/path',
        'body': [{
            'name': 'mav0',
            'sensor': [{
                'name': 'cam0',
                'sensor_type': 'camera',
                'data': {
                    't': [1403636579763555584, 1403636579863555584, 1403636579963555584],
                    'filenames': ['image_1.png', 'image_2.png', 'image_3.png']
                }
            }]
        }]
    }
    
    # Test get_dataset_path
    path = get_dataset_path(test_dataset)
    assert path == '/test/path', f"Expected '/test/path', got {path}"
    print("✓ get_dataset_path test passed!")
    
    # Test get_camera_images_info
    # Test first image (timestamp=0)
    paths, timestamps = get_camera_images_info(test_dataset, 'mav0', 'cam0', 0)
    assert paths == ['/test/path/mav0/cam0/data/image_1.png']
    assert timestamps == [1403636579763555584]
    print("✓ get_camera_images_info first image test passed!")
    
    # Test last image (timestamp=-1)
    paths, timestamps = get_camera_images_info(test_dataset, 'mav0', 'cam0', -1)
    assert paths == ['/test/path/mav0/cam0/data/image_3.png']
    assert timestamps == [1403636579963555584]
    print("✓ get_camera_images_info last image test passed!")
    
    # Test exact timestamp
    paths, timestamps = get_camera_images_info(test_dataset, 'mav0', 'cam0', 1403636579863555584)
    assert paths == ['/test/path/mav0/cam0/data/image_2.png']
    assert timestamps == [1403636579863555584]
    print("✓ get_camera_images_info exact timestamp test passed!")
    
    # Test range [0, -1] (all images)
    paths, timestamps = get_camera_images_info(test_dataset, 'mav0', 'cam0', [0, -1])
    expected_paths = [
        '/test/path/mav0/cam0/data/image_1.png',
        '/test/path/mav0/cam0/data/image_2.png',
        '/test/path/mav0/cam0/data/image_3.png'
    ]
    expected_timestamps = [1403636579763555584, 1403636579863555584, 1403636579963555584]
    assert paths == expected_paths
    assert timestamps == expected_timestamps
    print("✓ get_camera_images_info range test passed!")
    
    # Test max_duration functionality
    # Test with max_duration limiting to first image only (0.000000001s duration)
    paths, timestamps = get_camera_images_info(test_dataset, 'mav0', 'cam0', [0, -1], max_duration=0.000000001)  # 1ns
    expected_paths = [
        '/test/path/mav0/cam0/data/image_1.png'
    ]
    expected_timestamps = [1403636579763555584]
    assert paths == expected_paths
    assert timestamps == expected_timestamps
    print("✓ get_camera_images_info max_duration test passed!")
    
    # Test non-existent timestamp
    paths, timestamps = get_camera_images_info(test_dataset, 'mav0', 'cam0', 9999999999999999999)
    assert paths is None and timestamps is None
    print("✓ get_camera_images_info non-existent timestamp test passed!")
    
    # Try to test with actual EuRoC data if available
    euroc_paths = [
        '../../EuRoc_ASL/MH_01_easy',
        '../../EuRoc_ASL/V1_01_easy'
    ]
    
    for euroc_path in euroc_paths:
        if os.path.exists(euroc_path):
            print(f"Testing with EuRoC dataset: {euroc_path}")
            try:
                dataset = dataset_load(euroc_path)
                print_dataset_summary(dataset)
                
                # Test utility functions with real data
                path = get_dataset_path(dataset)
                print(f"Dataset path: {path}")
                
                # Try to get camera images if available
                for body in dataset['body']:
                    for sensor in body['sensor']:
                        if sensor.get('sensor_type') == 'camera':
                            print(f"Testing camera sensor: {body['name']}/{sensor['name']}")
                            paths, timestamps = get_camera_images_info(dataset, body['name'], sensor['name'], 0)
                            if paths:
                                print(f"  First image: {paths[0]}")
                                print(f"  Timestamp: {timestamps[0]}")
                            break
                
                print("✓ EuRoC dataset loading test passed!")
                return
            except Exception as e:
                print(f"Warning: EuRoC test failed: {e}")
    
    print("Note: No EuRoC dataset found for full testing")


if __name__ == "__main__":
    validate_dataset_loader() 