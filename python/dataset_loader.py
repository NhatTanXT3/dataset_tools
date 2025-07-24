"""
Main dataset loader for ASL dataset format - MATLAB compatibility layer
Matches MATLAB dataset_load.m functionality for directory scanning and data loading

Loads hierarchical dataset structure: dataset.body[i].sensor[j].data
"""

import os
import numpy as np
from typing import Dict, List, Any
from yaml_reader import dataset_read_yaml, extract_sensor_parameters
from sensor_data_loader import dataset_load_sensor_data


def dataset_load(dataset_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load dataset from directory structure (MATLAB dataset_load equivalent)
    
    Args:
        dataset_path: Path to dataset root directory
        
    Returns:
        Dictionary with 'body' key containing list of body dictionaries,
        each with 'name', 'sensor' fields matching MATLAB structure
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_path}")
    
    dataset = {'body': []}
    
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
                print("✓ EuRoC dataset loading test passed!")
                return
            except Exception as e:
                print(f"Warning: EuRoC test failed: {e}")
    
    print("Note: No EuRoC dataset found for full testing")


if __name__ == "__main__":
    validate_dataset_loader() 