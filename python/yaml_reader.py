"""
YAML configuration reader for dataset loading - MATLAB compatibility layer
Matches MATLAB dataset_read_yaml.m functionality including data matrix conversion

Handles sensor.yaml files with transformation matrices and parameters
"""

import yaml
import numpy as np
from typing import Dict, Any, Union


def dataset_read_yaml(yaml_file: str) -> Dict[str, Any]:
    """
    Read YAML file and convert numerical entries (MATLAB dataset_read_yaml equivalent)
    
    Args:
        yaml_file: Path to YAML file
        
    Returns:
        Dictionary with parsed YAML content, matrices converted from data/rows/cols format
    """
    with open(yaml_file, 'r') as f:
        yaml_content = yaml.safe_load(f)
    
    if yaml_content is None:
        yaml_content = {}
    
    # Ensure sensor_type is specified (MATLAB assertion equivalent)
    if 'sensor_type' not in yaml_content:
        raise ValueError(f"No sensor_type specified in {yaml_file}")
    
    # Process all fields for data matrix conversion
    for field_name, field_value in yaml_content.items():
        if isinstance(field_value, dict) and 'data' in field_value:
            # Found data entry - convert to matrix (MATLAB format)
            try:
                data_raw = field_value['data']
                cols = field_value['cols']
                rows = field_value['rows']
                
                # Convert to numpy array and reshape
                data_array = np.array(data_raw, dtype=np.float64)
                
                # MATLAB reshapes as (rows, cols) then transposes for column-major
                # Python equivalent: reshape to (cols, rows) then transpose
                data_matrix = data_array.reshape(rows, cols)
                
                # Replace the field with the matrix
                yaml_content[field_name] = data_matrix
                
            except (KeyError, ValueError, TypeError) as e:
                print(f"Warning: Could not convert data field '{field_name}': {e}")
                # Keep original data if conversion fails
                continue
    
    return yaml_content


def parse_transformation_matrix(yaml_content: Dict[str, Any], field_name: str = 'T_BS') -> np.ndarray:
    """
    Extract transformation matrix from YAML content
    
    Args:
        yaml_content: Parsed YAML dictionary
        field_name: Name of the transformation matrix field (default: 'T_BS')
        
    Returns:
        4x4 transformation matrix as numpy array
    """
    if field_name not in yaml_content:
        return np.eye(4)  # Return identity if not found
    
    T_matrix = yaml_content[field_name]
    
    if isinstance(T_matrix, (list, np.ndarray)):
        T_matrix = np.array(T_matrix, dtype=np.float64)
        
        # Ensure it's 4x4
        if T_matrix.shape != (4, 4):
            raise ValueError(f"Transformation matrix {field_name} must be 4x4, got {T_matrix.shape}")
    
    return T_matrix


def extract_sensor_parameters(yaml_content: Dict[str, Any]) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Extract key sensor parameters from YAML content
    
    Args:
        yaml_content: Parsed YAML dictionary
        
    Returns:
        Dictionary with standardized sensor parameters
    """
    params = {}
    
    # Basic sensor info
    params['sensor_type'] = yaml_content.get('sensor_type', 'unknown')
    params['comment'] = yaml_content.get('comment', '')
    
    # Transformation matrix
    if 'T_BS' in yaml_content:
        params['T_BS'] = parse_transformation_matrix(yaml_content, 'T_BS')
    
    # Common parameters
    if 'rate_hz' in yaml_content:
        params['rate_hz'] = float(yaml_content['rate_hz'])
    
    # Camera-specific parameters
    if 'resolution' in yaml_content:
        params['resolution'] = yaml_content['resolution']
    
    if 'camera_model' in yaml_content:
        params['camera_model'] = yaml_content['camera_model']
    
    if 'intrinsics' in yaml_content:
        params['intrinsics'] = np.array(yaml_content['intrinsics'], dtype=np.float64)
    
    if 'distortion_model' in yaml_content:
        params['distortion_model'] = yaml_content['distortion_model']
    
    if 'distortion_coefficients' in yaml_content:
        params['distortion_coefficients'] = np.array(yaml_content['distortion_coefficients'], dtype=np.float64)
    
    # IMU-specific parameters
    imu_params = [
        'gyroscope_noise_density',
        'gyroscope_random_walk', 
        'accelerometer_noise_density',
        'accelerometer_random_walk'
    ]
    
    for param in imu_params:
        if param in yaml_content:
            params[param] = float(yaml_content[param])
    
    return params


def validate_yaml_reader():
    """
    Test YAML reader functionality
    """
    # Create a test YAML content similar to EuRoC format
    test_yaml_content = {
        'sensor_type': 'imu',
        'comment': 'Test IMU sensor',
        'T_BS': np.eye(4).tolist(),
        'rate_hz': 200.0,
        'gyroscope_noise_density': 1.6968e-04,
        'accelerometer_noise_density': 2.0000e-3
    }
    
    # Test transformation matrix parsing
    T_BS = parse_transformation_matrix(test_yaml_content, 'T_BS')
    assert T_BS.shape == (4, 4), "Transformation matrix shape incorrect"
    assert np.allclose(T_BS, np.eye(4)), "Identity transformation failed"
    
    # Test parameter extraction
    params = extract_sensor_parameters(test_yaml_content)
    assert params['sensor_type'] == 'imu', "Sensor type extraction failed"
    assert params['rate_hz'] == 200.0, "Rate extraction failed"
    
    print("✓ YAML reader validation passed!")


if __name__ == "__main__":
    validate_yaml_reader() 