"""
Python Dataset Tools for ASL Dataset Format
============================================

A Python implementation of the MATLAB dataset tools for loading ASL format datasets,
specifically designed for EuRoC MAV dataset compatibility.

Main modules:
- dataset_loader: Main dataset loading functionality
- sensor_data_loader: CSV sensor data parsing for different sensor types  
- yaml_reader: YAML configuration file parsing
- quaternion_utils: Quaternion operations using transformations library

Usage:
------
```python
from dataset_tools.python import dataset_load, print_dataset_summary

# Load dataset
dataset = dataset_load('/path/to/euroc/MH_01_easy')

# Print summary
print_dataset_summary(dataset)

# Access specific sensors
imu_data = get_sensor_by_name(dataset, 'mav0', 'imu0')
camera_data = get_sensor_by_name(dataset, 'mav0', 'cam0')
```

Requirements:
- numpy
- pandas  
- PyYAML
- transformations (for quaternion operations)
"""

__version__ = "1.0.0"
__author__ = "Python reimplementation of ASL MATLAB dataset tools"

# Import main functions for easy access
try:
    from .dataset_loader import (
        dataset_load, 
        get_sensor_by_name, 
        get_sensor_by_type, 
        print_dataset_summary
    )
    from .quaternion_utils import (
        q_min, q_C2q, q_q2C, q_mul, q_inv, q_norm, skew_op
    )
    from .yaml_reader import (
        dataset_read_yaml, 
        parse_transformation_matrix, 
        extract_sensor_parameters
    )
    from .sensor_data_loader import dataset_load_sensor_data
    from .dataset_plot import dataset_plot, plot_euroc_dataset
    
    __all__ = [
        'dataset_load',
        'get_sensor_by_name', 
        'get_sensor_by_type',
        'print_dataset_summary',
        'dataset_plot',
        'plot_euroc_dataset',
        'q_min', 'q_C2q', 'q_q2C', 'q_mul', 'q_inv', 'q_norm', 'skew_op',
        'dataset_read_yaml', 
        'parse_transformation_matrix',
        'extract_sensor_parameters',
        'dataset_load_sensor_data'
    ]
except ImportError as e:
    # Handle case where dependencies are not available
    print(f"Warning: Some dataset tools dependencies not available: {e}")
    __all__ = [] 