"""
EuRoC Dataset Tools Package
===========================

This package provides Python tools for loading ASL format datasets.
"""

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
    from .image_processing import CameraUndistorter, create_undistorter_for_sensor
    
    __all__ = [
        'dataset_load',
        'get_sensor_by_name', 
        'get_sensor_by_type',
        'print_dataset_summary',
        'dataset_plot',
        'plot_euroc_dataset',
        'CameraUndistorter',
        'create_undistorter_for_sensor',
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