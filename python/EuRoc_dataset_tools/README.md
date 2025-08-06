# EuRoC Dataset Tools

A Python package for loading and analyzing EuRoC MAV datasets in ASL format.

## 📦 Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Install the package
```bash
# Navigate to the package directory
cd dataset_tools/python/EuRoc_dataset_tools

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Install with visualization support
```bash
# Install with optional visualization dependencies
pip install -e .[viz]
```

## 🚀 Quick Usage

### Basic Dataset Loading
```python
import euroc_dataset_tools

# Load a EuRoC dataset
dataset = euroc_dataset_tools.dataset_load('/path/to/euroc/MH_01_easy')

# Print dataset summary
euroc_dataset_tools.print_dataset_summary(dataset)
```

### Access Sensor Data
```python
# Get specific sensors
imu_data = euroc_dataset_tools.get_sensor_by_name(dataset, 'mav0', 'imu0')
camera_data = euroc_dataset_tools.get_sensor_by_name(dataset, 'mav0', 'cam0')

# Get all sensors of a type
all_cameras = euroc_dataset_tools.get_sensor_by_type(dataset, 'mav0', 'camera')
all_imus = euroc_dataset_tools.get_sensor_by_type(dataset, 'mav0', 'imu')
```

### Interactive Visualization
```python
# Visualize dataset with rerun
euroc_dataset_tools.dataset_plot(dataset, max_time_sec=30.0)

# With camera undistortion
euroc_dataset_tools.dataset_plot(dataset, undistort_images=True, max_time_sec=30.0)
```

### Quaternion Operations
```python
# Convert rotation matrix to quaternion
import numpy as np
R = np.eye(3)
q = euroc_dataset_tools.q_C2q(R)

# Convert quaternion to rotation matrix
R_reconstructed = euroc_dataset_tools.q_q2C(q)

# Ensure minimal representation
q_minimal = euroc_dataset_tools.q_min(q)
```

## 📋 Available Functions

### Core Functions
- `dataset_load()` - Load EuRoC dataset
- `print_dataset_summary()` - Print dataset overview
- `get_sensor_by_name()` - Get specific sensor
- `get_sensor_by_type()` - Get all sensors of a type

### Visualization
- `dataset_plot()` - Interactive 3D visualization
- `plot_euroc_dataset()` - Alternative plotting function

### Quaternion Utilities
- `q_C2q()` - Rotation matrix to quaternion
- `q_q2C()` - Quaternion to rotation matrix
- `q_min()` - Ensure minimal representation
- `q_mul()` - Quaternion multiplication
- `q_inv()` - Quaternion inverse
- `q_norm()` - Quaternion normalization
- `skew_op()` - Skew-symmetric operator

### YAML Processing
- `dataset_read_yaml()` - Read sensor configuration
- `parse_transformation_matrix()` - Parse transformation matrices
- `extract_sensor_parameters()` - Extract sensor parameters

### Image Processing
- `CameraUndistorter` - Camera undistortion class
- `create_undistorter_for_sensor()` - Create undistorter for sensor

## 🎯 Supported Sensor Types

- **IMU**: Gyroscope and accelerometer data
- **Camera**: Image timestamps and filenames
- **Position**: 3D position data
- **Pose**: Position and orientation (quaternions)
- **Visual-Inertial**: Full state with biases

## 📁 Package Structure

```
EuRoc_dataset_tools/
├── setup.py                    # Package configuration
├── README.md                   # This file
└── euroc_dataset_tools/        # Main package
    ├── __init__.py            # Package interface
    ├── dataset_loader.py      # Core loading functionality
    ├── sensor_data_loader.py  # CSV data parsing
    ├── yaml_reader.py         # Configuration parsing
    ├── quaternion_utils.py    # Quaternion operations
    ├── dataset_plot.py        # Visualization
    ├── image_processing.py    # Camera undistortion
    ├── demo_plot.py           # Demo scripts
    ├── dataset_load_test.py   # Testing utilities
    └── README.md             # Detailed documentation
```

## 🔧 Dependencies

### Required
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `PyYAML>=5.4.0`
- `transformations>=2021.6.6`

### Optional (for visualization)
- `rerun-sdk>=0.8.0`
- `opencv-python>=4.5.0`

## 📚 Documentation

For detailed documentation, examples, and advanced usage, see the [detailed README](euroc_dataset_tools/README.md) in the package directory.

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Ensure package is installed correctly
2. **Missing dependencies**: Install with `pip install -e .[viz]` for visualization
3. **Path issues**: Use absolute paths for dataset directories

### Development
```bash
# Install in development mode for live editing
pip install -e .

# Run tests
python euroc_dataset_tools/dataset_load_test.py
```

## 📄 License

This package is a Python reimplementation of the ASL MATLAB dataset tools for EuRoC MAV datasets. 