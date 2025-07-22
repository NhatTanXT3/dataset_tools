# Python Dataset Tools for ASL Dataset Format

A Python implementation of the MATLAB dataset tools for loading ASL format datasets, specifically designed for **EuRoC MAV dataset** compatibility.

## 🎯 Features

- **✅ Full MATLAB compatibility**: Identical data structures and quaternion conventions
- **🚀 Efficient loading**: Uses pandas for fast CSV parsing
- **🧭 Correct quaternions**: Uses transformations library with same conventions as MATLAB
- **📊 Rich analysis**: Built-in dataset summary and sensor analysis tools
- **🔧 Extensible**: Easy to add new sensor types and formats

## 📦 Installation

```bash
# Activate your virtual environment
source venv/bin/activate

# Install required dependencies (already installed in this environment)
pip install numpy pandas PyYAML transformations
```

## 🚀 Quick Start

### Basic Usage

```python
from dataset_tools.python import dataset_load, print_dataset_summary

# Load EuRoC dataset
dataset = dataset_load('../../EuRoc_ASL/MH_01_easy')

# Print dataset overview
print_dataset_summary(dataset)
```

### Accessing Sensor Data

```python
from dataset_tools.python import get_sensor_by_name, get_sensor_by_type

# Get specific sensor by name
imu_sensor = get_sensor_by_name(dataset, 'mav0', 'imu0')
cam0_sensor = get_sensor_by_name(dataset, 'mav0', 'cam0')

# Get all sensors of a specific type
all_cameras = get_sensor_by_type(dataset, 'mav0', 'camera')
all_imus = get_sensor_by_type(dataset, 'mav0', 'imu')

# Access IMU data
imu_data = imu_sensor['data']
timestamps = imu_data['t']          # Timestamps (N,)
gyro_data = imu_data['omega']       # Angular velocity (3, N)
accel_data = imu_data['a']          # Linear acceleration (3, N)

# Access camera data  
cam_data = cam0_sensor['data']
cam_timestamps = cam_data['t']      # Timestamps (N,)
image_files = cam_data['filenames'] # Image filenames (N,)
```

### Working with Quaternions

```python
from dataset_tools.python import q_C2q, q_q2C, q_min

# Get pose/groundtruth data
gt_sensor = get_sensor_by_name(dataset, 'mav0', 'state_groundtruth_estimate0')
gt_data = gt_sensor['data']

# Access pose data
positions = gt_data['p_RS_R']       # Positions (3, N) 
quaternions = gt_data['q_RS']       # Quaternions (4, N) - minimal representation

# Convert rotation matrix to quaternion (MATLAB compatible)
import numpy as np
R = np.eye(3)  # Example rotation matrix
q = q_C2q(R)   # Convert to quaternion [qw, qx, qy, qz]

# Convert quaternion back to rotation matrix
R_reconstructed = q_q2C(q)

# Ensure minimal quaternion representation (qw >= 0)
q_minimal = q_min(quaternions)
```

## 🔧 Command Line Usage

Run the test script (equivalent to MATLAB `dataset_load_test.m`):

```bash
# Test with default EuRoC dataset
python dataset_load_test.py

# Test with specific dataset path
python dataset_load_test.py /path/to/your/dataset

# Run with quaternion validation
python dataset_load_test.py --validate-quaternions

# Summary only (no detailed analysis)
python dataset_load_test.py --summary-only
```

## 📁 Module Overview

### Core Modules

- **`dataset_loader.py`**: Main dataset loading (equivalent to `dataset_load.m`)
- **`sensor_data_loader.py`**: CSV sensor data parsing (equivalent to `dataset_load_sensor_data.m`)
- **`yaml_reader.py`**: YAML configuration parsing (equivalent to `dataset_read_yaml.m`)
- **`quaternion_utils.py`**: Quaternion operations (equivalent to `quaternion/` folder)

### Supported Sensor Types

| Sensor Type | CSV Format | Data Fields |
|-------------|------------|-------------|
| `imu` | `timestamp,ωx,ωy,ωz,ax,ay,az` | `t`, `omega`, `a` |
| `camera` | `timestamp,filename` | `t`, `filenames` |
| `position` | `timestamp,px,py,pz` | `t`, `p_RS_R` |
| `pose` | `timestamp,px,py,pz,qw,qx,qy,qz` | `t`, `p_RS_R`, `q_RS` |
| `visual-inertial` | `timestamp,px,py,pz,qw,qx,qy,qz,vx,vy,vz,bωx,bωy,bωz,bax,bay,baz` | `t`, `p_RS_R`, `q_RS`, `v_RS_R`, `bw_S`, `ba_S` |

## 🧪 Validation Results

**✅ Successfully tested with EuRoC MH_01_easy dataset:**

```
Body: mav0
  Sensors: 5
    cam0 (camera): 3682 samples, 184.05s, 20.0Hz
    imu0 (imu): 36820 samples, 184.10s, 200.0Hz  
    leica0 (position): 3099 samples, 187.76s
    state_groundtruth_estimate0 (visual-inertial): 36382 samples, 181.90s, 200.0Hz
    cam1 (camera): 3682 samples, 184.05s, 20.0Hz

✅ Quaternion validation: All 36382 quaternions have w >= 0 (minimal representation)
✅ Data integrity: Quaternion norms in range [0.999999, 1.000068]
✅ Timing accuracy: Expected vs actual rates match within 0.1%
```

## 🔄 MATLAB Compatibility

This implementation maintains **100% compatibility** with the original MATLAB code:

| MATLAB Function | Python Equivalent | Compatibility |
|-----------------|-------------------|---------------|
| `dataset_load()` | `dataset_load()` | ✅ Identical structure |
| `dataset_load_sensor_data()` | `dataset_load_sensor_data()` | ✅ Same CSV parsing |
| `dataset_read_yaml()` | `dataset_read_yaml()` | ✅ Matrix conversion included |
| `q_C2q()` | `q_C2q()` | ✅ Same quaternion convention |
| `q_q2C()` | `q_q2C()` | ✅ Identical rotation matrices |
| `q_min()` | `q_min()` | ✅ Same minimal representation |
| `q_mul()` | `q_mul()` | ✅ Identical quaternion algebra |

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed in your virtual environment
2. **Path Issues**: Use absolute paths or check current working directory
3. **Data Format**: Ensure CSV files match expected EuRoC format
4. **Missing Files**: Check that `body.yaml` and `sensor.yaml` files exist

### Debug Mode

```python
# Enable detailed logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with validation
python dataset_load_test.py --validate-quaternions
```

## 📚 References

- [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
- [ASL Dataset Tools (MATLAB)](https://github.com/ethz-asl/dataset_tools)
- [Transformations Library](https://pypi.org/project/transformations/)

---

**🎉 Ready to use!** This Python implementation provides a complete, efficient, and MATLAB-compatible way to work with EuRoC and other ASL format datasets. 