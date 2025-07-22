#!/usr/bin/env python3
"""
Python equivalent of MATLAB dataset_load_test.m
Tests the dataset loading functionality with EuRoC datasets

Usage: python dataset_load_test.py [dataset_path]
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import dataset_load, print_dataset_summary, get_sensor_by_name, get_sensor_by_type
from quaternion_utils import validate_quaternion_conventions
import numpy as np


def main():
    """
    Main test function equivalent to MATLAB dataset_load_test.m
    """
    parser = argparse.ArgumentParser(description='Test dataset loading functionality')
    parser.add_argument('dataset_path', nargs='?', 
                       default='../../EuRoc_ASL/MH_01_easy',
                       help='Path to dataset folder (default: ../../EuRoc_ASL/MH_01_easy)')
    parser.add_argument('--validate-quaternions', action='store_true',
                       help='Run quaternion validation tests')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print dataset summary without detailed analysis')
    
    args = parser.parse_args()
    
    # Print header (MATLAB style)
    print('')
    print(f' > dataset_load_test [{args.dataset_path}]')
    print('')
    
    # Validate quaternion utilities if requested
    if args.validate_quaternions:
        print('Running quaternion validation tests...')
        validate_quaternion_conventions()
        print('')
    
    # Check if dataset path exists
    if not os.path.exists(args.dataset_path):
        print(f' > Dataset folder does not exist: {args.dataset_path}')
        print('   Please set correct dataset path or use default EuRoC location.')
        sys.exit(1)
    
    try:
        # Load dataset (equivalent to MATLAB dataset_load)
        print('Loading dataset...')
        dataset = dataset_load(args.dataset_path)
        print('Dataset loaded successfully!')
        
        # Print dataset summary
        print_dataset_summary(dataset)
        
        if not args.summary_only:
            # Perform detailed analysis
            analyze_dataset(dataset)
            
        print('')
        print('✓ Dataset loading test completed successfully!')
        
    except Exception as e:
        print(f'Error loading dataset: {e}')
        sys.exit(1)


def analyze_dataset(dataset):
    """
    Perform detailed analysis of loaded dataset
    
    Args:
        dataset: Loaded dataset dictionary
    """
    print('\n=== Detailed Dataset Analysis ===')
    
    # Analyze each body
    for body in dataset['body']:
        body_name = body['name']
        print(f'\nAnalyzing body: {body_name}')
        
        # Check for common sensors
        analyze_imu_sensors(dataset, body_name)
        analyze_camera_sensors(dataset, body_name)
        analyze_groundtruth_sensors(dataset, body_name)


def analyze_imu_sensors(dataset, body_name):
    """
    Analyze IMU sensors in the dataset
    """
    imu_sensors = get_sensor_by_type(dataset, body_name, 'imu')
    
    if not imu_sensors:
        return
        
    print(f'  IMU Sensors ({len(imu_sensors)}):')
    
    for imu in imu_sensors:
        name = imu['name']
        data = imu.get('data', {})
        
        if 't' in data and 'omega' in data and 'a' in data:
            n_samples = len(data['t'])
            duration = (data['t'][-1] - data['t'][0]) / 1e9
            rate = n_samples / duration if duration > 0 else 0
            
            # Compute statistics
            omega_rms = np.sqrt(np.mean(data['omega']**2, axis=1))
            accel_mean = np.mean(data['a'], axis=1)
            accel_std = np.std(data['a'], axis=1)
            
            print(f'    {name}: {n_samples} samples, {duration:.2f}s, {rate:.1f}Hz')
            print(f'      Gyro RMS: [{omega_rms[0]:.4f}, {omega_rms[1]:.4f}, {omega_rms[2]:.4f}] rad/s')
            print(f'      Accel mean: [{accel_mean[0]:.2f}, {accel_mean[1]:.2f}, {accel_mean[2]:.2f}] m/s²')
            print(f'      Accel std: [{accel_std[0]:.3f}, {accel_std[1]:.3f}, {accel_std[2]:.3f}] m/s²')
            
            # Check rate_hz parameter
            if 'rate_hz' in imu:
                expected_rate = imu['rate_hz']
                print(f'      Expected rate: {expected_rate}Hz, Actual: {rate:.1f}Hz')


def analyze_camera_sensors(dataset, body_name):
    """
    Analyze camera sensors in the dataset
    """
    camera_sensors = get_sensor_by_type(dataset, body_name, 'camera')
    
    if not camera_sensors:
        return
        
    print(f'  Camera Sensors ({len(camera_sensors)}):')
    
    for camera in camera_sensors:
        name = camera['name']
        data = camera.get('data', {})
        
        if 't' in data and 'filenames' in data:
            n_images = len(data['t'])
            duration = (data['t'][-1] - data['t'][0]) / 1e9
            rate = n_images / duration if duration > 0 else 0
            
            print(f'    {name}: {n_images} images, {duration:.2f}s, {rate:.1f}Hz')
            
            # Check camera parameters
            if 'resolution' in camera:
                print(f'      Resolution: {camera["resolution"]}')
            if 'intrinsics' in camera:
                intrinsics = camera['intrinsics']
                print(f'      Intrinsics: fu={intrinsics[0]:.1f}, fv={intrinsics[1]:.1f}, '
                     f'cu={intrinsics[2]:.1f}, cv={intrinsics[3]:.1f}')


def analyze_groundtruth_sensors(dataset, body_name):
    """
    Analyze groundtruth/pose sensors in the dataset
    """
    pose_sensors = []
    
    # Check for different groundtruth sensor types
    for sensor_type in ['pose', 'visual-inertial']:
        pose_sensors.extend(get_sensor_by_type(dataset, body_name, sensor_type))
    
    # Also check for specific groundtruth sensor names
    try:
        gt_sensor = get_sensor_by_name(dataset, body_name, 'state_groundtruth_estimate0')
        pose_sensors.append(gt_sensor)
    except ValueError:
        pass
    
    if not pose_sensors:
        return
        
    print(f'  Pose/Groundtruth Sensors ({len(pose_sensors)}):')
    
    for pose_sensor in pose_sensors:
        name = pose_sensor['name']
        sensor_type = pose_sensor.get('sensor_type', 'unknown')
        data = pose_sensor.get('data', {})
        
        if 't' in data:
            n_samples = len(data['t'])
            duration = (data['t'][-1] - data['t'][0]) / 1e9
            rate = n_samples / duration if duration > 0 else 0
            
            print(f'    {name} ({sensor_type}): {n_samples} samples, {duration:.2f}s, {rate:.1f}Hz')
            
            if 'p_RS_R' in data:
                positions = data['p_RS_R']
                pos_range = np.max(positions, axis=1) - np.min(positions, axis=1)
                print(f'      Position range: [{pos_range[0]:.2f}, {pos_range[1]:.2f}, {pos_range[2]:.2f}] m')
            
            if 'q_RS' in data:
                quaternions = data['q_RS']
                # Check quaternion validity
                q_norms = np.linalg.norm(quaternions, axis=0)
                print(f'      Quaternion norm range: [{np.min(q_norms):.6f}, {np.max(q_norms):.6f}]')
                
                # Check minimal representation
                n_positive_w = np.sum(quaternions[0, :] >= 0)
                print(f'      Minimal quaternions: {n_positive_w}/{n_samples} have w >= 0')


if __name__ == '__main__':
    main() 