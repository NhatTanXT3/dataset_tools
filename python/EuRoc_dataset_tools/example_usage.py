#!/usr/bin/env python3
"""
Example usage of the new utility functions for dataset loading
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from euroc_dataset_tools.dataset_loader import dataset_load, get_dataset_path, get_camera_images_info

def example_usage():
    """
    Example demonstrating the new utility functions
    """
    print("=== EuRoC Dataset Tools - Utility Functions Example ===\n")
    
    # Example dataset path (modify this to point to your EuRoC dataset)
    dataset_path = "../../EuRoc_ASL/MH_01_easy"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        print("Please modify the dataset_path variable to point to a valid EuRoC dataset.")
        return
    
    try:
        # Load the dataset
        print(f"Loading dataset from: {dataset_path}")
        dataset = dataset_load(dataset_path)
        
        # Example 1: Get dataset path
        print("\n1. Getting dataset path:")
        path = get_dataset_path(dataset)
        print(f"   Dataset path: {path}")
        
        # Example 2: Get camera images for different timestamp criteria
        print("\n2. Getting camera images:")
        
        # Find first camera sensor
        camera_sensor = None
        body_name = None
        sensor_name = None
        
        for body in dataset['body']:
            for sensor in body['sensor']:
                if sensor.get('sensor_type') == 'camera':
                    camera_sensor = sensor
                    body_name = body['name']
                    sensor_name = sensor['name']
                    break
            if camera_sensor:
                break
        
        if not camera_sensor:
            print("   No camera sensor found in dataset")
            return
        
        print(f"   Using camera: {body_name}/{sensor_name}")
        
        # Get first image (timestamp=0)
        paths, timestamps = get_camera_images_info(dataset, body_name, sensor_name, 0)
        if paths:
            print(f"   First image: {paths[0]}")
            print(f"   Timestamp: {timestamps[0]}")
        
        # Get last image (timestamp=-1)
        paths, timestamps = get_camera_images_info(dataset, body_name, sensor_name, -1)
        if paths:
            print(f"   Last image: {paths[0]}")
            print(f"   Timestamp: {timestamps[0]}")
        
        # Get all images in range [0, -1]
        paths, timestamps = get_camera_images_info(dataset, body_name, sensor_name, [0, -1])
        if paths:
            print(f"   Total images: {len(paths)}")
            print(f"   Time range: {timestamps[0]} to {timestamps[-1]}")
        
        # Example 3: Get images in specific time range
        if timestamps and len(timestamps) > 2:
            # Get middle 50% of images
            start_idx = len(timestamps) // 4
            end_idx = 3 * len(timestamps) // 4
            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx]
            
            print(f"\n3. Getting images in time range [{start_time}, {end_time}]:")
            paths, timestamps_range = get_camera_images_info(dataset, body_name, sensor_name, [start_time, end_time])
            if paths:
                print(f"   Found {len(paths)} images in range")
                print(f"   First: {paths[0]}")
                print(f"   Last: {paths[-1]}")
        
        # Example 4: Get images with max_duration limit
        if timestamps:
            # Get first 5 seconds of images
            max_duration_seconds = 5.0  # 5 seconds
            print(f"\n4. Getting first 5 seconds of images (max_duration={max_duration_seconds}s):")
            paths, timestamps_limited = get_camera_images_info(dataset, body_name, sensor_name, [0, -1], max_duration=max_duration_seconds)
            if paths:
                print(f"   Found {len(paths)} images in first 5 seconds")
                print(f"   First: {paths[0]}")
                print(f"   Last: {paths[-1]}")
                duration_seconds = (timestamps_limited[-1] - timestamps_limited[0]) / 1e9
                print(f"   Actual duration: {duration_seconds:.2f} seconds")
        
        print("\n=== Example completed successfully! ===")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    example_usage() 