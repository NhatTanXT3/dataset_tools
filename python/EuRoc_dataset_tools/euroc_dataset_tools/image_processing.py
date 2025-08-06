#!/usr/bin/env python3
"""
Image processing utilities for dataset visualization
Provides camera image undistortion functionality for EuRoC-style datasets

Features:
- Radial-tangential distortion correction using OpenCV
- Support for pinhole camera model with intrinsic parameters
- Efficient caching of undistortion maps for performance
- Integration with Rerun visualization pipeline
"""

import numpy as np
import os
from typing import Dict, Any, Tuple, Optional, Union
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV (cv2) not available. Image undistortion will be disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Some image format support may be limited.")


class CameraUndistorter:
    """
    Camera undistortion utility for removing lens distortion from images
    
    Supports radial-tangential distortion model commonly used in camera calibration.
    Caches undistortion maps for efficient processing of multiple images.
    """
    
    def __init__(self, sensor_config: Dict[str, Any]):
        """
        Initialize undistorter with camera sensor configuration
        
        Args:
            sensor_config: Camera sensor configuration dictionary containing:
                - intrinsics: [fu, fv, cu, cv] - focal lengths and principal point
                - distortion_coefficients: [k1, k2, p1, p2, k3] - distortion parameters
                - resolution: [width, height] - image dimensions
                - distortion_model: should be 'radial-tangential'
        """
        self.sensor_config = sensor_config
        self.camera_matrix = None
        self.dist_coeffs = None
        self.map1 = None
        self.map2 = None
        self.new_camera_matrix = None
        self.roi = None
        
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for image undistortion but not available")
        
        self._setup_calibration_parameters()
    
    def _setup_calibration_parameters(self):
        """Setup camera calibration parameters from sensor configuration"""
        
        # Validate required parameters
        required_params = ['intrinsics', 'distortion_coefficients', 'resolution', 'distortion_model']
        for param in required_params:
            if param not in self.sensor_config:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Check distortion model
        if self.sensor_config['distortion_model'] != 'radial-tangential':
            raise ValueError(f"Unsupported distortion model: {self.sensor_config['distortion_model']}. Only 'radial-tangential' is supported.")
        
        # Extract intrinsic parameters
        intrinsics = self.sensor_config['intrinsics']  # [fu, fv, cu, cv]
        if len(intrinsics) != 4:
            raise ValueError(f"Invalid intrinsics format. Expected [fu, fv, cu, cv], got {intrinsics}")
        
        fu, fv, cu, cv = intrinsics
        self.camera_matrix = np.array([
            [fu, 0,  cu],
            [0,  fv, cv],
            [0,  0,  1]
        ], dtype=np.float32)
        
        # Extract distortion coefficients
        dist_coeffs = self.sensor_config['distortion_coefficients']
        # OpenCV expects at least 4 coefficients [k1, k2, p1, p2] and optionally k3
        if len(dist_coeffs) < 4:
            raise ValueError(f"Invalid distortion coefficients. Expected at least [k1, k2, p1, p2], got {dist_coeffs}")
        
        # Pad with zeros if we have less than 5 coefficients
        if len(dist_coeffs) == 4:
            dist_coeffs = list(dist_coeffs) + [0.0]  # Add k3 = 0
        
        self.dist_coeffs = np.array(dist_coeffs[:5], dtype=np.float32)  # Use up to 5 coefficients
        
        # Get image resolution
        self.resolution = self.sensor_config['resolution']  # [width, height]
        if len(self.resolution) != 2:
            raise ValueError(f"Invalid resolution format. Expected [width, height], got {self.resolution}")
        
        self.width, self.height = self.resolution
        
        # Setup undistortion maps
        self._compute_undistortion_maps()
    
    def _compute_undistortion_maps(self):
        """Compute undistortion maps for efficient image processing"""
        
        # Get optimal new camera matrix and valid pixel ROI
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, 
            self.dist_coeffs, 
            (self.width, self.height), 
            alpha=1,  # alpha=1 retains all pixels, alpha=0 crops to valid region
            newImgSize=(self.width, self.height)
        )
        
        # Compute undistortion and rectification maps
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,  # No rectification matrix
            self.new_camera_matrix,
            (self.width, self.height),
            cv2.CV_32FC1
        )
        
        print(f"     Undistortion maps computed for resolution {self.width}x{self.height}")
        print(f"     Original camera matrix:\n{self.camera_matrix}")
        print(f"     New camera matrix:\n{self.new_camera_matrix}")
        print(f"     Distortion coefficients: {self.dist_coeffs}")
    
    def undistort_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Undistort a single image from file path
        
        Args:
            image_path: Path to the input image file
            
        Returns:
            Undistorted image as numpy array (BGR format), or None if loading failed
        """
        
        if not os.path.exists(image_path):
            logging.warning(f"Image file not found: {image_path}")
            return None
        
        try:
            # Load image using OpenCV (BGR format)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                logging.warning(f"Failed to load image: {image_path}")
                return None
            
            # Check image dimensions match expected resolution
            img_height, img_width = image.shape[:2]
            if img_width != self.width or img_height != self.height:
                logging.warning(f"Image resolution mismatch. Expected {self.width}x{self.height}, got {img_width}x{img_height}")
                # Resize if needed
                image = cv2.resize(image, (self.width, self.height))
            
            # Apply undistortion using precomputed maps
            undistorted = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
            
            return undistorted
            
        except Exception as e:
            logging.error(f"Error undistorting image {image_path}: {e}")
            return None
    
    def undistort_image_array(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort an image provided as numpy array
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Undistorted image as numpy array
        """
        
        if self.map1 is None or self.map2 is None:
            raise RuntimeError("Undistortion maps not computed. Call _compute_undistortion_maps() first.")
        
        # Check image dimensions
        img_height, img_width = image.shape[:2]
        if img_width != self.width or img_height != self.height:
            # Resize if needed
            image = cv2.resize(image, (self.width, self.height))
        
        # Apply undistortion
        undistorted = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
        
        return undistorted
    
    def get_undistorted_intrinsics(self) -> Dict[str, Any]:
        """
        Get the new camera intrinsics after undistortion
        
        Returns:
            Dictionary with updated intrinsics for use with Rerun Pinhole
        """
        
        if self.new_camera_matrix is None:
            raise RuntimeError("Undistortion maps not computed")
        
        fu = self.new_camera_matrix[0, 0]
        fv = self.new_camera_matrix[1, 1]
        cu = self.new_camera_matrix[0, 2]
        cv = self.new_camera_matrix[1, 2]
        
        return {
            'intrinsics': [fu, fv, cu, cv],
            'camera_matrix': self.new_camera_matrix,
            'resolution': [self.width, self.height],
            'roi': self.roi
        }


def create_undistorter_for_sensor(sensor: Dict[str, Any]) -> Optional[CameraUndistorter]:
    """
    Create an undistorter for a camera sensor configuration
    
    Args:
        sensor: Camera sensor dictionary from dataset
        
    Returns:
        CameraUndistorter instance, or None if undistortion not possible
    """
    
    if not CV2_AVAILABLE:
        logging.warning("OpenCV not available, skipping undistortion setup")
        return None
    
    # Check if this is a camera sensor with required parameters
    if sensor.get('sensor_type') != 'camera':
        return None
    
    required_params = ['camera_model', 'intrinsics', 'distortion_coefficients', 'distortion_model', 'resolution']
    for param in required_params:
        if param not in sensor:
            logging.warning(f"Camera sensor missing {param}, skipping undistortion")
            return None
    
    if sensor['camera_model'] != 'pinhole':
        logging.warning(f"Unsupported camera model: {sensor['camera_model']}, skipping undistortion")
        return None
    
    if sensor['distortion_model'] != 'radial-tangential':
        logging.warning(f"Unsupported distortion model: {sensor['distortion_model']}, skipping undistortion")
        return None
    
    try:
        undistorter = CameraUndistorter(sensor)
        return undistorter
        
    except Exception as e:
        logging.error(f"Failed to create undistorter for sensor {sensor.get('name', 'unknown')}: {e}")
        return None


def save_undistorted_image(undistorted_image: np.ndarray, output_path: str) -> bool:
    """
    Save undistorted image to file
    
    Args:
        undistorted_image: Undistorted image array (BGR format)
        output_path: Path to save the image
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        cv2.imwrite(output_path, undistorted_image)
        return True
    except Exception as e:
        logging.error(f"Failed to save undistorted image to {output_path}: {e}")
        return False


# Example usage and testing functions
def test_undistortion(dataset_path: str, sensor_name: str = "cam0", max_images: int = 5) -> None:
    """
    Test undistortion functionality on a few images from a dataset
    
    Args:
        dataset_path: Path to EuRoC dataset
        sensor_name: Name of camera sensor to test
        max_images: Maximum number of images to process for testing
    """
    
    if not CV2_AVAILABLE:
        print("OpenCV not available, cannot test undistortion")
        return
    
    from .dataset_loader import dataset_load, get_sensor_by_name
    
    print(f"Testing undistortion on dataset: {dataset_path}")
    
    try:
        # Load dataset
        dataset = dataset_load(dataset_path)
        
        # Get camera sensor
        camera_sensor = get_sensor_by_name(dataset, 'mav0', sensor_name)
        if camera_sensor is None:
            print(f"Camera sensor {sensor_name} not found")
            return
        
        # Create undistorter
        undistorter = create_undistorter_for_sensor(camera_sensor)
        if undistorter is None:
            print("Failed to create undistorter")
            return
        
        # Get some image paths
        data = camera_sensor.get('data', {})
        if 't' not in data or 'filenames' not in data:
            print("No image data found in sensor")
            return
        
        filenames = data['filenames'][:max_images]
        
        print(f"Processing {len(filenames)} test images...")
        
        for i, filename in enumerate(filenames):
            image_path = os.path.join(dataset_path, 'mav0', sensor_name, 'data', filename)
            
            print(f"  Processing image {i+1}/{len(filenames)}: {filename}")
            
            # Undistort image
            undistorted = undistorter.undistort_image(image_path)
            
            if undistorted is not None:
                # Save undistorted image for inspection
                output_path = f"test_undistorted_{sensor_name}_{i+1}.png"
                if save_undistorted_image(undistorted, output_path):
                    print(f"    Saved undistorted image to: {output_path}")
                else:
                    print("    Failed to save undistorted image")
            else:
                print("    Failed to undistort image")
        
        print("Undistortion test completed!")
        
    except Exception as e:
        print(f"Error during undistortion test: {e}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        sensor_name = sys.argv[2] if len(sys.argv) > 2 else "cam0"
        test_undistortion(dataset_path, sensor_name)
    else:
        # Default test
        test_undistortion("../../EuRoc_ASL/MH_01_easy", "cam0") 