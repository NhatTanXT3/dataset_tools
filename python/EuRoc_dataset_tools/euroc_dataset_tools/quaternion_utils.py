"""
Quaternion utilities for dataset loading - MATLAB compatibility layer
Using transformations library to match MATLAB ASL quaternion conventions

MATLAB quaternion format: [qw, qx, qy, qz] where qw >= 0 (minimal representation)
"""

import numpy as np
from transformations import (
    quaternion_from_matrix,
    quaternion_matrix, 
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_inverse
)


def q_min(q):
    """
    Ensure quaternion scalar part is non-negative (MATLAB q_min equivalent)
    
    Args:
        q: quaternion as [qw, qx, qy, qz] or array of quaternions (4, N)
        
    Returns:
        quaternion(s) with qw >= 0
    """
    q = np.asarray(q)
    
    if q.ndim == 1:
        # Single quaternion
        if q[0] < 0:
            return -q
        return q
    else:
        # Array of quaternions (4, N)
        q_min = q.copy()
        negative_mask = q[0, :] < 0
        q_min[:, negative_mask] = -q[:, negative_mask]
        return q_min


def q_C2q(rotation_matrix):
    """
    Convert rotation matrix to quaternion (MATLAB q_C2q equivalent)
    
    Args:
        rotation_matrix: 3x3 or 4x4 rotation matrix
        
    Returns:
        quaternion as [qw, qx, qy, qz] with qw >= 0
    """
    R = np.asarray(rotation_matrix)
    
    # Convert 3x3 to 4x4 homogeneous matrix if needed
    if R.shape == (3, 3):
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = R
        R = R_4x4
    elif R.shape != (4, 4):
        raise ValueError(f"Rotation matrix must be 3x3 or 4x4, got {R.shape}")
    
    # transformations.quaternion_from_matrix already ensures w >= 0
    return quaternion_from_matrix(R)


def q_q2C(quaternion):
    """
    Convert quaternion to rotation matrix (MATLAB q_q2C equivalent)
    
    Args:
        quaternion: [qw, qx, qy, qz]
        
    Returns:
        3x3 rotation matrix
    """
    # Get 4x4 homogeneous matrix and extract 3x3 rotation part
    homogeneous_matrix = quaternion_matrix(quaternion)
    return homogeneous_matrix[:3, :3]


def q_mul(q1, q2):
    """
    Quaternion multiplication (MATLAB q_mul equivalent)
    
    Args:
        q1, q2: quaternions as [qw, qx, qy, qz]
        
    Returns:
        quaternion product with minimal representation (qw >= 0)
    """
    result = quaternion_multiply(q1, q2)
    return q_min(result)


def q_inv(q):
    """
    Quaternion inverse (MATLAB q_inv equivalent)
    
    Args:
        q: quaternion as [qw, qx, qy, qz] or array (4, N)
        
    Returns:
        quaternion inverse with minimal representation
    """
    q = np.asarray(q)
    
    if q.ndim == 1:
        # Single quaternion: conjugate for unit quaternions
        # For MATLAB compatibility: [qw, -qx, -qy, -qz]
        return np.array([q[0], -q[1], -q[2], -q[3]])
    else:
        # Array of quaternions (4, N)
        q_inv_array = np.zeros_like(q)
        q_inv_array[0, :] = q[0, :]  # qw stays the same
        q_inv_array[1:4, :] = -q[1:4, :]  # negate vector part
        return q_inv_array


def skew_op(w):
    """
    Skew-symmetric matrix operator (MATLAB skewOp equivalent)
    
    Args:
        w: 3D vector [wx, wy, wz]
        
    Returns:
        3x3 skew-symmetric matrix
    """
    w = np.asarray(w)
    return np.array([
        [0,     -w[2],  w[1]],
        [w[2],   0,    -w[0]],
        [-w[1],  w[0],   0]
    ])


def q_norm(q):
    """
    Normalize quaternion (MATLAB q_norm equivalent)
    
    Args:
        q: quaternion as [qw, qx, qy, qz] or array (4, N)
        
    Returns:
        normalized quaternion(s)
    """
    q = np.asarray(q)
    
    if q.ndim == 1:
        # Single quaternion
        norm = np.linalg.norm(q)
        return q / norm if norm > 0 else q
    else:
        # Array of quaternions (4, N)
        norms = np.linalg.norm(q, axis=0)
        q_normalized = q / norms[np.newaxis, :]
        return q_normalized


# Validation functions for testing
def validate_quaternion_conventions():
    """
    Test quaternion operations against known results to ensure MATLAB compatibility
    """
    print("Testing quaternion utilities against MATLAB conventions...")
    
    # Test identity quaternion
    q_identity = np.array([1, 0, 0, 0])
    R_identity = q_q2C(q_identity)
    assert np.allclose(R_identity, np.eye(3)), "Identity quaternion failed"
    
    # Test rotation matrix to quaternion and back
    angle = np.pi / 4
    axis = np.array([0, 0, 1])
    R_test = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    
    q_test = q_C2q(R_test)
    R_reconstructed = q_q2C(q_test)
    assert np.allclose(R_test, R_reconstructed, atol=1e-10), "Round-trip conversion failed"
    
    # Test minimal representation
    q_positive = np.array([0.7071, 0, 0, 0.7071])
    q_negative = np.array([-0.7071, 0, 0, -0.7071])
    q_min_pos = q_min(q_positive)
    q_min_neg = q_min(q_negative)
    
    assert q_min_pos[0] >= 0, "Positive quaternion minimization failed"
    assert q_min_neg[0] >= 0, "Negative quaternion minimization failed"
    assert np.allclose(q_min_pos, q_min_neg), "Minimal representation not equivalent"
    
    print("✓ All quaternion convention tests passed!")


if __name__ == "__main__":
    validate_quaternion_conventions() 