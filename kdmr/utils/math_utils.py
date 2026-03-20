"""
Mathematical utilities for KDMR.

This module provides common mathematical operations used throughout KDMR:
- Quaternion operations (multiplication, conjugate, normalization)
- Rotation conversions (quaternion, rotation matrix, euler angles, axis-angle)
- Interpolation functions (SLERP for rotations, linear for positions)
- Utility functions for Lie algebra operations
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Union, Optional


class MathUtils:
    """
    Mathematical utilities for rotation and transformation operations.
    
    All quaternions are assumed to be in scalar-first format (w, x, y, z)
    unless otherwise specified.
    """
    
    # Small epsilon for numerical stability
    EPS = 1e-10
    
    @staticmethod
    def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions.
        
        Args:
            q1: First quaternion (w, x, y, z), shape (4,)
            q2: Second quaternion (w, x, y, z), shape (4,)
            
        Returns:
            Product quaternion q1 * q2, shape (4,)
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def quat_conjugate(q: np.ndarray) -> np.ndarray:
        """
        Compute quaternion conjugate.
        
        Args:
            q: Quaternion (w, x, y, z), shape (4,)
            
        Returns:
            Conjugate quaternion (w, -x, -y, -z), shape (4,)
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def quat_normalize(q: np.ndarray) -> np.ndarray:
        """
        Normalize quaternion to unit length.
        
        Args:
            q: Quaternion, shape (4,) or (N, 4)
            
        Returns:
            Normalized quaternion
        """
        q = np.asarray(q)
        if q.ndim == 1:
            norm = np.linalg.norm(q)
            if norm < MathUtils.EPS:
                return np.array([1.0, 0.0, 0.0, 0.0])
            return q / norm
        else:
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms = np.maximum(norms, MathUtils.EPS)
            return q / norms
    
    @staticmethod
    def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            q: Quaternion (w, x, y, z), shape (4,) or (N, 4)
            
        Returns:
            Rotation matrix, shape (3, 3) or (N, 3, 3)
        """
        q = np.asarray(q)
        if q.ndim == 1:
            rot = R.from_quat(q, scalar_first=True)
            return rot.as_matrix()
        else:
            rot = R.from_quat(q, scalar_first=True)
            return rot.as_matrix()
    
    @staticmethod
    def rotation_matrix_to_quat(mat: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion.
        
        Args:
            mat: Rotation matrix, shape (3, 3) or (N, 3, 3)
            
        Returns:
            Quaternion (w, x, y, z), shape (4,) or (N, 4)
        """
        mat = np.asarray(mat)
        if mat.ndim == 2:
            rot = R.from_matrix(mat)
            return rot.as_quat(scalar_first=True)
        else:
            rot = R.from_matrix(mat)
            return rot.as_quat(scalar_first=True)
    
    @staticmethod
    def quat_to_euler(q: np.ndarray, seq: str = 'xyz') -> np.ndarray:
        """
        Convert quaternion to Euler angles.
        
        Args:
            q: Quaternion (w, x, y, z), shape (4,)
            seq: Euler angle sequence ('xyz', 'zyx', etc.)
            
        Returns:
            Euler angles in radians, shape (3,)
        """
        rot = R.from_quat(q, scalar_first=True)
        return rot.as_euler(seq)
    
    @staticmethod
    def euler_to_quat(euler: np.ndarray, seq: str = 'xyz') -> np.ndarray:
        """
        Convert Euler angles to quaternion.
        
        Args:
            euler: Euler angles in radians, shape (3,)
            seq: Euler angle sequence
            
        Returns:
            Quaternion (w, x, y, z), shape (4,)
        """
        rot = R.from_euler(seq, euler)
        return rot.as_quat(scalar_first=True)
    
    @staticmethod
    def quat_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert quaternion to axis-angle representation.
        
        Args:
            q: Quaternion (w, x, y, z), shape (4,)
            
        Returns:
            Tuple of (axis, angle) where axis is unit vector and angle in radians
        """
        rot = R.from_quat(q, scalar_first=True)
        return rot.as_rotvec(), np.linalg.norm(rot.as_rotvec())
    
    @staticmethod
    def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Convert axis-angle to quaternion.
        
        Args:
            axis: Unit axis vector, shape (3,)
            angle: Rotation angle in radians
            
        Returns:
            Quaternion (w, x, y, z), shape (4,)
        """
        rotvec = axis * angle
        rot = R.from_rotvec(rotvec)
        return rot.as_quat(scalar_first=True)
    
    @staticmethod
    def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between two quaternions.
        
        Args:
            q1: Start quaternion (w, x, y, z), shape (4,)
            q2: End quaternion (w, x, y, z), shape (4,)
            t: Interpolation parameter in [0, 1]
            
        Returns:
            Interpolated quaternion, shape (4,)
        """
        q1 = MathUtils.quat_normalize(q1)
        q2 = MathUtils.quat_normalize(q2)
        
        # Compute dot product
        dot = np.dot(q1, q2)
        
        # If negative dot, negate one quaternion to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return MathUtils.quat_normalize(result)
        
        # SLERP formula
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        theta = theta_0 * t
        
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return MathUtils.quat_normalize(s0 * q1 + s1 * q2)
    
    @staticmethod
    def slerp_batch(q1: np.ndarray, q2: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Batch SLERP interpolation for multiple quaternions.
        
        Args:
            q1: Start quaternions, shape (N, 4)
            q2: End quaternions, shape (N, 4)
            t: Interpolation parameters, shape (M,)
            
        Returns:
            Interpolated quaternions, shape (M, N, 4)
        """
        q1 = np.asarray(q1)
        q2 = np.asarray(q2)
        t = np.asarray(t)
        
        result = np.zeros((len(t), len(q1), 4))
        for i, ti in enumerate(t):
            for j in range(len(q1)):
                result[i, j] = MathUtils.slerp(q1[j], q2[j], ti)
        
        return result
    
    @staticmethod
    def quat_error(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Compute quaternion error (rotation from q1 to q2).
        
        Args:
            q1: First quaternion, shape (4,)
            q2: Second quaternion, shape (4,)
            
        Returns:
            Error quaternion q2 * q1^{-1}, shape (4,)
        """
        return MathUtils.quat_multiply(q2, MathUtils.quat_conjugate(q1))
    
    @staticmethod
    def quat_log(q: np.ndarray) -> np.ndarray:
        """
        Quaternion logarithm mapping to tangent space.
        
        Maps unit quaternion to 3D vector in tangent space at identity.
        
        Args:
            q: Unit quaternion (w, x, y, z), shape (4,)
            
        Returns:
            3D vector in tangent space, shape (3,)
        """
        q = MathUtils.quat_normalize(q)
        w, v = q[0], q[1:4]
        
        norm_v = np.linalg.norm(v)
        
        if norm_v < MathUtils.EPS:
            return np.zeros(3)
        
        theta = np.arctan2(norm_v, w)
        return theta * v / norm_v
    
    @staticmethod
    def quat_exp(v: np.ndarray) -> np.ndarray:
        """
        Quaternion exponential mapping from tangent space.
        
        Maps 3D vector in tangent space to unit quaternion.
        
        Args:
            v: 3D vector in tangent space, shape (3,)
            
        Returns:
            Unit quaternion (w, x, y, z), shape (4,)
        """
        theta = np.linalg.norm(v)
        
        if theta < MathUtils.EPS:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        axis = v / theta
        w = np.cos(theta / 2)
        xyz = np.sin(theta / 2) * axis
        
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    @staticmethod
    def linear_interpolate(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
        """
        Linear interpolation between two points.
        
        Args:
            p1: Start point, shape (3,) or (N,)
            p2: End point, shape (3,) or (N,)
            t: Interpolation parameter in [0, 1]
            
        Returns:
            Interpolated point
        """
        return p1 + t * (p2 - p1)
    
    @staticmethod
    def compute_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute velocity from position sequence using finite differences.
        
        Args:
            positions: Position sequence, shape (N, 3) or (N, D)
            dt: Time step
            
        Returns:
            Velocities, shape (N-1, 3) or (N-1, D)
        """
        return np.diff(positions, axis=0) / dt
    
    @staticmethod
    def compute_acceleration(velocities: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute acceleration from velocity sequence.
        
        Args:
            velocities: Velocity sequence, shape (N, 3) or (N, D)
            dt: Time step
            
        Returns:
            Accelerations, shape (N-1, 3) or (N-1, D)
        """
        return np.diff(velocities, axis=0) / dt
    
    @staticmethod
    def compute_jerk(positions: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute jerk (derivative of acceleration) for smoothness evaluation.
        
        Args:
            positions: Position sequence, shape (N, 3) or (N, D)
            dt: Time step
            
        Returns:
            Jerk values, shape (N-3, 3) or (N-3, D)
        """
        v = MathUtils.compute_velocity(positions, dt)
        a = MathUtils.compute_acceleration(v, dt)
        j = MathUtils.compute_acceleration(a, dt)
        return j
    
    @staticmethod
    def angular_velocity_from_quat(quats: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute angular velocity from quaternion sequence.
        
        Args:
            quats: Quaternion sequence, shape (N, 4)
            dt: Time step
            
        Returns:
            Angular velocities in world frame, shape (N-1, 3)
        """
        ang_vel = []
        for i in range(len(quats) - 1):
            q_error = MathUtils.quat_error(quats[i], quats[i+1])
            log_q = MathUtils.quat_log(q_error)
            ang_vel.append(2.0 * log_q / dt)
        
        return np.array(ang_vel)
    
    @staticmethod
    def transform_point(pos: np.ndarray, quat: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
        Transform a point by position and orientation.
        
        Args:
            pos: Translation, shape (3,)
            quat: Orientation quaternion, shape (4,)
            point: Point to transform, shape (3,)
            
        Returns:
            Transformed point, shape (3,)
        """
        # Convert quaternion to rotation matrix
        R_mat = MathUtils.quat_to_rotation_matrix(quat)
        return R_mat @ point + pos
    
    @staticmethod
    def inverse_transform_point(pos: np.ndarray, quat: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
        Inverse transform a point (world to local frame).
        
        Args:
            pos: Translation, shape (3,)
            quat: Orientation quaternion, shape (4,)
            point: Point in world frame, shape (3,)
            
        Returns:
            Point in local frame, shape (3,)
        """
        R_mat = MathUtils.quat_to_rotation_matrix(quat)
        return R_mat.T @ (point - pos)
    
    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        """
        Compute skew-symmetric matrix from vector.
        
        Used for cross product: skew(v) @ w = v x w
        
        Args:
            v: 3D vector, shape (3,)
            
        Returns:
            Skew-symmetric matrix, shape (3, 3)
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    @staticmethod
    def smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Apply moving average smoothing to signal.
        
        Args:
            signal: Input signal, shape (N,) or (N, D)
            window_size: Size of smoothing window
            
        Returns:
            Smoothed signal
        """
        signal = np.asarray(signal)
        if signal.ndim == 1:
            kernel = np.ones(window_size) / window_size
            return np.convolve(signal, kernel, mode='same')
        else:
            kernel = np.ones((window_size, 1)) / window_size
            result = np.zeros_like(signal)
            for i in range(signal.shape[1]):
                result[:, i] = np.convolve(signal[:, i], kernel.flatten(), mode='same')
            return result
    
    @staticmethod
    def resample_trajectory(positions: np.ndarray, 
                           original_fps: float, 
                           target_fps: float) -> np.ndarray:
        """
        Resample trajectory to different frame rate.
        
        Args:
            positions: Position sequence, shape (N, D)
            original_fps: Original frame rate
            target_fps: Target frame rate
            
        Returns:
            Resampled positions
        """
        from scipy.interpolate import interp1d
        
        n_original = len(positions)
        n_target = int(n_original * target_fps / original_fps)
        
        original_time = np.linspace(0, 1, n_original)
        target_time = np.linspace(0, 1, n_target)
        
        result = np.zeros((n_target, positions.shape[1]))
        for i in range(positions.shape[1]):
            interp_func = interp1d(original_time, positions[:, i], kind='cubic')
            result[:, i] = interp_func(target_time)
        
        return result
