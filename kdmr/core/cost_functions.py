"""
Cost functions for KDMR trajectory optimization.

This module defines various cost functions used in the optimization:
- Trajectory tracking cost (match human motion)
- Control effort cost (minimize torque)
- Smoothness cost (minimize jerk)
- Contact cost (match GRF)
- Constraint violation penalty
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class CostConfig:
    """Configuration for cost function weights."""
    tracking_pos: float = 100.0
    tracking_rot: float = 10.0
    control_effort: float = 0.01
    smoothness: float = 0.1
    contact_force: float = 1.0
    joint_limit: float = 1000.0
    torque_limit: float = 100.0
    ground_penetration: float = 1000.0
    foot_sliding: float = 100.0


class CostFunction(ABC):
    """Base class for cost functions."""
    
    @abstractmethod
    def compute(self, 
                state: np.ndarray, 
                control: Optional[np.ndarray] = None,
                **kwargs) -> float:
        """Compute cost value."""
        pass
    
    @abstractmethod
    def compute_gradient(self,
                        state: np.ndarray,
                        control: Optional[np.ndarray] = None,
                        **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute gradients w.r.t. state and control."""
        pass


class TrackingCost(CostFunction):
    """
    Cost for tracking reference trajectory.
    
    Penalizes deviation from reference positions and orientations.
    """
    
    def __init__(self,
                 weight_pos: float = 100.0,
                 weight_rot: float = 10.0,
                 joint_weights: Optional[Dict[str, float]] = None):
        """
        Initialize tracking cost.
        
        Args:
            weight_pos: Position tracking weight
            weight_rot: Rotation tracking weight
            joint_weights: Per-joint weights (optional)
        """
        self.weight_pos = weight_pos
        self.weight_rot = weight_rot
        self.joint_weights = joint_weights or {}
    
    def compute(self,
                state: np.ndarray,
                reference: np.ndarray,
                mask: Optional[np.ndarray] = None) -> float:
        """
        Compute tracking cost.
        
        Args:
            state: Current state (position + quaternion + joints)
            reference: Reference state
            mask: Optional mask for active DOFs
            
        Returns:
            Tracking cost value
        """
        # Split state into position, rotation, and joints
        pos_current = state[:3]
        pos_ref = reference[:3]
        
        quat_current = state[3:7]
        quat_ref = reference[3:7]
        
        joints_current = state[7:]
        joints_ref = reference[7:]
        
        # Position error
        pos_error = np.sum((pos_current - pos_ref) ** 2)
        
        # Rotation error (quaternion difference)
        # q_error = q_ref * q_current^{-1}
        quat_error = self._quat_error(quat_current, quat_ref)
        rot_error = 2 * np.arccos(np.clip(quat_error[0], -1, 1)) ** 2
        
        # Joint error
        if mask is not None:
            joint_error = np.sum(mask * (joints_current - joints_ref) ** 2)
        else:
            joint_error = np.sum((joints_current - joints_ref) ** 2)
        
        return self.weight_pos * pos_error + \
               self.weight_rot * rot_error + \
               self.weight_pos * joint_error
    
    def compute_gradient(self,
                        state: np.ndarray,
                        reference: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, None]:
        """Compute gradient w.r.t. state."""
        n = len(state)
        grad = np.zeros(n)
        
        # Position gradient
        grad[:3] = 2 * self.weight_pos * (state[:3] - reference[:3])
        
        # Rotation gradient (simplified)
        grad[3:7] = 2 * self.weight_rot * (state[3:7] - reference[3:7])
        
        # Joint gradient
        if mask is not None:
            grad[7:] = 2 * self.weight_pos * mask * (state[7:] - reference[7:])
        else:
            grad[7:] = 2 * self.weight_pos * (state[7:] - reference[7:])
        
        return grad, None
    
    def _quat_error(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Compute quaternion error q2 * q1^{-1}."""
        # q1^{-1} = conjugate for unit quaternion
        q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
        
        # Multiply q2 * q1_inv
        w1, x1, y1, z1 = q1_inv
        w2, x2, y2, z2 = q2
        
        return np.array([
            w2*w1 - x2*x1 - y2*y1 - z2*z1,
            w2*x1 + x2*w1 + y2*z1 - z2*y1,
            w2*y1 - x2*z1 + y2*w1 + z2*x1,
            w2*z1 + x2*y1 - y2*x1 + z2*w1
        ])


class ControlEffortCost(CostFunction):
    """
    Cost for control effort (torque).
    
    Penalizes large joint torques.
    """
    
    def __init__(self, 
                 weight: float = 0.01,
                 torque_weights: Optional[np.ndarray] = None):
        """
        Initialize control effort cost.
        
        Args:
            weight: Global weight for control effort
            torque_weights: Per-joint weights (optional)
        """
        self.weight = weight
        self.torque_weights = torque_weights
    
    def compute(self,
                state: np.ndarray,
                control: np.ndarray) -> float:
        """Compute control effort cost."""
        if self.torque_weights is not None:
            return self.weight * np.sum(self.torque_weights * control ** 2)
        return self.weight * np.sum(control ** 2)
    
    def compute_gradient(self,
                        state: np.ndarray,
                        control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients."""
        state_grad = np.zeros(len(state))
        
        if self.torque_weights is not None:
            control_grad = 2 * self.weight * self.torque_weights * control
        else:
            control_grad = 2 * self.weight * control
        
        return state_grad, control_grad


class SmoothnessCost(CostFunction):
    """
    Cost for trajectory smoothness.
    
    Penalizes high accelerations and jerk.
    """
    
    def __init__(self, 
                 weight: float = 0.1,
                 order: int = 2):
        """
        Initialize smoothness cost.
        
        Args:
            weight: Smoothness weight
            order: Derivative order (1=velocity, 2=acceleration, 3=jerk)
        """
        self.weight = weight
        self.order = order
    
    def compute(self,
                state_sequence: np.ndarray,
                dt: float) -> float:
        """
        Compute smoothness cost over trajectory.
        
        Args:
            state_sequence: State trajectory (T, n)
            dt: Time step
            
        Returns:
            Smoothness cost
        """
        cost = 0.0
        deriv = state_sequence.copy()
        
        for _ in range(self.order):
            deriv = np.diff(deriv, axis=0) / dt
        
        cost = self.weight * np.sum(deriv ** 2)
        return cost
    
    def compute_gradient(self,
                        state_sequence: np.ndarray,
                        dt: float) -> np.ndarray:
        """Compute gradient w.r.t. state sequence."""
        T, n = state_sequence.shape
        grad = np.zeros_like(state_sequence)
        
        # Finite difference gradient
        if self.order == 1:
            # Velocity penalty
            for t in range(1, T-1):
                grad[t] = 2 * self.weight * (2 * state_sequence[t] - 
                            state_sequence[t-1] - state_sequence[t+1]) / (dt ** 2)
        elif self.order == 2:
            # Acceleration penalty
            for t in range(2, T-2):
                grad[t] = 2 * self.weight * (6 * state_sequence[t] - 
                            4 * state_sequence[t-1] - 4 * state_sequence[t+1] +
                            state_sequence[t-2] + state_sequence[t+2]) / (dt ** 4)
        
        return grad


class ContactForceCost(CostFunction):
    """
    Cost for matching contact forces.
    
    Penalizes deviation from reference GRF.
    """
    
    def __init__(self,
                 weight: float = 1.0,
                 vertical_weight: float = 1.0,
                 horizontal_weight: float = 0.5):
        """
        Initialize contact force cost.
        
        Args:
            weight: Global weight
            vertical_weight: Weight for vertical force
            horizontal_weight: Weight for horizontal forces
        """
        self.weight = weight
        self.vertical_weight = vertical_weight
        self.horizontal_weight = horizontal_weight
    
    def compute(self,
                contact_force: np.ndarray,
                reference_force: np.ndarray,
                in_contact: bool = True) -> float:
        """
        Compute contact force cost.
        
        Args:
            contact_force: Current contact force (3,)
            reference_force: Reference GRF (3,)
            in_contact: Whether foot is in contact
            
        Returns:
            Contact force cost
        """
        if not in_contact:
            # Penalize any force when not in contact
            return self.weight * np.sum(contact_force ** 2)
        
        # Weighted error
        error = contact_force - reference_force
        weighted_error = np.array([
            self.horizontal_weight * error[0],
            self.horizontal_weight * error[1],
            self.vertical_weight * error[2]
        ])
        
        return self.weight * np.sum(weighted_error ** 2)
    
    def compute_gradient(self,
                        contact_force: np.ndarray,
                        reference_force: np.ndarray,
                        in_contact: bool = True) -> np.ndarray:
        """Compute gradient w.r.t. contact force."""
        if not in_contact:
            return 2 * self.weight * contact_force
        
        error = contact_force - reference_force
        return 2 * self.weight * np.array([
            self.horizontal_weight * error[0],
            self.horizontal_weight * error[1],
            self.vertical_weight * error[2]
        ])


class JointLimitCost(CostFunction):
    """
    Soft penalty for joint limit violations.
    
    Uses smooth barrier function near limits.
    """
    
    def __init__(self,
                 lower_limits: np.ndarray,
                 upper_limits: np.ndarray,
                 weight: float = 1000.0,
                 margin: float = 0.05):
        """
        Initialize joint limit cost.
        
        Args:
            lower_limits: Lower joint limits (rad)
            upper_limits: Upper joint limits (rad)
            weight: Penalty weight
            margin: Margin for smooth barrier
        """
        self.lower_limits = np.asarray(lower_limits)
        self.upper_limits = np.asarray(upper_limits)
        self.weight = weight
        self.margin = margin
    
    def compute(self,
                joint_angles: np.ndarray) -> float:
        """Compute joint limit penalty."""
        cost = 0.0
        
        # Lower limit violations
        lower_violation = self.lower_limits - joint_angles
        lower_penalty = np.maximum(lower_violation + self.margin, 0) ** 2
        cost += np.sum(lower_penalty)
        
        # Upper limit violations
        upper_violation = joint_angles - self.upper_limits
        upper_penalty = np.maximum(upper_violation + self.margin, 0) ** 2
        cost += np.sum(upper_penalty)
        
        return self.weight * cost
    
    def compute_gradient(self,
                        joint_angles: np.ndarray) -> np.ndarray:
        """Compute gradient w.r.t. joint angles."""
        grad = np.zeros_like(joint_angles)
        
        # Lower limit gradient
        lower_violation = self.lower_limits - joint_angles
        lower_active = lower_violation > -self.margin
        grad[lower_active] = -2 * self.weight * (lower_violation[lower_active] + self.margin)
        
        # Upper limit gradient
        upper_violation = joint_angles - self.upper_limits
        upper_active = upper_violation > -self.margin
        grad[upper_active] += 2 * self.weight * (upper_violation[upper_active] + self.margin)
        
        return grad


class GroundPenetrationCost(CostFunction):
    """
    Penalty for foot penetrating the ground.
    """
    
    def __init__(self,
                 ground_height: float = 0.0,
                 weight: float = 1000.0):
        """
        Initialize ground penetration cost.
        
        Args:
            ground_height: Height of ground plane
            weight: Penalty weight
        """
        self.ground_height = ground_height
        self.weight = weight
    
    def compute(self,
                foot_height: float,
                in_contact: bool = True) -> float:
        """
        Compute ground penetration cost.
        
        Args:
            foot_height: Height of foot above ground
            in_contact: Whether foot should be in contact
            
        Returns:
            Penetration cost
        """
        if foot_height < self.ground_height:
            # Penetration
            return self.weight * (self.ground_height - foot_height) ** 2
        
        if in_contact and foot_height > 0.01:
            # Should be in contact but floating
            return self.weight * foot_height ** 2
        
        return 0.0
    
    def compute_gradient(self,
                        foot_height: float,
                        in_contact: bool = True) -> float:
        """Compute gradient w.r.t. foot height."""
        if foot_height < self.ground_height:
            return -2 * self.weight * (self.ground_height - foot_height)
        
        if in_contact and foot_height > 0.01:
            return 2 * self.weight * foot_height
        
        return 0.0


class FootSlidingCost(CostFunction):
    """
    Penalty for foot sliding during contact.
    """
    
    def __init__(self,
                 weight: float = 100.0,
                 velocity_threshold: float = 0.1):
        """
        Initialize foot sliding cost.
        
        Args:
            weight: Penalty weight
            velocity_threshold: Velocity threshold for sliding detection
        """
        self.weight = weight
        self.velocity_threshold = velocity_threshold
    
    def compute(self,
                foot_velocity: np.ndarray,
                in_contact: bool = True) -> float:
        """
        Compute foot sliding cost.
        
        Args:
            foot_velocity: Foot velocity (3,)
            in_contact: Whether foot is in contact
            
        Returns:
            Sliding cost
        """
        if not in_contact:
            return 0.0
        
        # Horizontal velocity magnitude
        horizontal_vel = np.linalg.norm(foot_velocity[:2])
        
        if horizontal_vel > self.velocity_threshold:
            return self.weight * (horizontal_vel - self.velocity_threshold) ** 2
        
        return 0.0


class CostFunctions:
    """
    Container for all cost functions used in KDMR.
    
    Provides unified interface for computing total cost and gradients.
    """
    
    def __init__(self, config: Optional[CostConfig] = None):
        """
        Initialize cost functions.
        
        Args:
            config: Cost configuration
        """
        self.config = config or CostConfig()
        
        # Initialize individual cost functions
        self.tracking = TrackingCost(
            weight_pos=self.config.tracking_pos,
            weight_rot=self.config.tracking_rot
        )
        
        self.control = ControlEffortCost(
            weight=self.config.control_effort
        )
        
        self.smoothness = SmoothnessCost(
            weight=self.config.smoothness
        )
        
        self.contact = ContactForceCost(
            weight=self.config.contact_force
        )
    
    def compute_total_cost(self,
                          state: np.ndarray,
                          control: Optional[np.ndarray] = None,
                          reference: Optional[np.ndarray] = None,
                          contact_force: Optional[np.ndarray] = None,
                          reference_force: Optional[np.ndarray] = None,
                          in_contact: bool = False) -> float:
        """
        Compute total cost.
        
        Args:
            state: Current state
            control: Control input (torques)
            reference: Reference state for tracking
            contact_force: Current contact force
            reference_force: Reference GRF
            in_contact: Contact flag
            
        Returns:
            Total cost value
        """
        total = 0.0
        
        # Tracking cost
        if reference is not None:
            total += self.tracking.compute(state, reference)
        
        # Control effort cost
        if control is not None:
            total += self.control.compute(state, control)
        
        # Contact force cost
        if contact_force is not None and reference_force is not None:
            total += self.contact.compute(contact_force, reference_force, in_contact)
        
        return total
    
    def compute_total_gradient(self,
                              state: np.ndarray,
                              control: Optional[np.ndarray] = None,
                              reference: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute total gradient.
        
        Returns:
            Tuple of (state_gradient, control_gradient)
        """
        state_grad = np.zeros(len(state))
        control_grad = np.zeros(len(control)) if control is not None else None
        
        # Tracking gradient
        if reference is not None:
            track_grad, _ = self.tracking.compute_gradient(state, reference)
            state_grad += track_grad
        
        # Control gradient
        if control is not None:
            _, ctrl_grad = self.control.compute_gradient(state, control)
            control_grad = ctrl_grad
        
        return state_grad, control_grad
