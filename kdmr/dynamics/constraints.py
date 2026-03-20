"""
Physical constraints for KDMR trajectory optimization.

This module defines constraint classes for:
- Dynamics equation constraints
- Contact complementarity constraints
- Friction cone constraints
- Joint and torque limits
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ConstraintViolation:
    """Information about constraint violation."""
    name: str
    value: float
    is_violated: bool
    gradient: Optional[np.ndarray] = None


class Constraint(ABC):
    """Base class for constraints."""
    
    @abstractmethod
    def compute(self, 
                state: np.ndarray,
                control: Optional[np.ndarray] = None,
                **kwargs) -> float:
        """Compute constraint value (should be <= 0 for satisfied)."""
        pass
    
    @abstractmethod
    def compute_gradient(self,
                        state: np.ndarray,
                        control: Optional[np.ndarray] = None,
                        **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute gradients w.r.t. state and control."""
        pass
    
    def is_satisfied(self,
                    state: np.ndarray,
                    control: Optional[np.ndarray] = None,
                    tolerance: float = 1e-6,
                    **kwargs) -> bool:
        """Check if constraint is satisfied."""
        return self.compute(state, control, **kwargs) <= tolerance


class DynamicsConstraint(Constraint):
    """
    Dynamics equation constraint.
    
    Enforces: M(q)q̈ + C(q, q̇) + G(q) = τ + J^T @ F_contact
    
    This is typically handled implicitly by the dynamics model,
    but can be used to penalize dynamics violations.
    """
    
    def __init__(self,
                 dynamics_func: Callable,
                 weight: float = 1000.0):
        """
        Initialize dynamics constraint.
        
        Args:
            dynamics_func: Function computing dynamics residual
            weight: Constraint weight for penalty
        """
        self.dynamics_func = dynamics_func
        self.weight = weight
    
    def compute(self,
                state: np.ndarray,
                control: Optional[np.ndarray] = None,
                **kwargs) -> float:
        """Compute dynamics residual."""
        if control is None:
            return 0.0
        
        residual = self.dynamics_func(state, control)
        return self.weight * np.sum(residual ** 2)
    
    def compute_gradient(self,
                        state: np.ndarray,
                        control: Optional[np.ndarray] = None,
                        **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute gradients via finite differences."""
        eps = 1e-5
        
        state_grad = np.zeros(len(state))
        control_grad = np.zeros(len(control)) if control is not None else None
        
        base_val = self.compute(state, control)
        
        # State gradient
        for i in range(len(state)):
            state_plus = state.copy()
            state_plus[i] += eps
            val_plus = self.compute(state_plus, control)
            state_grad[i] = (val_plus - base_val) / eps
        
        # Control gradient
        if control is not None:
            for i in range(len(control)):
                control_plus = control.copy()
                control_plus[i] += eps
                val_plus = self.compute(state, control_plus)
                control_grad[i] = (val_plus - base_val) / eps
        
        return state_grad, control_grad


class ContactComplementarityConstraint(Constraint):
    """
    Contact complementarity constraint.
    
    Enforces:
    - λ >= 0 (contact force is non-negative)
    - d >= 0 (distance is non-negative)
    - λ * d = 0 (complementarity: contact or separation)
    
    This is non-convex and requires special handling in SCP.
    """
    
    def __init__(self,
                 weight: float = 1000.0,
                 relaxation: float = 0.01):
        """
        Initialize complementarity constraint.
        
        Args:
            weight: Constraint weight
            relaxation: Relaxation parameter for smooth approximation
        """
        self.weight = weight
        self.relaxation = relaxation
    
    def compute(self,
                distance: float,
                force: float,
                **kwargs) -> float:
        """
        Compute complementarity violation.
        
        Args:
            distance: Distance to contact (positive = separated)
            force: Normal contact force
            
        Returns:
            Constraint violation (should be <= 0)
        """
        # Non-negativity constraints
        force_violation = max(-force, 0)
        distance_violation = max(-distance, 0)
        
        # Complementarity: force * distance should be 0
        complementarity = force * distance
        
        return force_violation + distance_violation + complementarity
    
    def compute_gradient(self,
                        distance: float,
                        force: float,
                        **kwargs) -> Tuple[float, float]:
        """Compute gradients w.r.t. distance and force."""
        d_grad = 0.0
        f_grad = 0.0
        
        # Distance gradient
        if distance < 0:
            d_grad = -1.0
        d_grad += force  # From complementarity
        
        # Force gradient
        if force < 0:
            f_grad = -1.0
        f_grad += distance  # From complementarity
        
        return d_grad, f_grad
    
    def compute_smooth(self,
                       distance: float,
                       force: float) -> float:
        """
        Smooth approximation of complementarity.
        
        Uses Fischer-Burmeister function:
        φ(a, b) = sqrt(a² + b²) - a - b
        
        This is smooth and has φ(a, b) = 0 iff a >= 0, b >= 0, ab = 0
        """
        a = distance
        b = force
        
        phi = np.sqrt(a**2 + b**2 + self.relaxation**2) - a - b
        
        return self.weight * phi


class FrictionConeConstraint(Constraint):
    """
    Friction cone constraint.
    
    Enforces: |F_tangent| <= μ * F_normal
    
    This ensures contact forces stay within the friction cone.
    """
    
    def __init__(self,
                 friction_coef: float = 1.0,
                 weight: float = 100.0):
        """
        Initialize friction cone constraint.
        
        Args:
            friction_coef: Friction coefficient
            weight: Constraint weight
        """
        self.friction_coef = friction_coef
        self.weight = weight
    
    def compute(self,
                contact_force: np.ndarray,
                **kwargs) -> float:
        """
        Compute friction cone violation.
        
        Args:
            contact_force: Force in contact frame [normal, tangent1, tangent2]
            
        Returns:
            Constraint violation (positive = violated)
        """
        normal = contact_force[0]
        tangent = contact_force[1:3]
        
        # Normal force must be non-negative
        if normal < 0:
            return self.weight * (-normal) ** 2
        
        # Check friction cone
        tangent_magnitude = np.linalg.norm(tangent)
        max_friction = self.friction_coef * normal
        
        violation = max(tangent_magnitude - max_friction, 0)
        
        return self.weight * violation ** 2
    
    def compute_gradient(self,
                        contact_force: np.ndarray,
                        **kwargs) -> np.ndarray:
        """Compute gradient w.r.t. contact force."""
        gradient = np.zeros(3)
        
        normal = contact_force[0]
        tangent = contact_force[1:3]
        tangent_magnitude = np.linalg.norm(tangent)
        max_friction = self.friction_coef * normal
        
        if normal < 0:
            gradient[0] = -2 * self.weight * normal
        elif tangent_magnitude > max_friction and tangent_magnitude > 0:
            violation = tangent_magnitude - max_friction
            gradient[0] = -2 * self.weight * violation * self.friction_coef
            gradient[1:3] = 2 * self.weight * violation * tangent / tangent_magnitude
        
        return gradient
    
    def project_to_cone(self,
                       contact_force: np.ndarray) -> np.ndarray:
        """
        Project force onto friction cone.
        
        Args:
            contact_force: Force in contact frame
            
        Returns:
            Projected force
        """
        projected = contact_force.copy()
        
        normal = projected[0]
        tangent = projected[1:3]
        
        # Ensure non-negative normal
        if normal < 0:
            projected[:] = 0
            return projected
        
        # Project tangent onto cone
        tangent_magnitude = np.linalg.norm(tangent)
        max_friction = self.friction_coef * normal
        
        if tangent_magnitude > max_friction and tangent_magnitude > 0:
            projected[1:3] = tangent * max_friction / tangent_magnitude
        
        return projected


class JointLimitConstraint(Constraint):
    """
    Joint limit constraint.
    
    Enforces: q_lower <= q <= q_upper
    """
    
    def __init__(self,
                 lower_limits: np.ndarray,
                 upper_limits: np.ndarray,
                 weight: float = 1000.0,
                 margin: float = 0.0):
        """
        Initialize joint limit constraint.
        
        Args:
            lower_limits: Lower joint limits
            upper_limits: Upper joint limits
            weight: Constraint weight
            margin: Safety margin
        """
        self.lower_limits = np.asarray(lower_limits)
        self.upper_limits = np.asarray(upper_limits)
        self.weight = weight
        self.margin = margin
    
    def compute(self,
                joint_angles: np.ndarray,
                **kwargs) -> float:
        """Compute joint limit violation."""
        violation = 0.0
        
        # Lower limit violations
        lower_violation = self.lower_limits + self.margin - joint_angles
        violation += np.sum(np.maximum(lower_violation, 0) ** 2)
        
        # Upper limit violations
        upper_violation = joint_angles - (self.upper_limits - self.margin)
        violation += np.sum(np.maximum(upper_violation, 0) ** 2)
        
        return self.weight * violation
    
    def compute_gradient(self,
                        joint_angles: np.ndarray,
                        **kwargs) -> np.ndarray:
        """Compute gradient w.r.t. joint angles."""
        gradient = np.zeros_like(joint_angles)
        
        # Lower limit gradient
        lower_violation = self.lower_limits + self.margin - joint_angles
        lower_active = lower_violation > 0
        gradient[lower_active] = -2 * self.weight * lower_violation[lower_active]
        
        # Upper limit gradient
        upper_violation = joint_angles - (self.upper_limits - self.margin)
        upper_active = upper_violation > 0
        gradient[upper_active] += 2 * self.weight * upper_violation[upper_active]
        
        return gradient
    
    def clip_to_limits(self,
                       joint_angles: np.ndarray) -> np.ndarray:
        """Clip joint angles to limits."""
        return np.clip(
            joint_angles,
            self.lower_limits + self.margin,
            self.upper_limits - self.margin
        )


class TorqueLimitConstraint(Constraint):
    """
    Torque limit constraint.
    
    Enforces: |τ| <= τ_max
    """
    
    def __init__(self,
                 torque_limits: np.ndarray,
                 weight: float = 100.0):
        """
        Initialize torque limit constraint.
        
        Args:
            torque_limits: Maximum torque magnitudes
            weight: Constraint weight
        """
        self.torque_limits = np.asarray(torque_limits)
        self.weight = weight
    
    def compute(self,
                control: np.ndarray,
                **kwargs) -> float:
        """Compute torque limit violation."""
        violation = np.abs(control) - self.torque_limits
        return self.weight * np.sum(np.maximum(violation, 0) ** 2)
    
    def compute_gradient(self,
                        control: np.ndarray,
                        **kwargs) -> np.ndarray:
        """Compute gradient w.r.t. control."""
        gradient = np.zeros_like(control)
        
        violation = np.abs(control) - self.torque_limits
        active = violation > 0
        
        gradient[active] = 2 * self.weight * violation[active] * np.sign(control[active])
        
        return gradient
    
    def clip_to_limits(self,
                       control: np.ndarray) -> np.ndarray:
        """Clip control to limits."""
        return np.clip(control, -self.torque_limits, self.torque_limits)


class GroundConstraint(Constraint):
    """
    Ground penetration constraint.
    
    Enforces: z >= ground_height for stance feet
    """
    
    def __init__(self,
                 ground_height: float = 0.0,
                 weight: float = 1000.0):
        """
        Initialize ground constraint.
        
        Args:
            ground_height: Height of ground plane
            weight: Constraint weight
        """
        self.ground_height = ground_height
        self.weight = weight
    
    def compute(self,
                foot_height: float,
                in_contact: bool = True,
                **kwargs) -> float:
        """
        Compute ground constraint violation.
        
        Args:
            foot_height: Height of foot above ground
            in_contact: Whether foot should be in contact
            
        Returns:
            Constraint violation
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
                        in_contact: bool = True,
                        **kwargs) -> float:
        """Compute gradient w.r.t. foot height."""
        if foot_height < self.ground_height:
            return -2 * self.weight * (self.ground_height - foot_height)
        
        if in_contact and foot_height > 0.01:
            return 2 * self.weight * foot_height
        
        return 0.0


class ConstraintSet:
    """
    Container for multiple constraints.
    
    Provides unified interface for computing total constraint violation
    and gradients.
    """
    
    def __init__(self):
        """Initialize empty constraint set."""
        self.constraints: Dict[str, Constraint] = {}
        self.weights: Dict[str, float] = {}
    
    def add_constraint(self,
                      name: str,
                      constraint: Constraint,
                      weight: float = 1.0):
        """Add a constraint to the set."""
        self.constraints[name] = constraint
        self.weights[name] = weight
    
    def compute_total_violation(self,
                               state: np.ndarray,
                               control: Optional[np.ndarray] = None,
                               **kwargs) -> float:
        """Compute total constraint violation."""
        total = 0.0
        
        for name, constraint in self.constraints.items():
            violation = constraint.compute(state, control, **kwargs)
            total += self.weights[name] * violation
        
        return total
    
    def compute_violations(self,
                          state: np.ndarray,
                          control: Optional[np.ndarray] = None,
                          **kwargs) -> Dict[str, ConstraintViolation]:
        """Compute individual constraint violations."""
        violations = {}
        
        for name, constraint in self.constraints.items():
            value = constraint.compute(state, control, **kwargs)
            gradient = constraint.compute_gradient(state, control, **kwargs)
            
            violations[name] = ConstraintViolation(
                name=name,
                value=value,
                is_violated=value > 0,
                gradient=gradient[0]  # State gradient
            )
        
        return violations
    
    def compute_total_gradient(self,
                              state: np.ndarray,
                              control: Optional[np.ndarray] = None,
                              **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute total gradient."""
        state_grad = np.zeros(len(state))
        control_grad = np.zeros(len(control)) if control is not None else None
        
        for name, constraint in self.constraints.items():
            s_grad, c_grad = constraint.compute_gradient(state, control, **kwargs)
            state_grad += self.weights[name] * s_grad
            if c_grad is not None and control_grad is not None:
                control_grad += self.weights[name] * c_grad
        
        return state_grad, control_grad
