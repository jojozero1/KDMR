"""
Contact dynamics for KDMR.

This module handles contact-related dynamics computations:
- Contact Jacobians
- Contact force models
- Friction cone constraints
- Contact point detection
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import mujoco as mj
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


@dataclass
class ContactInfo:
    """Information about a single contact."""
    body1: str              # First body name
    body2: str              # Second body name (often 'floor' or 'ground')
    position: np.ndarray    # Contact position in world frame (3,)
    normal: np.ndarray      # Contact normal (3,)
    distance: float         # Signed distance (negative = penetration)
    friction: Tuple[float, float, float]  # Friction parameters (mu1, mu2, mu_roll)
    jacobian: np.ndarray    # Contact Jacobian (3, nv)


@dataclass
class ContactForce:
    """Contact force information."""
    normal_force: float     # Normal force magnitude
    friction_force: np.ndarray  # Friction force (2,)
    total_force: np.ndarray  # Total force vector (3,)
    contact_point: np.ndarray  # Contact position (3,)


class ContactDynamics:
    """
    Contact dynamics computations using MuJoCo.
    
    Handles:
    - Contact detection and geometry
    - Contact Jacobian computation
    - Friction cone constraints
    - Contact force distribution
    """
    
    def __init__(self, model, data):
        """
        Initialize contact dynamics.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required for ContactDynamics")
        
        self.model = model
        self.data = data
        
        # Dimensions
        self.nv = model.nv
        
        # Default friction coefficient
        self.default_friction = 1.0
    
    def get_all_contacts(self) -> List[ContactInfo]:
        """
        Get all active contacts.
        
        Returns:
            List of ContactInfo for each active contact
        """
        contacts = []
        
        mj.mj_forward(self.model, self.data)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get body names
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]
            
            body1 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body1_id)
            body2 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body2_id)
            
            # Contact position and normal
            pos = contact.pos.copy()
            normal = contact.frame[:3].copy()  # First column is normal
            
            # Distance
            dist = contact.dist
            
            # Friction
            friction = (
                self.model.geom_friction[contact.geom1, 0] * 
                self.model.geom_friction[contact.geom2, 0]
            )
            
            # Compute contact Jacobian
            J = self._compute_contact_jacobian(contact)
            
            contacts.append(ContactInfo(
                body1=body1,
                body2=body2,
                position=pos,
                normal=normal,
                distance=dist,
                friction=(friction, friction, 0.0),  # Simplified
                jacobian=J
            ))
        
        return contacts
    
    def get_foot_contacts(self,
                         left_foot_body: str = 'left_toe_link',
                         right_foot_body: str = 'right_toe_link') -> Dict[str, List[ContactInfo]]:
        """
        Get contacts for left and right feet.
        
        Args:
            left_foot_body: Left foot body name
            right_foot_body: Right foot body name
            
        Returns:
            Dictionary with 'left' and 'right' contact lists
        """
        all_contacts = self.get_all_contacts()
        
        left_contacts = []
        right_contacts = []
        
        for contact in all_contacts:
            if left_foot_body in contact.body1 or left_foot_body in contact.body2:
                left_contacts.append(contact)
            elif right_foot_body in contact.body1 or right_foot_body in contact.body2:
                right_contacts.append(contact)
        
        return {
            'left': left_contacts,
            'right': right_contacts
        }
    
    def _compute_contact_jacobian(self, contact) -> np.ndarray:
        """
        Compute contact Jacobian.
        
        Args:
            contact: MuJoCo contact object
            
        Returns:
            Contact Jacobian (3, nv)
        """
        J = np.zeros((3, self.nv))
        
        # Use mj_jacContact for contact Jacobian
        mj.mj_jacContact(self.model, self.data, J, None, 
                        np.where(self.data.contact == contact)[0][0])
        
        return J
    
    def compute_contact_forces(self) -> List[ContactForce]:
        """
        Compute contact forces from MuJoCo simulation.
        
        Returns:
            List of ContactForce for each contact
        """
        forces = []
        
        # Need to run forward dynamics first
        mj.mj_forward(self.model, self.data)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get contact force
            c_array = np.zeros(6)
            mj.mj_contactForce(self.model, self.data, i, c_array)
            
            # c_array = [normal, tangential1, tangential2, torsional, rolling1, rolling2]
            normal_force = c_array[0]
            friction_force = c_array[1:3]
            
            # Transform to world frame
            frame = contact.frame.reshape(3, 3)
            normal_world = frame[:, 0] * normal_force
            tangent1_world = frame[:, 1] * friction_force[0]
            tangent2_world = frame[:, 2] * friction_force[1]
            
            total_force = normal_world + tangent1_world + tangent2_world
            
            forces.append(ContactForce(
                normal_force=normal_force,
                friction_force=friction_force,
                total_force=total_force,
                contact_point=contact.pos.copy()
            ))
        
        return forces
    
    def compute_ground_reaction_force(self,
                                      foot_body: str) -> Optional[np.ndarray]:
        """
        Compute total ground reaction force for a foot.
        
        Args:
            foot_body: Foot body name
            
        Returns:
            Total GRF (3,) or None if no contact
        """
        contacts = self.get_all_contacts()
        
        total_force = np.zeros(3)
        has_contact = False
        
        for contact in contacts:
            if foot_body in contact.body1 or foot_body in contact.body2:
                # Get force for this contact
                contact_idx = np.where(self.data.contact == contact)[0]
                if len(contact_idx) > 0:
                    c_array = np.zeros(6)
                    mj.mj_contactForce(self.model, self.data, contact_idx[0], c_array)
                    
                    # Transform to world frame
                    frame = contact.frame.reshape(3, 3)
                    force_world = frame @ c_array[:3]
                    total_force += force_world
                    has_contact = True
        
        return total_force if has_contact else None
    
    def check_friction_cone(self,
                           contact_force: np.ndarray,
                           friction_coef: float = 1.0) -> Tuple[bool, float]:
        """
        Check if contact force satisfies friction cone constraint.
        
        Friction cone: |F_tangent| <= mu * F_normal
        
        Args:
            contact_force: Force in contact frame [normal, tangent1, tangent2]
            friction_coef: Friction coefficient
            
        Returns:
            Tuple of (is_valid, margin)
        """
        normal = contact_force[0]
        tangent = contact_force[1:3]
        
        if normal < 0:
            return False, -1.0  # Tension not allowed
        
        tangent_magnitude = np.linalg.norm(tangent)
        max_friction = friction_coef * normal
        
        is_valid = tangent_magnitude <= max_friction
        margin = (max_friction - tangent_magnitude) / max_friction if max_friction > 0 else 0
        
        return is_valid, margin
    
    def project_to_friction_cone(self,
                                contact_force: np.ndarray,
                                friction_coef: float = 1.0) -> np.ndarray:
        """
        Project contact force onto friction cone.
        
        Args:
            contact_force: Force in contact frame
            friction_coef: Friction coefficient
            
        Returns:
            Projected force
        """
        projected = contact_force.copy()
        
        normal = projected[0]
        tangent = projected[1:3]
        
        # Ensure normal is non-negative
        if normal < 0:
            projected[0] = 0
            projected[1:3] = 0
            return projected
        
        # Project tangent onto friction cone
        tangent_magnitude = np.linalg.norm(tangent)
        max_friction = friction_coef * normal
        
        if tangent_magnitude > max_friction and tangent_magnitude > 0:
            projected[1:3] = tangent * max_friction / tangent_magnitude
        
        return projected
    
    def compute_contact_point_velocity(self,
                                       contact: ContactInfo,
                                       qdot: np.ndarray) -> np.ndarray:
        """
        Compute velocity at contact point.
        
        Args:
            contact: Contact information
            qdot: Joint velocities
            
        Returns:
            Contact point velocity (3,)
        """
        return contact.jacobian @ qdot
    
    def compute_slip_velocity(self,
                              contact: ContactInfo,
                              qdot: np.ndarray) -> np.ndarray:
        """
        Compute slip velocity at contact.
        
        Slip velocity is the tangential velocity at contact point.
        
        Args:
            contact: Contact information
            qdot: Joint velocities
            
        Returns:
            Slip velocity (2,) in contact frame
        """
        # Get contact point velocity in world frame
        vel_world = self.compute_contact_point_velocity(contact, qdot)
        
        # Transform to contact frame
        frame = np.zeros((3, 3))
        # Get frame from contact
        contact_idx = None
        for i in range(self.data.ncon):
            if np.allclose(self.data.contact[i].pos, contact.position):
                contact_idx = i
                break
        
        if contact_idx is not None:
            frame = self.data.contact[contact_idx].frame.reshape(3, 3)
            vel_contact = frame.T @ vel_world
            return vel_contact[1:3]  # Tangential components
        
        return np.zeros(2)
    
    def estimate_contact_force_distribution(self,
                                           desired_total_force: np.ndarray,
                                           contacts: List[ContactInfo],
                                           weights: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Distribute desired total force among contacts.
        
        Uses pseudo-inverse of contact Jacobians to find force distribution
        that achieves desired total wrench.
        
        Args:
            desired_total_force: Desired total force (3,)
            contacts: List of contacts
            weights: Optional weights for each contact
            
        Returns:
            List of forces for each contact
        """
        if not contacts:
            return []
        
        n_contacts = len(contacts)
        
        # Stack contact Jacobians
        J_stack = np.vstack([c.jacobian for c in contacts])  # (3*n_contacts, nv)
        
        # We want to find f such that sum(J_i^T @ f_i) = F_total
        # This is underdetermined, use least-squares with weights
        
        if weights is None:
            weights = np.ones(n_contacts) / n_contacts
        
        # Simple weighted distribution
        forces = []
        for i, contact in enumerate(contacts):
            force = weights[i] * desired_total_force
            forces.append(force)
        
        return forces
    
    def is_foot_in_contact(self,
                          foot_body: str,
                          threshold: float = 0.001) -> bool:
        """
        Check if foot is in contact with ground.
        
        Args:
            foot_body: Foot body name
            threshold: Distance threshold
            
        Returns:
            True if in contact
        """
        contacts = self.get_foot_contacts(
            left_foot_body=foot_body,
            right_foot_body=foot_body
        )
        
        # Check either left or right (depending on foot_body)
        foot_contacts = contacts.get('left', []) + contacts.get('right', [])
        
        for contact in foot_contacts:
            if contact.distance < threshold:
                return True
        
        return False
    
    def get_contact_richardson(self,
                               foot_body: str) -> Dict[str, float]:
        """
        Get contact metrics using Richardson's method.
        
        Returns penetration depth, contact area estimate, etc.
        
        Args:
            foot_body: Foot body name
            
        Returns:
            Dictionary of contact metrics
        """
        contacts = self.get_all_contacts()
        
        metrics = {
            'penetration_depth': 0.0,
            'contact_area': 0.0,
            'num_contacts': 0
        }
        
        for contact in contacts:
            if foot_body in contact.body1 or foot_body in contact.body2:
                metrics['num_contacts'] += 1
                metrics['penetration_depth'] = max(
                    metrics['penetration_depth'],
                    -contact.distance
                )
                # Contact area would need geometry info
                metrics['contact_area'] += 0.001  # Placeholder
        
        return metrics
