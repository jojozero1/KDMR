"""
Rigid body dynamics computations for KDMR.

This module provides dynamics computations using MuJoCo:
- Mass matrix M(q)
- Coriolis and centrifugal forces C(q, q̇)
- Gravity forces G(q)
- Forward and inverse dynamics
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
class DynamicsState:
    """Container for dynamics-related quantities."""
    q: np.ndarray          # Joint positions
    qdot: np.ndarray       # Joint velocities
    qddot: np.ndarray      # Joint accelerations
    M: np.ndarray          # Mass matrix
    C: np.ndarray          # Coriolis forces
    G: np.ndarray          # Gravity forces
    tau: np.ndarray        # Joint torques


class RigidBodyDynamics:
    """
    Rigid body dynamics computations using MuJoCo.
    
    Provides efficient computation of:
    - Mass matrix: M(q)
    - Coriolis/centrifugal: C(q, q̇)
    - Gravity: G(q)
    - Forward dynamics: q̈ = f(q, q̇, τ, F_ext)
    - Inverse dynamics: τ = f(q, q̇, q̈, F_ext)
    """
    
    def __init__(self, model, data):
        """
        Initialize dynamics from MuJoCo model.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required for RigidBodyDynamics")
        
        self.model = model
        self.data = data
        
        # Dimensions
        self.nq = model.nq  # Position dimension
        self.nv = model.nv  # Velocity dimension
        self.nu = model.nu  # Actuator dimension
    
    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute mass matrix M(q).
        
        The mass matrix relates joint accelerations to forces:
        M(q) * q̈ = τ - C(q, q̇) - G(q) + J^T * F_ext
        
        Args:
            q: Joint positions (nq,)
            
        Returns:
            Mass matrix (nv, nv)
        """
        # Set configuration
        self.data.qpos[:len(q)] = q
        
        # Compute mass matrix
        mj.mj_forward(self.model, self.data)
        
        M = np.zeros((self.nv, self.nv))
        mj.mj_fullM(self.model, M, self.data.qM)
        
        return M
    
    def compute_mass_matrix_inverse(self, q: np.ndarray) -> np.ndarray:
        """
        Compute inverse of mass matrix M(q)^{-1}.
        
        Args:
            q: Joint positions
            
        Returns:
            Inverse mass matrix (nv, nv)
        """
        M = self.compute_mass_matrix(q)
        
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)
    
    def compute_coriolis_gravity(self, 
                                 q: np.ndarray, 
                                 qdot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Coriolis/centrifugal and gravity forces.
        
        Args:
            q: Joint positions (nq,)
            qdot: Joint velocities (nv,)
            
        Returns:
            Tuple of (C, G) where:
            - C: Coriolis/centrifugal forces (nv,)
            - G: Gravity forces (nv,)
        """
        # Set state
        self.data.qpos[:len(q)] = q
        self.data.qvel[:len(qdot)] = qdot
        
        # Forward to compute qfrc_bias
        mj.mj_forward(self.model, self.data)
        
        # qfrc_bias = C + G (combined)
        bias = self.data.qfrc_bias.copy()
        
        # To separate C and G, compute G separately with zero velocity
        self.data.qpos[:len(q)] = q
        self.data.qvel[:] = 0
        mj.mj_forward(self.model, self.data)
        
        G = self.data.qfrc_bias.copy()
        C = bias - G
        
        return C, G
    
    def forward_dynamics(self,
                        q: np.ndarray,
                        qdot: np.ndarray,
                        tau: np.ndarray,
                        f_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute forward dynamics: q̈ = f(q, q̇, τ, F_ext)
        
        Given positions, velocities, and applied torques/forces,
        compute accelerations.
        
        Args:
            q: Joint positions (nq,)
            qdot: Joint velocities (nv,)
            tau: Applied torques (nu,)
            f_ext: External forces (nv,) optional
            
        Returns:
            Joint accelerations (nv,)
        """
        # Set state
        self.data.qpos[:len(q)] = q
        self.data.qvel[:len(qdot)] = qdot
        self.data.ctrl[:len(tau)] = tau
        
        # Set external forces if provided
        if f_ext is not None:
            self.data.qfrc_applied[:len(f_ext)] = f_ext
        
        # Compute forward dynamics
        mj.mj_forward(self.model, self.data)
        
        return self.data.qacc.copy()
    
    def inverse_dynamics(self,
                        q: np.ndarray,
                        qdot: np.ndarray,
                        qddot: np.ndarray,
                        f_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute inverse dynamics: τ = f(q, q̇, q̈, F_ext)
        
        Given positions, velocities, and desired accelerations,
        compute required torques.
        
        Args:
            q: Joint positions (nq,)
            qdot: Joint velocities (nv,)
            qddot: Desired accelerations (nv,)
            f_ext: External forces (nv,) optional
            
        Returns:
            Required torques (nv,)
        """
        # Set state
        self.data.qpos[:len(q)] = q
        self.data.qvel[:len(qdot)] = qdot
        self.data.qacc[:len(qddot)] = qddot
        
        # Set external forces if provided
        if f_ext is not None:
            self.data.qfrc_applied[:len(f_ext)] = f_ext
        
        # Compute inverse dynamics
        mj.mj_inverse(self.model, self.data)
        
        return self.data.qfrc_inverse.copy()
    
    def compute_com_position(self, q: np.ndarray) -> np.ndarray:
        """
        Compute center of mass position.
        
        Args:
            q: Joint positions
            
        Returns:
            CoM position in world frame (3,)
        """
        self.data.qpos[:len(q)] = q
        mj.mj_forward(self.model, self.data)
        
        # CoM is computed during forward
        # Use subtree_com for root body
        return self.data.subtree_com[1, :].copy()  # Root subtree
    
    def compute_com_velocity(self, 
                            q: np.ndarray, 
                            qdot: np.ndarray) -> np.ndarray:
        """
        Compute center of mass velocity.
        
        Args:
            q: Joint positions
            qdot: Joint velocities
            
        Returns:
            CoM velocity in world frame (3,)
        """
        self.data.qpos[:len(q)] = q
        self.data.qvel[:len(qdot)] = qdot
        mj.mj_forward(self.model, self.data)
        
        # Compute CoM velocity from qdot and Jacobian
        J_com = self.compute_com_jacobian(q)
        return J_com @ qdot
    
    def compute_com_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian for center of mass.
        
        Args:
            q: Joint positions
            
        Returns:
            CoM Jacobian (3, nv)
        """
        self.data.qpos[:len(q)] = q
        mj.mj_forward(self.model, self.data)
        
        J_com = np.zeros((3, self.nv))
        mj.mj_jacCom(self.model, self.data, J_com, None)
        
        return J_com
    
    def compute_body_jacobian(self, 
                             q: np.ndarray,
                             body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian for a specific body.
        
        Args:
            q: Joint positions
            body_name: Name of body
            
        Returns:
            Tuple of (position Jacobian, rotation Jacobian)
            - J_pos: (3, nv)
            - J_rot: (3, nv)
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found")
        
        self.data.qpos[:len(q)] = q
        mj.mj_forward(self.model, self.data)
        
        J_pos = np.zeros((3, self.nv))
        J_rot = np.zeros((3, self.nv))
        mj.mj_jacBody(self.model, self.data, J_pos, J_rot, body_id)
        
        return J_pos, J_rot
    
    def compute_site_jacobian(self,
                             q: np.ndarray,
                             site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian for a specific site.
        
        Args:
            q: Joint positions
            site_name: Name of site
            
        Returns:
            Tuple of (position Jacobian, rotation Jacobian)
        """
        site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site '{site_name}' not found")
        
        self.data.qpos[:len(q)] = q
        mj.mj_forward(self.model, self.data)
        
        J_pos = np.zeros((3, self.nv))
        J_rot = np.zeros((3, self.nv))
        mj.mj_jacSite(self.model, self.data, J_pos, J_rot, site_id)
        
        return J_pos, J_rot
    
    def compute_momentum(self,
                        q: np.ndarray,
                        qdot: np.ndarray) -> np.ndarray:
        """
        Compute generalized momentum.
        
        p = M(q) * qdot
        
        Args:
            q: Joint positions
            qdot: Joint velocities
            
        Returns:
            Generalized momentum (nv,)
        """
        M = self.compute_mass_matrix(q)
        return M @ qdot
    
    def compute_kinetic_energy(self,
                              q: np.ndarray,
                              qdot: np.ndarray) -> float:
        """
        Compute kinetic energy.
        
        KE = 0.5 * qdot^T * M(q) * qdot
        
        Args:
            q: Joint positions
            qdot: Joint velocities
            
        Returns:
            Kinetic energy
        """
        M = self.compute_mass_matrix(q)
        return 0.5 * qdot @ M @ qdot
    
    def compute_potential_energy(self, q: np.ndarray) -> float:
        """
        Compute potential energy (gravitational).
        
        Args:
            q: Joint positions
            
        Returns:
            Potential energy
        """
        self.data.qpos[:len(q)] = q
        mj.mj_forward(self.model, self.data)
        
        return self.data.energy[1]  # Potential energy
    
    def compute_total_energy(self,
                            q: np.ndarray,
                            qdot: np.ndarray) -> float:
        """
        Compute total mechanical energy.
        
        Args:
            q: Joint positions
            qdot: Joint velocities
            
        Returns:
            Total energy (KE + PE)
        """
        self.data.qpos[:len(q)] = q
        self.data.qvel[:len(qdot)] = qdot
        mj.mj_forward(self.model, self.data)
        
        return self.data.energy[0] + self.data.energy[1]
    
    def integrate_velocity(self,
                          q: np.ndarray,
                          qdot: np.ndarray,
                          dt: float) -> np.ndarray:
        """
        Integrate velocity to get new position.
        
        Uses MuJoCo's Euler integrator for quaternion handling.
        
        Args:
            q: Current positions
            qdot: Current velocities
            dt: Time step
            
        Returns:
            New positions
        """
        # Create temporary data
        self.data.qpos[:len(q)] = q
        self.data.qvel[:len(qdot)] = qdot
        
        # Integrate
        mj.mj_integratePos(self.model, self.data.qpos, self.data.qvel, dt)
        
        return self.data.qpos[:len(q)].copy()
    
    def get_state(self) -> DynamicsState:
        """Get current dynamics state."""
        return DynamicsState(
            q=self.data.qpos.copy(),
            qdot=self.data.qvel.copy(),
            qddot=self.data.qacc.copy(),
            M=self.compute_mass_matrix(self.data.qpos),
            C=self.data.qfrc_bias.copy(),
            G=np.zeros(self.nv),  # Would need separate computation
            tau=self.data.ctrl.copy()
        )
