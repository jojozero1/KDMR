"""
Trajectory Optimizer for KDMR.

This module provides the main trajectory optimization class that integrates:
- Kinematic retargeting (from GMR)
- Dynamics constraints
- Contact sequence handling
- SCP-DDP solver
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time

try:
    import mujoco as mj
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

from kdmr.core.scp_ddp_solver import SCPDDPSolver, SCPDDPConfig, SCPDDPResult
from kdmr.core.cost_functions import CostFunctions, CostConfig
from kdmr.contact.contact_mode import ContactMode, ContactSequence, DualContactSequence
from kdmr.contact.contact_estimator import ContactEstimator
from kdmr.contact.grf_processor import GRFProcessor, ProcessedGRF
from kdmr.utils.math_utils import MathUtils
from kdmr.utils.data_loader import HumanMotionData, RobotTrajectory


@dataclass
class OptimizationResult:
    """Result from trajectory optimization."""
    # Optimized trajectory
    trajectory: RobotTrajectory
    
    # Contact information
    contact_sequence: DualContactSequence
    
    # Optimization details
    cost_history: List[float]
    iterations: int
    converged: bool
    solve_time: float
    
    # Metrics
    dynamic_feasibility: float  # Dynamics equation residual
    constraint_violation: float  # Total constraint violation
    tracking_error: float       # Position tracking RMSE
    smoothness: float           # Trajectory smoothness metric


class TrajectoryOptimizer:
    """
    Main trajectory optimization class for KDMR.
    
    This class orchestrates the complete optimization pipeline:
    1. Load human motion and GRF data
    2. Estimate contact sequence from GRF
    3. Generate initial guess via kinematic retargeting
    4. Run SCP-DDP optimization with dynamics constraints
    5. Validate and post-process results
    """
    
    def __init__(self,
                 robot_xml_path: str,
                 config: Optional[Dict] = None):
        """
        Initialize trajectory optimizer.
        
        Args:
            robot_xml_path: Path to MuJoCo robot XML
            config: Configuration dictionary
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required for TrajectoryOptimizer")
        
        self.robot_xml_path = Path(robot_xml_path)
        self.config = config or {}
        
        # Load MuJoCo model
        self.model = mj.MjModel.from_xml_path(str(self.robot_xml_path))
        self.data = mj.MjData(self.model)
        
        # Problem dimensions
        self.nq = self.model.nq  # Generalized positions
        self.nv = self.model.nv  # Generalized velocities
        self.nu = self.model.nu  # Actuators
        
        # Initialize components
        self.grf_processor = GRFProcessor()
        self.contact_estimator = ContactEstimator()
        self.cost_functions = CostFunctions(
            CostConfig(**self.config.get('cost', {}))
        )
        self.solver = None
        
        # Contact body names
        self.left_foot_body = self.config.get('left_foot_body', 'left_toe_link')
        self.right_foot_body = self.config.get('right_foot_body', 'right_toe_link')
        
        # Solver configuration
        self.solver_config = SCPDDPConfig(
            **self.config.get('solver', {})
        )
    
    def optimize(self,
                 human_motion: HumanMotionData,
                 grf_data: Optional[ProcessedGRF] = None,
                 initial_trajectory: Optional[np.ndarray] = None,
                 dt: Optional[float] = None) -> OptimizationResult:
        """
        Run complete trajectory optimization.
        
        Args:
            human_motion: Human motion data
            grf_data: Ground reaction force data (optional)
            initial_trajectory: Initial guess (optional, uses kinematic retargeting)
            dt: Time step (default from motion fps)
            
        Returns:
            OptimizationResult with optimized trajectory
        """
        start_time = time.time()
        
        # Determine time step
        if dt is None:
            dt = 1.0 / human_motion.fps
        
        T = len(human_motion)
        
        # Step 1: Estimate contact sequence
        if grf_data is not None:
            contact_sequence = self.contact_estimator.estimate_dual_contact(grf_data)
        else:
            # Estimate from motion
            contact_sequence = self._estimate_contact_from_motion(human_motion)
        
        # Step 2: Generate initial trajectory
        if initial_trajectory is None:
            initial_trajectory = self._kinematic_retargeting(human_motion)
        
        # Step 3: Setup dynamics function
        dynamics_func = self._create_dynamics_function(dt)
        
        # Step 4: Create reference trajectory
        reference_trajectory = self._create_reference_trajectory(
            human_motion, initial_trajectory
        )
        
        # Step 5: Initialize and run solver
        self.solver = SCPDDPSolver(
            dynamics_func=dynamics_func,
            cost_functions=self.cost_functions,
            config=self.solver_config
        )
        
        result = self.solver.solve(
            initial_trajectory=initial_trajectory,
            reference_trajectory=reference_trajectory,
            dt=dt,
            contact_modes=contact_sequence
        )
        
        # Step 6: Create output trajectory
        trajectory = RobotTrajectory(
            qpos=result.trajectory,
            qvel=self._compute_velocities(result.trajectory, dt),
            tau=result.controls,
            contact_forces=None,  # Would need forward dynamics
            fps=human_motion.fps
        )
        
        # Step 7: Compute metrics
        metrics = self._compute_metrics(
            trajectory, 
            reference_trajectory,
            contact_sequence,
            dt
        )
        
        solve_time = time.time() - start_time
        
        return OptimizationResult(
            trajectory=trajectory,
            contact_sequence=contact_sequence,
            cost_history=result.cost_history,
            iterations=result.scp_iterations,
            converged=result.converged,
            solve_time=solve_time,
            **metrics
        )
    
    def _estimate_contact_from_motion(self,
                                      human_motion: HumanMotionData) -> DualContactSequence:
        """Estimate contact sequence from motion data alone."""
        # Extract foot positions (assuming standard joint names)
        left_foot_idx = self._find_joint_index(human_motion, 'left_foot')
        right_foot_idx = self._find_joint_index(human_motion, 'right_foot')
        
        if left_foot_idx is None or right_foot_idx is None:
            # Create default contact sequence
            return self._create_default_contact_sequence(len(human_motion))
        
        left_positions = human_motion.positions[:, left_foot_idx, :]
        right_positions = human_motion.positions[:, right_foot_idx, :]
        
        left_seq = self.contact_estimator.estimate_from_motion_only(
            left_positions, human_motion.fps
        )
        left_seq.foot_name = 'left'
        
        right_seq = self.contact_estimator.estimate_from_motion_only(
            right_positions, human_motion.fps
        )
        right_seq.foot_name = 'right'
        
        return DualContactSequence(left=left_seq, right=right_seq)
    
    def _find_joint_index(self,
                          human_motion: HumanMotionData,
                          joint_name: str) -> Optional[int]:
        """Find joint index by name."""
        for i, name in enumerate(human_motion.joint_names):
            if joint_name.lower() in name.lower():
                return i
        return None
    
    def _create_default_contact_sequence(self, n_frames: int) -> DualContactSequence:
        """Create default contact sequence (all flat contact)."""
        left_modes = [ContactMode.FLAT] * n_frames
        right_modes = [ContactMode.FLAT] * n_frames
        
        return DualContactSequence(
            left=ContactSequence(
                foot_name='left',
                modes=left_modes,
                timestamps=np.arange(n_frames) / 30.0
            ),
            right=ContactSequence(
                foot_name='right',
                modes=right_modes,
                timestamps=np.arange(n_frames) / 30.0
            )
        )
    
    def _kinematic_retargeting(self,
                               human_motion: HumanMotionData) -> np.ndarray:
        """
        Generate initial trajectory via kinematic retargeting.
        
        This uses a simplified IK approach to map human motion to robot.
        For full functionality, should integrate with GMR.
        """
        T = len(human_motion)
        trajectory = np.zeros((T, self.nq))
        
        # For each frame, compute robot configuration
        for t in range(T):
            frame = human_motion.get_frame(t)
            qpos = self._solve_ik_frame(frame)
            trajectory[t] = qpos
        
        return trajectory
    
    def _solve_ik_frame(self,
                        human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Solve IK for a single frame.
        
        Simplified version - full implementation should use GMR.
        """
        qpos = np.zeros(self.nq)
        
        # Set root position from pelvis
        if 'pelvis' in human_frame:
            pos, quat = human_frame['pelvis']
            qpos[:3] = pos
            qpos[3:7] = quat
        
        # Joint angles would be computed via proper IK
        # This is a placeholder
        return qpos
    
    def _create_dynamics_function(self, dt: float) -> callable:
        """
        Create dynamics function for trajectory optimization.
        
        Returns function f(x, u) -> x_next
        """
        def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            """
            Compute next state using MuJoCo dynamics.
            
            Args:
                x: State (position + velocity)
                u: Control (joint torques)
                
            Returns:
                Next state
            """
            # Set state
            self.data.qpos[:7] = x[:7]  # Root pos + quat
            self.data.qpos[7:] = x[7:self.nq]  # Joint angles
            
            if len(x) > self.nq:
                self.data.qvel[:] = x[self.nq:self.nq + self.nv]
            else:
                self.data.qvel[:] = 0
            
            # Set control
            self.data.ctrl[:len(u)] = u
            
            # Step simulation
            mj.mj_step(self.model, self.data)
            
            # Extract next state
            x_next = np.zeros(len(x))
            x_next[:self.nq] = self.data.qpos[:self.nq]
            x_next[self.nq:] = self.data.qvel[:]
            
            return x_next
        
        return dynamics
    
    def _create_reference_trajectory(self,
                                    human_motion: HumanMotionData,
                                    initial_traj: np.ndarray) -> np.ndarray:
        """Create reference trajectory for tracking cost."""
        # Use initial trajectory as reference
        # Could also blend with human motion
        return initial_traj.copy()
    
    def _compute_velocities(self,
                           trajectory: np.ndarray,
                           dt: float) -> np.ndarray:
        """Compute velocities from position trajectory."""
        qvel = np.zeros((len(trajectory), self.nv))
        
        for i in range(1, len(trajectory)):
            # Root velocity
            qvel[i, :3] = (trajectory[i, :3] - trajectory[i-1, :3]) / dt
            
            # Joint velocities
            qvel[i, 6:] = (trajectory[i, 7:] - trajectory[i-1, 7:]) / dt
        
        return qvel
    
    def _compute_metrics(self,
                        trajectory: RobotTrajectory,
                        reference: np.ndarray,
                        contact_sequence: DualContactSequence,
                        dt: float) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Dynamic feasibility (dynamics equation residual)
        dynamic_feasibility = self._compute_dynamic_feasibility(trajectory, dt)
        
        # Constraint violation
        constraint_violation = self._compute_constraint_violation(
            trajectory, contact_sequence
        )
        
        # Tracking error
        tracking_error = np.sqrt(np.mean((trajectory.qpos - reference) ** 2))
        
        # Smoothness (jerk)
        jerk = MathUtils.compute_jerk(trajectory.qpos, dt)
        smoothness = np.mean(np.sum(jerk ** 2, axis=1))
        
        return {
            'dynamic_feasibility': dynamic_feasibility,
            'constraint_violation': constraint_violation,
            'tracking_error': tracking_error,
            'smoothness': smoothness
        }
    
    def _compute_dynamic_feasibility(self,
                                    trajectory: RobotTrajectory,
                                    dt: float) -> float:
        """
        Compute dynamics equation residual.
        
        Checks if M*q̈ + C + G = τ + J^T * F
        """
        residuals = []
        
        for t in range(1, len(trajectory) - 1):
            # Set state
            self.data.qpos[:] = trajectory.qpos[t]
            if trajectory.qvel is not None:
                self.data.qvel[:] = trajectory.qvel[t]
            
            # Compute dynamics
            mj.mj_forward(self.model, self.data)
            
            # Compute acceleration
            if trajectory.qvel is not None:
                qddot = (trajectory.qvel[t+1] - trajectory.qvel[t-1]) / (2 * dt)
            else:
                qddot = np.zeros(self.nv)
            
            # Compute inverse dynamics
            desired_qacc = qddot
            mj.mj_inverse(self.model, self.data)
            
            # Residual
            if trajectory.tau is not None and t-1 < len(trajectory.tau):
                residual = np.linalg.norm(self.data.qfrc_inverse[:self.nu] - trajectory.tau[t-1])
            else:
                residual = np.linalg.norm(self.data.qfrc_inverse)
            
            residuals.append(residual)
        
        return np.mean(residuals) if residuals else 0.0
    
    def _compute_constraint_violation(self,
                                     trajectory: RobotTrajectory,
                                     contact_sequence: DualContactSequence) -> float:
        """Compute total constraint violation."""
        violation = 0.0
        
        for t in range(len(trajectory)):
            # Set state
            self.data.qpos[:] = trajectory.qpos[t]
            mj.mj_forward(self.model, self.data)
            
            # Check foot constraints
            left_mode, right_mode = contact_sequence.get_phase_at_index(t)
            
            # Left foot
            left_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.left_foot_body)
            if left_id >= 0:
                left_pos = self.data.xpos[left_id]
                left_height = left_pos[2]
                
                if left_mode.is_stance() and left_height > 0.01:
                    violation += left_height ** 2
                elif left_mode.is_swing() and left_height < 0:
                    violation += left_height ** 2
            
            # Right foot
            right_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.right_foot_body)
            if right_id >= 0:
                right_pos = self.data.xpos[right_id]
                right_height = right_pos[2]
                
                if right_mode.is_stance() and right_height > 0.01:
                    violation += right_height ** 2
                elif right_mode.is_swing() and right_height < 0:
                    violation += right_height ** 2
        
        return violation
    
    def set_cost_weights(self, **kwargs):
        """Update cost function weights."""
        for key, value in kwargs.items():
            if hasattr(self.cost_functions.config, key):
                setattr(self.cost_functions.config, key, value)
    
    def get_robot_info(self) -> Dict[str, Any]:
        """Get robot model information."""
        return {
            'nq': self.nq,
            'nv': self.nv,
            'nu': self.nu,
            'body_names': [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i) 
                          for i in range(self.model.nbody)],
            'joint_names': [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i)
                           for i in range(self.model.njnt)],
            'actuator_names': [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
                              for i in range(self.model.nu)]
        }
