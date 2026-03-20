"""
KDMR - Kinodynamic Motion Retargeting main module.

This module provides the main KDMR class that integrates all components
for dynamics-constrained motion retargeting.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time

from kdmr.core.trajectory_optimizer import TrajectoryOptimizer, OptimizationResult
from kdmr.retargeting.kinematic_retarget import KinematicRetarget
from kdmr.contact.contact_estimator import ContactEstimator
from kdmr.contact.grf_processor import GRFProcessor, ProcessedGRF
from kdmr.contact.contact_mode import DualContactSequence
from kdmr.utils.data_loader import (
    HumanMotionData, 
    GRFData, 
    RobotTrajectory,
    DataLoader
)


@dataclass
class KDMRConfig:
    """Configuration for KDMR."""
    # Cost weights
    tracking_weight: float = 100.0
    smoothness_weight: float = 0.1
    control_weight: float = 0.01
    contact_weight: float = 1.0
    
    # Solver settings
    max_scp_iterations: int = 10
    max_ddp_iterations: int = 50
    convergence_threshold: float = 1e-4
    
    # Contact settings
    force_threshold: float = 20.0
    use_grf_data: bool = True
    
    # Verbosity
    verbose: bool = True


@dataclass
class KDMRResult:
    """Result from KDMR retargeting."""
    # Optimized trajectory
    trajectory: RobotTrajectory
    
    # Contact sequence
    contact_sequence: DualContactSequence
    
    # Optimization details
    optimization_result: OptimizationResult
    
    # Metrics
    metrics: Dict[str, float]
    
    # Timing
    total_time: float


class KDMR:
    """
    Kinodynamic Motion Retargeting (KDMR) main class.
    
    This class provides the complete KDMR pipeline:
    1. Load human motion and GRF data
    2. Estimate contact sequence from GRF
    3. Generate initial guess via kinematic retargeting
    4. Run dynamics-constrained trajectory optimization
    5. Validate and output results
    
    Compared to pure kinematic methods (GMR), KDMR:
    - Enforces dynamics constraints
    - Uses GRF data for contact detection
    - Produces dynamically feasible trajectories
    """
    
    def __init__(self,
                 robot_xml_path: str,
                 config: Optional[KDMRConfig] = None,
                 ik_config_path: Optional[str] = None):
        """
        Initialize KDMR.
        
        Args:
            robot_xml_path: Path to MuJoCo robot XML
            config: KDMR configuration
            ik_config_path: Path to IK configuration (GMR format)
        """
        self.robot_xml_path = Path(robot_xml_path)
        self.config = config or KDMRConfig()
        
        # Initialize components
        self.kinematic_retarget = KinematicRetarget(robot_xml_path)
        
        # Load IK config if provided
        if ik_config_path is not None:
            self.kinematic_retarget.load_mapping_from_config(ik_config_path)
        
        self.grf_processor = GRFProcessor()
        self.contact_estimator = ContactEstimator()
        
        # Trajectory optimizer (initialized when needed)
        self.trajectory_optimizer = None
        
        # Results storage
        self.last_result: Optional[KDMRResult] = None
    
    def retarget(self,
                 human_motion: HumanMotionData,
                 grf_data: Optional[GRFData] = None,
                 initial_trajectory: Optional[np.ndarray] = None) -> KDMRResult:
        """
        Perform kinodynamic motion retargeting.
        
        Args:
            human_motion: Human motion data
            grf_data: Ground reaction force data (optional)
            initial_trajectory: Initial guess (optional)
            
        Returns:
            KDMRResult with optimized trajectory
        """
        start_time = time.time()
        
        if self.config.verbose:
            print(f"[KDMR] Starting retargeting for {len(human_motion)} frames")
        
        # Step 1: Process GRF data if available
        processed_grf = None
        if grf_data is not None and self.config.use_grf_data:
            processed_grf = self.grf_processor.process(
                forces=grf_data.forces,
                timestamps=grf_data.timestamps,
                fps=grf_data.fps,
                body_weight=None  # Will be estimated
            )
            if self.config.verbose:
                print(f"[KDMR] Processed GRF data: {len(processed_grf.timestamps)} samples")
        
        # Step 2: Estimate contact sequence
        if processed_grf is not None:
            contact_sequence = self.contact_estimator.estimate_dual_contact(processed_grf)
        else:
            # Estimate from motion
            contact_sequence = self._estimate_contact_from_motion(human_motion)
        
        if self.config.verbose:
            left_stance = sum(1 for m in contact_sequence.left.modes if m.is_stance())
            right_stance = sum(1 for m in contact_sequence.right.modes if m.is_stance())
            print(f"[KDMR] Contact sequence: left stance={left_stance}, right stance={right_stance}")
        
        # Step 3: Generate initial trajectory via kinematic retargeting
        if initial_trajectory is None:
            if self.config.verbose:
                print("[KDMR] Running kinematic retargeting for initial guess...")
            initial_trajectory = self.kinematic_retarget.retarget_trajectory(human_motion)
        
        # Step 4: Setup trajectory optimizer
        if self.trajectory_optimizer is None:
            optimizer_config = {
                'cost': {
                    'tracking_pos': self.config.tracking_weight,
                    'smoothness': self.config.smoothness_weight,
                    'control_effort': self.config.control_weight,
                    'contact_force': self.config.contact_weight,
                },
                'solver': {
                    'max_scp_iterations': self.config.max_scp_iterations,
                    'max_ddp_iterations': self.config.max_ddp_iterations,
                    'scp_convergence_threshold': self.config.convergence_threshold,
                    'verbose': self.config.verbose,
                }
            }
            self.trajectory_optimizer = TrajectoryOptimizer(
                str(self.robot_xml_path),
                optimizer_config
            )
        
        # Step 5: Run trajectory optimization
        if self.config.verbose:
            print("[KDMR] Running dynamics-constrained optimization...")
        
        opt_result = self.trajectory_optimizer.optimize(
            human_motion=human_motion,
            grf_data=processed_grf,
            initial_trajectory=initial_trajectory,
            dt=1.0 / human_motion.fps
        )
        
        # Step 6: Compute metrics
        metrics = self._compute_metrics(
            opt_result.trajectory,
            initial_trajectory,
            contact_sequence,
            human_motion.fps
        )
        
        total_time = time.time() - start_time
        
        if self.config.verbose:
            print(f"[KDMR] Retargeting complete in {total_time:.2f}s")
            print(f"[KDMR] Metrics: tracking_error={metrics['tracking_error']:.4f}, "
                  f"dynamic_feasibility={metrics['dynamic_feasibility']:.4f}")
        
        result = KDMRResult(
            trajectory=opt_result.trajectory,
            contact_sequence=contact_sequence,
            optimization_result=opt_result,
            metrics=metrics,
            total_time=total_time
        )
        
        self.last_result = result
        return result
    
    def retarget_with_gmr_integration(self,
                                      human_motion: HumanMotionData,
                                      gmr_retarget_func: callable,
                                      grf_data: Optional[GRFData] = None) -> KDMRResult:
        """
        Retarget using GMR for initial kinematic retargeting.
        
        This method integrates with the existing GMR pipeline.
        
        Args:
            human_motion: Human motion data
            gmr_retarget_func: GMR retargeting function
            grf_data: GRF data
            
        Returns:
            KDMRResult
        """
        # Use GMR for initial retargeting
        initial_trajectory = []
        
        for t in range(len(human_motion)):
            frame = human_motion.get_frame(t)
            qpos = gmr_retarget_func(frame)
            initial_trajectory.append(qpos)
        
        initial_trajectory = np.array(initial_trajectory)
        
        # Run KDMR optimization
        return self.retarget(human_motion, grf_data, initial_trajectory)
    
    def _estimate_contact_from_motion(self,
                                     human_motion: HumanMotionData) -> DualContactSequence:
        """Estimate contact sequence from motion data alone."""
        # Find foot joint indices
        left_foot_idx = None
        right_foot_idx = None
        
        for i, name in enumerate(human_motion.joint_names):
            name_lower = name.lower()
            if 'left' in name_lower and 'foot' in name_lower:
                left_foot_idx = i
            elif 'right' in name_lower and 'foot' in name_lower:
                right_foot_idx = i
        
        # Estimate contact from foot height
        n_frames = len(human_motion)
        
        left_modes = []
        right_modes = []
        
        height_threshold = 0.05  # 5 cm
        velocity_threshold = 0.3  # m/s
        
        for t in range(n_frames):
            # Left foot
            if left_foot_idx is not None:
                left_pos = human_motion.positions[t, left_foot_idx]
                left_height = left_pos[2]
                
                # Compute velocity
                if t > 0:
                    left_vel = np.linalg.norm(
                        human_motion.positions[t, left_foot_idx] - 
                        human_motion.positions[t-1, left_foot_idx]
                    ) * human_motion.fps
                else:
                    left_vel = 0
                
                if left_height < height_threshold and left_vel < velocity_threshold:
                    left_modes.append(self._estimate_contact_phase(left_height, left_vel))
                else:
                    left_modes.append(self.contact_estimator.contact_mode.ContactMode.SWING)
            else:
                left_modes.append(self.contact_estimator.contact_mode.ContactMode.FLAT)
            
            # Right foot
            if right_foot_idx is not None:
                right_pos = human_motion.positions[t, right_foot_idx]
                right_height = right_pos[2]
                
                if t > 0:
                    right_vel = np.linalg.norm(
                        human_motion.positions[t, right_foot_idx] - 
                        human_motion.positions[t-1, right_foot_idx]
                    ) * human_motion.fps
                else:
                    right_vel = 0
                
                if right_height < height_threshold and right_vel < velocity_threshold:
                    right_modes.append(self._estimate_contact_phase(right_height, right_vel))
                else:
                    right_modes.append(self.contact_estimator.contact_mode.ContactMode.SWING)
            else:
                right_modes.append(self.contact_estimator.contact_mode.ContactMode.FLAT)
        
        from kdmr.contact.contact_mode import ContactSequence, ContactMode
        
        return DualContactSequence(
            left=ContactSequence(
                foot_name='left',
                modes=left_modes,
                timestamps=human_motion.timestamps if hasattr(human_motion, 'timestamps') 
                          else np.arange(n_frames) / human_motion.fps
            ),
            right=ContactSequence(
                foot_name='right',
                modes=right_modes,
                timestamps=human_motion.timestamps if hasattr(human_motion, 'timestamps')
                          else np.arange(n_frames) / human_motion.fps
            )
        )
    
    def _estimate_contact_phase(self, height: float, velocity: float):
        """Estimate contact phase from height and velocity."""
        from kdmr.contact.contact_mode import ContactMode
        
        # Simple heuristic
        if height < 0.01:
            return ContactMode.FLAT
        elif velocity < 0.1:
            return ContactMode.FLAT
        else:
            return ContactMode.HEEL  # Or TOE depending on velocity direction
    
    def _compute_metrics(self,
                        trajectory: RobotTrajectory,
                        reference: np.ndarray,
                        contact_sequence: DualContactSequence,
                        fps: float) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from kdmr.utils.math_utils import MathUtils
        
        # Tracking error
        tracking_error = np.sqrt(np.mean((trajectory.qpos - reference) ** 2))
        
        # Smoothness (jerk)
        jerk = MathUtils.compute_jerk(trajectory.qpos, 1.0 / fps)
        smoothness = np.mean(np.sum(jerk ** 2, axis=1))
        
        # Contact consistency
        stance_frames = sum(
            1 for i in range(len(contact_sequence))
            if contact_sequence.left.modes[i].is_stance() or 
               contact_sequence.right.modes[i].is_stance()
        )
        contact_ratio = stance_frames / len(contact_sequence)
        
        return {
            'tracking_error': tracking_error,
            'smoothness': smoothness,
            'contact_ratio': contact_ratio,
            'dynamic_feasibility': 0.0,  # Would need forward dynamics
            'constraint_violation': 0.0,  # Would need constraint checking
        }
    
    def save_result(self, output_path: str):
        """Save last result to file."""
        if self.last_result is None:
            raise ValueError("No result to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save trajectory
        np.savez(
            output_path.with_suffix('.npz'),
            qpos=self.last_result.trajectory.qpos,
            qvel=self.last_result.trajectory.qvel,
            tau=self.last_result.trajectory.tau,
            fps=self.last_result.trajectory.fps,
            metrics=self.last_result.metrics
        )
    
    def compare_with_gmr(self,
                        gmr_trajectory: np.ndarray,
                        human_motion: HumanMotionData) -> Dict[str, Dict[str, float]]:
        """
        Compare KDMR result with GMR baseline.
        
        Args:
            gmr_trajectory: GMR retargeted trajectory
            human_motion: Original human motion
            
        Returns:
            Dictionary comparing metrics
        """
        if self.last_result is None:
            raise ValueError("No KDMR result available. Run retarget() first.")
        
        kdmr_traj = self.last_result.trajectory.qpos
        
        # Compute metrics for both
        fps = human_motion.fps
        
        from kdmr.utils.math_utils import MathUtils
        
        # Smoothness
        gmr_jerk = MathUtils.compute_jerk(gmr_trajectory, 1.0 / fps)
        kdmr_jerk = MathUtils.compute_jerk(kdmr_traj, 1.0 / fps)
        
        gmr_smoothness = np.mean(np.sum(gmr_jerk ** 2, axis=1))
        kdmr_smoothness = np.mean(np.sum(kdmr_jerk ** 2, axis=1))
        
        # Foot height consistency (lower is better for stance)
        gmr_foot_heights = gmr_trajectory[:, 2]  # Root height as proxy
        kdmr_foot_heights = kdmr_traj[:, 2]
        
        return {
            'GMR': {
                'smoothness': gmr_smoothness,
                'mean_foot_height': np.mean(gmr_foot_heights),
                'std_foot_height': np.std(gmr_foot_heights),
            },
            'KDMR': {
                'smoothness': kdmr_smoothness,
                'mean_foot_height': np.mean(kdmr_foot_heights),
                'std_foot_height': np.std(kdmr_foot_heights),
            },
            'improvement': {
                'smoothness': (gmr_smoothness - kdmr_smoothness) / gmr_smoothness * 100,
            }
        }


def create_kdmr(robot_name: str,
                config: Optional[KDMRConfig] = None,
                assets_dir: Optional[str] = None) -> KDMR:
    """
    Factory function to create KDMR instance.
    
    Args:
        robot_name: Name of robot (e.g., 'unitree_g1')
        config: KDMR configuration
        assets_dir: Path to assets directory
        
    Returns:
        Configured KDMR instance
    """
    if assets_dir is None:
        # Default to GMR assets
        assets_dir = Path(__file__).parent.parent.parent / 'assets'
    
    # Find robot XML
    robot_xml_map = {
        'unitree_g1': 'unitree_g1/g1_mocap_29dof.xml',
        'unitree_h1': 'unitree_h1/h1.xml',
        'booster_t1': 'booster_t1/T1_locomotion.xml',
    }
    
    if robot_name not in robot_xml_map:
        raise ValueError(f"Unknown robot: {robot_name}")
    
    robot_xml_path = Path(assets_dir) / robot_xml_map[robot_name]
    
    # Find IK config
    ik_config_map = {
        'unitree_g1': 'smplx_to_g1.json',
        'unitree_h1': 'smplx_to_h1.json',
        'booster_t1': 'smplx_to_t1.json',
    }
    
    ik_config_path = None
    if robot_name in ik_config_map:
        ik_path = Path(__file__).parent.parent.parent / 'general_motion_retargeting' / 'ik_configs' / ik_config_map[robot_name]
        if ik_path.exists():
            ik_config_path = str(ik_path)
    
    return KDMR(
        robot_xml_path=str(robot_xml_path),
        config=config,
        ik_config_path=ik_config_path
    )
