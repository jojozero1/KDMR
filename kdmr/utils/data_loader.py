"""
Data loading utilities for KDMR.

This module provides unified data loading for:
- Human motion data (SMPLX, BVH, FBX formats)
- Ground Reaction Force (GRF) data
- Robot model configurations
"""

import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy.interpolate import interp1d


@dataclass
class HumanMotionData:
    """
    Container for human motion data.
    
    Attributes:
        positions: Joint positions in world frame, shape (N, J, 3)
        orientations: Joint orientations as quaternions, shape (N, J, 4)
        joint_names: List of joint names
        fps: Frame rate
        duration: Duration in seconds
    """
    positions: np.ndarray
    orientations: np.ndarray
    joint_names: List[str]
    fps: float
    duration: float
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def get_frame(self, idx: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get a single frame as dict of {joint_name: (position, orientation)}."""
        return {
            name: (self.positions[idx, i], self.orientations[idx, i])
            for i, name in enumerate(self.joint_names)
        }


@dataclass
class GRFData:
    """
    Container for Ground Reaction Force data.
    
    Attributes:
        forces: Force vectors, shape (N, 3) for single foot or (N, 2, 3) for both feet
        moments: Moment vectors, shape (N, 3) or (N, 2, 3)
        cop: Center of pressure, shape (N, 2) or (N, 2, 2)
        timestamps: Timestamps in seconds, shape (N,)
        fps: Sampling rate
    """
    forces: np.ndarray
    moments: Optional[np.ndarray]
    cop: Optional[np.ndarray]
    timestamps: np.ndarray
    fps: float
    
    def get_vertical_force(self, foot_idx: int = 0) -> np.ndarray:
        """Get vertical (z) force component for a foot."""
        if self.forces.ndim == 2:
            return self.forces[:, 2]
        return self.forces[:, foot_idx, 2]
    
    def get_horizontal_forces(self, foot_idx: int = 0) -> np.ndarray:
        """Get horizontal (x, y) force components for a foot."""
        if self.forces.ndim == 2:
            return self.forces[:, :2]
        return self.forces[:, foot_idx, :2]


@dataclass
class RobotTrajectory:
    """
    Container for robot trajectory data.
    
    Attributes:
        qpos: Joint positions (root_pos, root_quat, joint_angles), shape (N, D)
        qvel: Joint velocities, shape (N, D-1) or (N, D)
        tau: Joint torques, shape (N, n_actuators)
        contact_forces: Contact forces, shape (N, n_contacts, 3)
        fps: Frame rate
    """
    qpos: np.ndarray
    qvel: Optional[np.ndarray]
    tau: Optional[np.ndarray]
    contact_forces: Optional[np.ndarray]
    fps: float
    
    @property
    def root_pos(self) -> np.ndarray:
        """Get root positions, shape (N, 3)."""
        return self.qpos[:, :3]
    
    @property
    def root_quat(self) -> np.ndarray:
        """Get root orientations (quaternions), shape (N, 4)."""
        return self.qpos[:, 3:7]
    
    @property
    def joint_angles(self) -> np.ndarray:
        """Get joint angles, shape (N, nq-7)."""
        return self.qpos[:, 7:]


class DataLoader:
    """
    Unified data loader for KDMR.
    
    Supports loading:
    - Human motion from various formats (SMPLX NPZ, BVH, FBX)
    - GRF data from CSV or NPZ
    - Robot configurations from YAML
    """
    
    # Standard joint name mapping for different formats
    SMPLX_JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye',
        'right_eye', 'left_index1', 'left_index2', 'left_index3', 'left_middle1',
        'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3',
        'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2',
        'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1',
        'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3',
        'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2',
        'right_thumb3', 'nose', 'right_eye_brow', 'left_eye_brow', 'right_eye_lower',
        'left_eye_lower', 'right_eye_upper', 'left_eye_upper', 'right_eyelid',
        'left_eyelid', 'mouth_left', 'mouth_right'
    ]
    
    BVH_JOINT_NAMES = [
        'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder',
        'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm',
        'RightForeArm', 'RightHand', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe',
        'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe'
    ]
    
    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_root: Root directory for data files
        """
        self.data_root = Path(data_root) if data_root else Path.cwd()
    
    def load_smplx_motion(self, 
                          filepath: str, 
                          body_model_path: Optional[str] = None) -> HumanMotionData:
        """
        Load human motion from SMPLX NPZ file.
        
        Args:
            filepath: Path to NPZ file
            body_model_path: Path to SMPLX body model (optional, for forward kinematics)
            
        Returns:
            HumanMotionData object
        """
        filepath = self._resolve_path(filepath)
        data = np.load(filepath, allow_pickle=True)
        
        # Extract motion parameters
        root_orient = data['root_orient']  # (N, 3) axis-angle
        pose_body = data['pose_body']      # (N, 63) body pose
        trans = data['trans']              # (N, 3) translation
        fps = data.get('mocap_frame_rate', data.get('mocap_framerate', 30.0))
        
        if hasattr(fps, 'item'):
            fps = fps.item()
        
        n_frames = len(root_orient)
        
        # If body model is provided, use forward kinematics
        if body_model_path is not None:
            positions, orientations = self._smplx_forward_kinematics(
                root_orient, pose_body, trans, body_model_path
            )
        else:
            # Use simplified joint positions from pose parameters
            positions, orientations = self._estimate_joint_positions_smplx(
                root_orient, pose_body, trans
            )
        
        # Get relevant joint names (body only, no hands/face)
        joint_names = self.SMPLX_JOINT_NAMES[:22]  # First 22 body joints
        
        return HumanMotionData(
            positions=positions,
            orientations=orientations,
            joint_names=joint_names,
            fps=fps,
            duration=n_frames / fps
        )
    
    def load_bvh_motion(self, filepath: str) -> HumanMotionData:
        """
        Load human motion from BVH file.
        
        Args:
            filepath: Path to BVH file
            
        Returns:
            HumanMotionData object
        """
        filepath = self._resolve_path(filepath)
        
        # Parse BVH file
        positions, orientations, joint_names, fps = self._parse_bvh(filepath)
        
        return HumanMotionData(
            positions=positions,
            orientations=orientations,
            joint_names=joint_names,
            fps=fps,
            duration=len(positions) / fps
        )
    
    def load_grf_data(self, 
                      filepath: str, 
                      format: str = 'auto') -> GRFData:
        """
        Load Ground Reaction Force data.
        
        Args:
            filepath: Path to GRF data file
            format: Data format ('csv', 'npz', 'c3d', 'auto')
            
        Returns:
            GRFData object
        """
        filepath = self._resolve_path(filepath)
        
        if format == 'auto':
            format = filepath.suffix[1:].lower()
        
        if format == 'npz':
            return self._load_grf_npz(filepath)
        elif format == 'csv':
            return self._load_grf_csv(filepath)
        else:
            raise ValueError(f"Unsupported GRF format: {format}")
    
    def load_robot_config(self, robot_name: str) -> Dict[str, Any]:
        """
        Load robot configuration from YAML file.
        
        Args:
            robot_name: Name of the robot (e.g., 'unitree_g1')
            
        Returns:
            Configuration dictionary
        """
        config_path = self.data_root / 'configs' / 'robots' / f'{robot_name}.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Robot config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_ik_config(self, 
                       src_format: str, 
                       robot_name: str) -> Dict[str, Any]:
        """
        Load IK configuration (compatible with GMR format).
        
        Args:
            src_format: Source format (e.g., 'smplx', 'bvh_lafan1')
            robot_name: Target robot name
            
        Returns:
            IK configuration dictionary
        """
        # Try to load from GMR's ik_configs directory
        gmr_config_path = self.data_root.parent / 'general_motion_retargeting' / 'ik_configs' / f'{src_format}_to_{robot_name}.json'
        
        if gmr_config_path.exists():
            with open(gmr_config_path, 'r') as f:
                return json.load(f)
        
        # Fallback to KDMR configs
        config_path = self.data_root / 'configs' / 'ik' / f'{src_format}_to_{robot_name}.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        raise FileNotFoundError(f"IK config not found for {src_format} -> {robot_name}")
    
    def save_trajectory(self, 
                       trajectory: RobotTrajectory, 
                       filepath: str, 
                       format: str = 'npz'):
        """
        Save robot trajectory to file.
        
        Args:
            trajectory: RobotTrajectory object
            filepath: Output path
            format: Output format ('npz', 'pkl')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npz':
            np.savez(
                filepath,
                qpos=trajectory.qpos,
                qvel=trajectory.qvel if trajectory.qvel is not None else np.array([]),
                tau=trajectory.tau if trajectory.tau is not None else np.array([]),
                contact_forces=trajectory.contact_forces if trajectory.contact_forces is not None else np.array([]),
                fps=trajectory.fps
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_trajectory(self, filepath: str) -> RobotTrajectory:
        """
        Load robot trajectory from file.
        
        Args:
            filepath: Path to trajectory file
            
        Returns:
            RobotTrajectory object
        """
        filepath = self._resolve_path(filepath)
        data = np.load(filepath, allow_pickle=True)
        
        qvel = data['qvel'] if data['qvel'].size > 0 else None
        tau = data['tau'] if data['tau'].size > 0 else None
        contact_forces = data['contact_forces'] if data['contact_forces'].size > 0 else None
        
        return RobotTrajectory(
            qpos=data['qpos'],
            qvel=qvel,
            tau=tau,
            contact_forces=contact_forces,
            fps=float(data['fps'])
        )
    
    def _resolve_path(self, filepath: Union[str, Path]) -> Path:
        """Resolve path relative to data root."""
        filepath = Path(filepath)
        if filepath.is_absolute():
            return filepath
        return self.data_root / filepath
    
    def _smplx_forward_kinematics(self, 
                                  root_orient: np.ndarray,
                                  pose_body: np.ndarray,
                                  trans: np.ndarray,
                                  body_model_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute joint positions and orientations using SMPLX forward kinematics.
        
        Requires smplx library and body model files.
        """
        try:
            import torch
            import smplx
            from smplx.joint_names import JOINT_NAMES
        except ImportError:
            raise ImportError("smplx library required for forward kinematics")
        
        n_frames = len(root_orient)
        
        # Load body model
        body_model = smplx.create(
            body_model_path,
            'smplx',
            gender='neutral',
            use_pca=False
        )
        
        # Process in batches to avoid memory issues
        batch_size = 100
        all_positions = []
        all_orientations = []
        
        for i in range(0, n_frames, batch_size):
            end_idx = min(i + batch_size, n_frames)
            
            output = body_model(
                global_orient=torch.tensor(root_orient[i:end_idx]).float(),
                body_pose=torch.tensor(pose_body[i:end_idx]).float(),
                transl=torch.tensor(trans[i:end_idx]).float(),
                return_full_pose=True
            )
            
            joints = output.joints.detach().numpy()
            full_pose = output.full_pose.detach().numpy()
            
            all_positions.append(joints)
            
            # Compute orientations from pose parameters
            orientations = self._compute_orientations_from_pose(
                output.global_orient.numpy(),
                full_pose.reshape(-1, 55, 3),
                body_model.parents
            )
            all_orientations.append(orientations)
        
        positions = np.concatenate(all_positions, axis=0)
        orientations = np.concatenate(all_orientations, axis=0)
        
        # Select body joints only (first 21 joints)
        return positions[:, :22, :], orientations[:, :22, :]
    
    def _estimate_joint_positions_smplx(self,
                                        root_orient: np.ndarray,
                                        pose_body: np.ndarray,
                                        trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate joint positions from SMPLX parameters without body model.
        
        This is a simplified approximation for when body model is not available.
        """
        from scipy.spatial.transform import Rotation as R
        
        n_frames = len(root_orient)
        n_joints = 22  # Body joints
        
        # Simplified: use trans as pelvis position and estimate others
        positions = np.zeros((n_frames, n_joints, 3))
        orientations = np.zeros((n_frames, n_joints, 4))
        
        for i in range(n_frames):
            # Root orientation
            root_rot = R.from_rotvec(root_orient[i])
            
            # Pelvis
            positions[i, 0] = trans[i]
            orientations[i, 0] = root_rot.as_quat(scalar_first=True)
            
            # Simplified joint estimation (would need proper FK for accuracy)
            # This is a placeholder - real implementation needs body model
            for j in range(1, n_joints):
                positions[i, j] = trans[i]  # Placeholder
                orientations[i, j] = [1, 0, 0, 0]  # Identity
        
        return positions, orientations
    
    def _compute_orientations_from_pose(self,
                                        global_orient: np.ndarray,
                                        full_pose: np.ndarray,
                                        parents: np.ndarray) -> np.ndarray:
        """Compute global joint orientations from pose parameters."""
        from scipy.spatial.transform import Rotation as R
        
        n_frames = len(global_orient)
        n_joints = len(parents)
        
        orientations = np.zeros((n_frames, n_joints, 4))
        
        for f in range(n_frames):
            joint_rots = []
            for j in range(n_joints):
                if j == 0:
                    rot = R.from_rotvec(global_orient[f])
                else:
                    parent_rot = joint_rots[parents[j]]
                    local_rot = R.from_rotvec(full_pose[f, j])
                    rot = parent_rot * local_rot
                
                joint_rots.append(rot)
                orientations[f, j] = rot.as_quat(scalar_first=True)
        
        return orientations
    
    def _parse_bvh(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
        """Parse BVH file to extract motion data."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse skeleton hierarchy
        joint_names = []
        joint_offsets = {}
        joint_parents = {}
        joint_channels = {}
        
        current_joint = None
        parent_stack = []
        channel_idx = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('ROOT') or line.startswith('JOINT'):
                parts = line.split()
                joint_name = parts[1]
                joint_names.append(joint_name)
                current_joint = joint_name
                
                if parent_stack:
                    joint_parents[joint_name] = parent_stack[-1]
                else:
                    joint_parents[joint_name] = None
                
                parent_stack.append(joint_name)
            
            elif line.startswith('OFFSET'):
                parts = line.split()
                offset = [float(x) for x in parts[1:4]]
                joint_offsets[current_joint] = offset
            
            elif line.startswith('CHANNELS'):
                parts = line.split()
                n_channels = int(parts[1])
                channels = parts[2:2+n_channels]
                joint_channels[current_joint] = (channel_idx, channels)
                channel_idx += n_channels
            
            elif line == '}':
                if parent_stack:
                    parent_stack.pop()
            
            elif line == 'MOTION':
                break
            
            i += 1
        
        # Parse motion data
        i += 1
        n_frames = 0
        fps = 30.0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('Frames:'):
                n_frames = int(line.split(':')[1].strip())
            elif line.startswith('Frame Time:'):
                frame_time = float(line.split(':')[1].strip())
                fps = 1.0 / frame_time
            elif line and not line.startswith('Frame'):
                break
            
            i += 1
        
        # Parse frame data
        n_channels = channel_idx
        frame_data = np.zeros((n_frames, n_channels))
        
        for frame_idx in range(n_frames):
            if i + frame_idx < len(lines):
                values = [float(x) for x in lines[i + frame_idx].strip().split()]
                frame_data[frame_idx] = values[:n_channels]
        
        # Convert to positions and orientations
        # This is simplified - full implementation would do proper FK
        positions = np.zeros((n_frames, len(joint_names), 3))
        orientations = np.zeros((n_frames, len(joint_names), 4))
        
        for j, joint_name in enumerate(joint_names):
            if joint_name in joint_channels:
                idx, channels = joint_channels[joint_name]
                
                # Extract position and rotation from channels
                pos_idx = [i for i, c in enumerate(channels) if 'position' in c.lower()]
                rot_idx = [i for i, c in enumerate(channels) if 'rotation' in c.lower()]
                
                if pos_idx:
                    for k, pi in enumerate(pos_idx):
                        positions[:, j, k] = frame_data[:, idx + pi]
                
                if len(rot_idx) >= 3:
                    from scipy.spatial.transform import Rotation as R
                    euler = frame_data[:, idx + rot_idx[0]:idx + rot_idx[2] + 1]
                    for f in range(n_frames):
                        rot = R.from_euler('xyz', euler[f], degrees=True)
                        orientations[f, j] = rot.as_quat(scalar_first=True)
                else:
                    orientations[:, j] = [1, 0, 0, 0]
        
        return positions, orientations, joint_names, fps
    
    def _load_grf_npz(self, filepath: Path) -> GRFData:
        """Load GRF data from NPZ file."""
        data = np.load(filepath, allow_pickle=True)
        
        return GRFData(
            forces=data['forces'],
            moments=data.get('moments', None),
            cop=data.get('cop', None),
            timestamps=data.get('timestamps', np.arange(len(data['forces']))),
            fps=float(data.get('fps', 100.0))
        )
    
    def _load_grf_csv(self, filepath: Path) -> GRFData:
        """Load GRF data from CSV file."""
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        
        # Assume format: time, Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, ...
        timestamps = data[:, 0]
        
        # Detect if single or dual foot
        if data.shape[1] >= 7:
            # Single foot: time, Fx, Fy, Fz, Mx, My, Mz
            forces = data[:, 1:4]
            moments = data[:, 4:7] if data.shape[1] >= 7 else None
        else:
            forces = data[:, 1:4]
            moments = None
        
        fps = 1.0 / np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 100.0
        
        return GRFData(
            forces=forces,
            moments=moments,
            cop=None,
            timestamps=timestamps,
            fps=fps
        )
