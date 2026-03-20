"""
Contact sequence estimator for KDMR.

This module implements the biomechanics-inspired contact sequence estimation
from GRF data as described in the KDMR paper (arXiv:2603.09956).

The estimator detects:
- Heel-only contact phase
- Flat-foot (full contact) phase  
- Toe-only contact phase
- Swing phase
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from kdmr.contact.contact_mode import (
    ContactMode, 
    ContactSequence, 
    DualContactSequence,
    ContactState
)
from kdmr.contact.grf_processor import ProcessedGRF, GRFProcessor


class ContactEstimator:
    """
    Estimate contact sequence from GRF data.
    
    Based on biomechanical principles of human walking:
    1. Vertical GRF shows characteristic double-peak during stance
    2. Mid-stance dip corresponds to flat-foot phase
    3. Rising edge before dip is heel contact
    4. Falling edge after dip is toe contact
    """
    
    def __init__(self,
                 force_threshold: float = 20.0,
                 normalized_threshold: float = 0.05,
                 min_stance_frames: int = 5,
                 use_normalized: bool = True):
        """
        Initialize contact estimator.
        
        Args:
            force_threshold: Absolute force threshold (N)
            normalized_threshold: Normalized threshold (fraction of body weight)
            min_stance_frames: Minimum frames for valid stance phase
            use_normalized: Whether to use normalized threshold
        """
        self.force_threshold = force_threshold
        self.normalized_threshold = normalized_threshold
        self.min_stance_frames = min_stance_frames
        self.use_normalized = use_normalized
    
    def estimate_from_grf(self,
                         processed_grf: ProcessedGRF,
                         foot_name: str = 'left') -> ContactSequence:
        """
        Estimate contact sequence from processed GRF data for one foot.
        
        Args:
            processed_grf: Processed GRF data
            foot_name: 'left' or 'right'
            
        Returns:
            ContactSequence for the specified foot
        """
        # Get vertical force for this foot
        if foot_name == 'left':
            vertical_force = processed_grf.vertical_left
        else:
            vertical_force = processed_grf.vertical_right
        
        # Get threshold
        if self.use_normalized:
            threshold = self.normalized_threshold * processed_grf.body_weight
        else:
            threshold = self.force_threshold
        
        # Detect stance and swing phases
        modes = self._detect_contact_modes(
            vertical_force, 
            threshold,
            processed_grf.fps
        )
        
        return ContactSequence(
            foot_name=foot_name,
            modes=modes,
            timestamps=processed_grf.timestamps,
            forces=np.stack([
                processed_grf.vertical_left if foot_name == 'left' 
                else processed_grf.vertical_right,
                processed_grf.horizontal_left[:, 0] if foot_name == 'left'
                else processed_grf.horizontal_right[:, 0],
                processed_grf.horizontal_left[:, 1] if foot_name == 'left'
                else processed_grf.horizontal_right[:, 1]
            ], axis=1) if processed_grf.horizontal_left is not None else None
        )
    
    def estimate_dual_contact(self,
                              processed_grf: ProcessedGRF) -> DualContactSequence:
        """
        Estimate contact sequences for both feet.
        
        Args:
            processed_grf: Processed GRF data
            
        Returns:
            DualContactSequence for both feet
        """
        left_seq = self.estimate_from_grf(processed_grf, 'left')
        right_seq = self.estimate_from_grf(processed_grf, 'right')
        
        return DualContactSequence(left=left_seq, right=right_seq)
    
    def _detect_contact_modes(self,
                              vertical_force: np.ndarray,
                              threshold: float,
                              fps: float) -> List[ContactMode]:
        """
        Detect contact modes from vertical GRF profile.
        
        The algorithm:
        1. Identify stance phases (Fz > threshold)
        2. For each stance phase:
           a. Find mid-stance dip (local minimum in Fz)
           b. Define flat-foot phase around the dip
           c. Pre-flat region = heel contact
           d. Post-flat region = toe contact
        
        Args:
            vertical_force: Vertical GRF time series
            threshold: Contact threshold
            fps: Sampling rate
            
        Returns:
            List of ContactMode for each frame
        """
        n_frames = len(vertical_force)
        modes = [ContactMode.SWING] * n_frames
        
        # Find stance phases
        in_stance = vertical_force > threshold
        
        # Find stance phase boundaries
        stance_starts, stance_ends = self._find_stance_phases(in_stance)
        
        for start, end in zip(stance_starts, stance_ends):
            if end - start < self.min_stance_frames:
                continue
            
            # Get force profile for this stance phase
            stance_force = vertical_force[start:end]
            
            # Find mid-stance dip
            dip_idx = self._find_mid_stance_dip(stance_force)
            
            # Determine phase boundaries
            heel_end, flat_start, flat_end, toe_start = self._determine_phase_boundaries(
                stance_force, dip_idx, fps
            )
            
            # Assign modes
            # Heel phase: start to heel_end
            for i in range(start, start + heel_end):
                if i < n_frames:
                    modes[i] = ContactMode.HEEL
            
            # Flat phase: flat_start to flat_end
            for i in range(start + flat_start, start + flat_end):
                if i < n_frames:
                    modes[i] = ContactMode.FLAT
            
            # Toe phase: toe_start to end
            for i in range(start + toe_start, end):
                if i < n_frames:
                    modes[i] = ContactMode.TOE
        
        return modes
    
    def _find_stance_phases(self, 
                           in_stance: np.ndarray) -> Tuple[List[int], List[int]]:
        """Find start and end indices of stance phases."""
        starts = []
        ends = []
        
        in_phase = False
        
        for i, stance in enumerate(in_stance):
            if stance and not in_phase:
                starts.append(i)
                in_phase = True
            elif not stance and in_phase:
                ends.append(i)
                in_phase = False
        
        # Handle case where stance continues to end
        if in_phase:
            ends.append(len(in_stance))
        
        return starts, ends
    
    def _find_mid_stance_dip(self, stance_force: np.ndarray) -> int:
        """
        Find the mid-stance dip in vertical GRF.
        
        The vertical GRF during walking typically shows:
        - First peak (weight acceptance)
        - Mid-stance dip (body over stance leg)
        - Second peak (push-off)
        
        Args:
            stance_force: Vertical GRF during stance phase
            
        Returns:
            Index of mid-stance dip (local minimum)
        """
        if len(stance_force) < 5:
            return len(stance_force) // 2
        
        # Smooth the force profile
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(stance_force, size=3)
        
        # Find local minima
        # Mid-stance dip is typically in the middle third
        start_search = len(stance_force) // 4
        end_search = 3 * len(stance_force) // 4
        
        search_region = smoothed[start_search:end_search]
        
        if len(search_region) == 0:
            return len(stance_force) // 2
        
        # Find minimum in search region
        local_min_idx = np.argmin(search_region)
        
        return start_search + local_min_idx
    
    def _determine_phase_boundaries(self,
                                   stance_force: np.ndarray,
                                   dip_idx: int,
                                   fps: float) -> Tuple[int, int, int, int]:
        """
        Determine boundaries for heel, flat, and toe phases.
        
        Args:
            stance_force: Vertical GRF during stance
            dip_idx: Index of mid-stance dip
            fps: Sampling rate
            
        Returns:
            Tuple of (heel_end, flat_start, flat_end, toe_start)
        """
        n = len(stance_force)
        
        # Define flat-foot phase as window around mid-stance dip
        # Typical flat-foot duration is ~20-30% of stance phase
        flat_duration = max(3, int(0.2 * n))
        
        flat_start = max(0, dip_idx - flat_duration // 2)
        flat_end = min(n, dip_idx + flat_duration // 2)
        
        # Heel phase: from start to flat phase
        heel_end = flat_start
        
        # Toe phase: from flat phase to end
        toe_start = flat_end
        
        # Ensure minimum durations
        min_phase = 2
        if heel_end < min_phase:
            heel_end = min_phase
            flat_start = min_phase
        
        if n - toe_start < min_phase:
            toe_start = n - min_phase
            flat_end = n - min_phase
        
        return heel_end, flat_start, flat_end, toe_start
    
    def estimate_from_motion_only(self,
                                  foot_positions: np.ndarray,
                                  fps: float,
                                  ground_height: float = 0.0,
                                  velocity_threshold: float = 0.5) -> ContactSequence:
        """
        Estimate contact sequence from foot motion alone (no GRF).
        
        This is a fallback when GRF data is not available.
        Uses foot height and velocity to estimate contact.
        
        Args:
            foot_positions: Foot position trajectory (N, 3)
            fps: Frame rate
            ground_height: Ground plane height
            velocity_threshold: Velocity threshold for contact detection
            
        Returns:
            Estimated ContactSequence
        """
        n_frames = len(foot_positions)
        
        # Compute foot height
        foot_height = foot_positions[:, 2] - ground_height
        
        # Compute foot velocity
        dt = 1.0 / fps
        velocity = np.gradient(foot_positions, dt, axis=0)
        vertical_velocity = velocity[:, 2]
        horizontal_velocity = np.linalg.norm(velocity[:, :2], axis=1)
        
        # Simple contact detection: low height + low velocity
        height_threshold = 0.05  # 5 cm
        vel_threshold = velocity_threshold
        
        potential_contact = (foot_height < height_threshold) & \
                           (horizontal_velocity < vel_threshold)
        
        # Smooth the contact signal
        from scipy.ndimage import binary_closing, binary_opening
        contact = binary_closing(binary_opening(potential_contact, iterations=2), iterations=2)
        
        # Convert to modes (simplified: no heel/toe distinction)
        modes = [ContactMode.FLAT if c else ContactMode.SWING for c in contact]
        
        timestamps = np.arange(n_frames) / fps
        
        return ContactSequence(
            foot_name='unknown',
            modes=modes,
            timestamps=timestamps
        )
    
    def refine_with_cop(self,
                       contact_seq: ContactSequence,
                       cop_data: np.ndarray,
                       foot_length: float = 0.26) -> ContactSequence:
        """
        Refine contact modes using Center of Pressure data.
        
        COP position along foot can help distinguish heel vs toe contact.
        
        Args:
            contact_seq: Initial contact sequence
            cop_data: COP positions along foot (N, 2) - (x, y) in foot frame
            foot_length: Foot length in meters
            
        Returns:
            Refined ContactSequence
        """
        if cop_data is None:
            return contact_seq
        
        refined_modes = list(contact_seq.modes)
        
        # COP x-position indicates contact location
        # x < 0.3 * foot_length: heel contact
        # x > 0.7 * foot_length: toe contact
        # otherwise: flat contact
        
        heel_threshold = 0.3 * foot_length
        toe_threshold = 0.7 * foot_length
        
        for i, mode in enumerate(refined_modes):
            if mode == ContactMode.SWING:
                continue
            
            if i < len(cop_data):
                cop_x = cop_data[i, 0]
                
                if cop_x < heel_threshold:
                    refined_modes[i] = ContactMode.HEEL
                elif cop_x > toe_threshold:
                    refined_modes[i] = ContactMode.TOE
                else:
                    refined_modes[i] = ContactMode.FLAT
        
        return ContactSequence(
            foot_name=contact_seq.foot_name,
            modes=refined_modes,
            timestamps=contact_seq.timestamps,
            positions=contact_seq.positions,
            forces=contact_seq.forces
        )
    
    def get_contact_forces_at_frame(self,
                                   contact_seq: ContactSequence,
                                   frame_idx: int) -> Optional[np.ndarray]:
        """
        Get contact force at a specific frame.
        
        Args:
            contact_seq: Contact sequence
            frame_idx: Frame index
            
        Returns:
            Contact force (3,) or None if not in contact
        """
        if frame_idx >= len(contact_seq):
            return None
        
        state = contact_seq[frame_idx]
        
        if state.mode.is_swing():
            return None
        
        return state.force
    
    def compute_contact_force_distribution(self,
                                          dual_seq: DualContactSequence,
                                          total_force: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Distribute total GRF between feet based on contact modes.
        
        Args:
            dual_seq: Dual contact sequence
            total_force: Total vertical GRF (N,)
            
        Returns:
            Dictionary with 'left' and 'right' force distributions
        """
        n_frames = len(dual_seq)
        
        left_force = np.zeros(n_frames)
        right_force = np.zeros(n_frames)
        
        for i in range(n_frames):
            left_mode, right_mode = dual_seq.get_phase_at_index(i)
            
            left_stance = left_mode.is_stance()
            right_stance = right_mode.is_stance()
            
            if left_stance and right_stance:
                # Double support - distribute evenly
                left_force[i] = total_force[i] * 0.5
                right_force[i] = total_force[i] * 0.5
            elif left_stance:
                left_force[i] = total_force[i]
            elif right_stance:
                right_force[i] = total_force[i]
        
        return {'left': left_force, 'right': right_force}


def create_contact_estimator_from_config(config: Dict) -> ContactEstimator:
    """
    Create ContactEstimator from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ContactEstimator
    """
    return ContactEstimator(
        force_threshold=config.get('force_threshold', 20.0),
        normalized_threshold=config.get('normalized_threshold', 0.05),
        min_stance_frames=config.get('min_stance_frames', 5),
        use_normalized=config.get('use_normalized', True)
    )
