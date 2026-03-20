"""
Contact mode definitions for KDMR.

This module defines contact modes and related data structures
for multi-contact locomotion modeling.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterator
import numpy as np


class ContactMode(Enum):
    """
    Contact modes for foot-ground interaction.
    
    Based on biomechanics of human walking:
    - HEEL: Initial contact phase, heel touching ground
    - FLAT: Full foot contact, both heel and toe on ground
    - TOE: Push-off phase, only toe on ground
    - SWING: Swing phase, foot in air
    """
    HEEL = auto()
    FLAT = auto()
    TOE = auto()
    SWING = auto()
    
    @classmethod
    def from_string(cls, mode_str: str) -> 'ContactMode':
        """Convert string to ContactMode."""
        mode_map = {
            'heel': cls.HEEL,
            'flat': cls.FLAT,
            'toe': cls.TOE,
            'swing': cls.SWING,
            'stance': cls.FLAT,  # Alias for flat
        }
        return mode_map.get(mode_str.lower(), cls.SWING)
    
    def is_stance(self) -> bool:
        """Check if this is a stance phase (foot on ground)."""
        return self in (ContactMode.HEEL, ContactMode.FLAT, ContactMode.TOE)
    
    def is_swing(self) -> bool:
        """Check if this is a swing phase (foot in air)."""
        return self == ContactMode.SWING
    
    def get_color(self) -> Tuple[float, float, float, float]:
        """Get RGBA color for visualization."""
        colors = {
            ContactMode.HEEL: (1.0, 0.0, 0.0, 0.8),    # Red
            ContactMode.FLAT: (0.0, 1.0, 0.0, 0.8),    # Green
            ContactMode.TOE: (0.0, 0.0, 1.0, 0.8),     # Blue
            ContactMode.SWING: (0.5, 0.5, 0.5, 0.3),   # Gray
        }
        return colors[self]


@dataclass
class ContactState:
    """
    Contact state for a single foot at a single time step.
    
    Attributes:
        mode: Contact mode
        position: Contact position in world frame (3,)
        normal: Contact normal direction (3,)
        force: Contact force (3,) - only valid for stance phases
        cop: Center of pressure relative to foot frame (2,)
    """
    mode: ContactMode
    position: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None
    force: Optional[np.ndarray] = None
    cop: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate and convert arrays."""
        if self.position is not None:
            self.position = np.asarray(self.position)
        if self.normal is not None:
            self.normal = np.asarray(self.normal)
        if self.force is not None:
            self.force = np.asarray(self.force)
        if self.cop is not None:
            self.cop = np.asarray(self.cop)


@dataclass
class ContactSequence:
    """
    Contact sequence for a single foot over time.
    
    This class stores and manipulates the contact mode sequence
    for one foot throughout a motion trajectory.
    
    Attributes:
        foot_name: Name of the foot ('left' or 'right')
        modes: List of contact modes over time
        timestamps: Timestamps for each mode
        positions: Contact positions over time (N, 3)
        forces: Contact forces over time (N, 3)
    """
    foot_name: str
    modes: List[ContactMode] = field(default_factory=list)
    timestamps: np.ndarray = None
    positions: Optional[np.ndarray] = None
    forces: Optional[np.ndarray] = None
    
    def __len__(self) -> int:
        return len(self.modes)
    
    def __getitem__(self, idx: int) -> ContactState:
        """Get contact state at index."""
        pos = self.positions[idx] if self.positions is not None else None
        force = self.forces[idx] if self.forces is not None else None
        
        return ContactState(
            mode=self.modes[idx],
            position=pos,
            force=force
        )
    
    def __iter__(self) -> Iterator[ContactState]:
        """Iterate over contact states."""
        for i in range(len(self)):
            yield self[i]
    
    def add_mode(self, mode: ContactMode, timestamp: float = None):
        """Add a contact mode to the sequence."""
        self.modes.append(mode)
        if self.timestamps is None:
            self.timestamps = np.array([timestamp] if timestamp else [len(self.modes) - 1])
        elif timestamp is not None:
            self.timestamps = np.append(self.timestamps, timestamp)
    
    def get_phase_indices(self, mode: ContactMode) -> List[Tuple[int, int]]:
        """
        Get start and end indices for all phases of a given mode.
        
        Args:
            mode: Contact mode to find
            
        Returns:
            List of (start_idx, end_idx) tuples for each phase
        """
        phases = []
        in_phase = False
        start_idx = 0
        
        for i, m in enumerate(self.modes):
            if m == mode and not in_phase:
                start_idx = i
                in_phase = True
            elif m != mode and in_phase:
                phases.append((start_idx, i))
                in_phase = False
        
        if in_phase:
            phases.append((start_idx, len(self.modes)))
        
        return phases
    
    def get_stance_phases(self) -> List[Tuple[int, int]]:
        """Get all stance phase indices."""
        phases = []
        in_stance = False
        start_idx = 0
        
        for i, mode in enumerate(self.modes):
            if mode.is_stance() and not in_stance:
                start_idx = i
                in_stance = True
            elif mode.is_swing() and in_stance:
                phases.append((start_idx, i))
                in_stance = False
        
        if in_stance:
            phases.append((start_idx, len(self.modes)))
        
        return phases
    
    def get_swing_phases(self) -> List[Tuple[int, int]]:
        """Get all swing phase indices."""
        return self.get_phase_indices(ContactMode.SWING)
    
    def get_mode_at_time(self, t: float) -> ContactMode:
        """Get contact mode at a specific time."""
        if self.timestamps is None:
            idx = int(t)
        else:
            idx = np.searchsorted(self.timestamps, t)
        
        idx = min(idx, len(self.modes) - 1)
        return self.modes[idx]
    
    def get_stance_duration(self) -> float:
        """Get total stance duration in seconds."""
        if self.timestamps is None:
            return sum(1 for m in self.modes if m.is_stance())
        
        total = 0.0
        for start, end in self.get_stance_phases():
            total += self.timestamps[end-1] - self.timestamps[start]
        
        return total
    
    def get_swing_duration(self) -> float:
        """Get total swing duration in seconds."""
        if self.timestamps is None:
            return sum(1 for m in self.modes if m.is_swing())
        
        total = 0.0
        for start, end in self.get_swing_phases():
            total += self.timestamps[end-1] - self.timestamps[start]
        
        return total
    
    def get_duty_factor(self) -> float:
        """
        Get duty factor (fraction of time in stance).
        
        Duty factor for walking is typically ~0.6, for running ~0.3.
        """
        if len(self.modes) == 0:
            return 0.0
        
        stance_count = sum(1 for m in self.modes if m.is_stance())
        return stance_count / len(self.modes)
    
    def to_array(self) -> np.ndarray:
        """Convert contact modes to numeric array."""
        mode_values = {
            ContactMode.HEEL: 0,
            ContactMode.FLAT: 1,
            ContactMode.TOE: 2,
            ContactMode.SWING: 3,
        }
        return np.array([mode_values[m] for m in self.modes])
    
    @classmethod
    def from_array(cls, 
                   foot_name: str,
                   mode_array: np.ndarray,
                   timestamps: Optional[np.ndarray] = None) -> 'ContactSequence':
        """Create ContactSequence from numeric array."""
        mode_map = {
            0: ContactMode.HEEL,
            1: ContactMode.FLAT,
            2: ContactMode.TOE,
            3: ContactMode.SWING,
        }
        modes = [mode_map[int(v)] for v in mode_array]
        
        return cls(
            foot_name=foot_name,
            modes=modes,
            timestamps=timestamps
        )


@dataclass
class DualContactSequence:
    """
    Contact sequences for both feet.
    
    Attributes:
        left: Contact sequence for left foot
        right: Contact sequence for right foot
    """
    left: ContactSequence
    right: ContactSequence
    
    def __len__(self) -> int:
        return len(self.left)
    
    def get_phase_at_index(self, idx: int) -> Tuple[ContactMode, ContactMode]:
        """Get contact modes for both feet at index."""
        return self.left.modes[idx], self.right.modes[idx]
    
    def get_double_support_phases(self) -> List[Tuple[int, int]]:
        """
        Get double support phase indices.
        
        Double support is when both feet are on the ground.
        """
        phases = []
        in_double = False
        start_idx = 0
        
        for i in range(len(self)):
            left_stance = self.left.modes[i].is_stance()
            right_stance = self.right.modes[i].is_stance()
            
            if left_stance and right_stance and not in_double:
                start_idx = i
                in_double = True
            elif not (left_stance and right_stance) and in_double:
                phases.append((start_idx, i))
                in_double = False
        
        if in_double:
            phases.append((start_idx, len(self)))
        
        return phases
    
    def get_single_support_phases(self) -> List[Tuple[int, int, str]]:
        """
        Get single support phase indices.
        
        Returns:
            List of (start_idx, end_idx, stance_foot_name)
        """
        phases = []
        
        for foot_name, seq in [('left', self.left), ('right', self.right)]:
            in_single = False
            start_idx = 0
            
            for i in range(len(self)):
                this_stance = seq.modes[i].is_stance()
                other_stance = (self.right if foot_name == 'left' else self.left).modes[i].is_stance()
                
                if this_stance and not other_stance and not in_single:
                    start_idx = i
                    in_single = True
                elif not (this_stance and not other_stance) and in_single:
                    phases.append((start_idx, i, foot_name))
                    in_single = False
            
            if in_single:
                phases.append((start_idx, len(self), foot_name))
        
        return phases
    
    def is_gait_periodic(self, tolerance: float = 0.1) -> bool:
        """
        Check if gait appears periodic.
        
        Compares duty factors and phase durations between feet.
        """
        left_duty = self.left.get_duty_factor()
        right_duty = self.right.get_duty_factor()
        
        return abs(left_duty - right_duty) < tolerance


def compute_gait_parameters(contact_seq: DualContactSequence,
                           fps: float) -> Dict[str, float]:
    """
    Compute standard gait parameters from contact sequence.
    
    Args:
        contact_seq: Dual contact sequence
        fps: Frame rate
        
    Returns:
        Dictionary of gait parameters:
        - stride_time: Time for one complete gait cycle
        - step_time: Time between consecutive foot contacts
        - stance_time: Time in stance phase
        - swing_time: Time in swing phase
        - double_support_time: Time in double support
        - duty_factor: Fraction of stride in stance
        - cadence: Steps per minute
    """
    dt = 1.0 / fps
    
    # Get stance phases for each foot
    left_stances = contact_seq.left.get_stance_phases()
    right_stances = contact_seq.right.get_stance_phases()
    
    # Compute times
    left_stance_time = len(left_stances) * dt if left_stances else 0
    right_stance_time = len(right_stances) * dt if right_stances else 0
    
    # Double support
    double_support = contact_seq.get_double_support_phases()
    double_support_time = sum(end - start for start, end in double_support) * dt
    
    # Stride time (time between same foot contacts)
    if len(left_stances) >= 2:
        stride_time = (left_stances[1][0] - left_stances[0][0]) * dt
    else:
        stride_time = len(contact_seq) * dt
    
    # Step time (time between left and right contacts)
    if left_stances and right_stances:
        step_time = abs(right_stances[0][0] - left_stances[0][0]) * dt
    else:
        step_time = stride_time / 2
    
    # Duty factor
    total_frames = len(contact_seq)
    stance_frames = sum(1 for i in range(total_frames) 
                       if contact_seq.left.modes[i].is_stance())
    duty_factor = stance_frames / total_frames if total_frames > 0 else 0
    
    return {
        'stride_time': stride_time,
        'step_time': step_time,
        'stance_time': (left_stance_time + right_stance_time) / 2,
        'swing_time': stride_time - (left_stance_time + right_stance_time) / 2,
        'double_support_time': double_support_time,
        'duty_factor': duty_factor,
        'cadence': 60.0 / step_time if step_time > 0 else 0,  # steps per minute
    }
