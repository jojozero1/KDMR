"""
Ground Reaction Force (GRF) data processor for KDMR.

This module handles:
- GRF data loading and preprocessing
- Force signal filtering and smoothing
- Vertical/horizontal force decomposition
- COP (Center of Pressure) computation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import signal
from scipy.ndimage import uniform_filter1d


@dataclass
class ProcessedGRF:
    """
    Processed GRF data ready for contact estimation.
    
    Attributes:
        vertical_left: Vertical force for left foot (N,)
        vertical_right: Vertical force for right foot (N,)
        horizontal_left: Horizontal forces for left foot (N, 2)
        horizontal_right: Horizontal forces for right foot (N, 2)
        cop_left: Center of pressure for left foot (N, 2)
        cop_right: Center of pressure for right foot (N, 2)
        timestamps: Time stamps (N,)
        fps: Sampling rate
        body_weight: Estimated body weight (for normalization)
    """
    vertical_left: np.ndarray
    vertical_right: np.ndarray
    horizontal_left: np.ndarray
    horizontal_right: np.ndarray
    cop_left: Optional[np.ndarray]
    cop_right: Optional[np.ndarray]
    timestamps: np.ndarray
    fps: float
    body_weight: float
    
    def get_total_vertical(self) -> np.ndarray:
        """Get total vertical force (both feet)."""
        return self.vertical_left + self.vertical_right
    
    def normalize_by_weight(self) -> 'ProcessedGRF':
        """Return GRF normalized by body weight."""
        if self.body_weight <= 0:
            return self
        
        return ProcessedGRF(
            vertical_left=self.vertical_left / self.body_weight,
            vertical_right=self.vertical_right / self.body_weight,
            horizontal_left=self.horizontal_left / self.body_weight,
            horizontal_right=self.horizontal_right / self.body_weight,
            cop_left=self.cop_left,
            cop_right=self.cop_right,
            timestamps=self.timestamps,
            fps=self.fps,
            body_weight=1.0  # Normalized
        )


class GRFProcessor:
    """
    Processor for Ground Reaction Force data.
    
    Handles loading, filtering, and preprocessing of GRF measurements
    from force plates or other measurement systems.
    """
    
    # Default thresholds for contact detection
    DEFAULT_FORCE_THRESHOLD = 20.0  # Newtons
    DEFAULT_NORMALIZED_THRESHOLD = 0.05  # Fraction of body weight
    
    def __init__(self,
                 lowpass_cutoff: float = 10.0,
                 force_threshold: float = DEFAULT_FORCE_THRESHOLD,
                 use_normalized_threshold: bool = True):
        """
        Initialize GRF processor.
        
        Args:
            lowpass_cutoff: Low-pass filter cutoff frequency (Hz)
            force_threshold: Force threshold for contact detection (N)
            use_normalized_threshold: Whether to use normalized threshold
        """
        self.lowpass_cutoff = lowpass_cutoff
        self.force_threshold = force_threshold
        self.use_normalized_threshold = use_normalized_threshold
    
    def process(self,
                raw_forces: np.ndarray,
                timestamps: Optional[np.ndarray] = None,
                fps: Optional[float] = None,
                body_weight: Optional[float] = None) -> ProcessedGRF:
        """
        Process raw GRF data.
        
        Args:
            raw_forces: Raw force data
                       - Shape (N, 3) for single foot
                       - Shape (N, 6) for dual foot [left_x,y,z, right_x,y,z]
                       - Shape (N, 2, 3) for dual foot with explicit structure
            timestamps: Optional timestamps
            fps: Sampling rate
            body_weight: Subject body weight for normalization
            
        Returns:
            ProcessedGRF object
        """
        raw_forces = np.asarray(raw_forces)
        
        # Determine data structure and extract forces
        if raw_forces.ndim == 2:
            if raw_forces.shape[1] == 3:
                # Single foot
                left_forces = raw_forces
                right_forces = np.zeros_like(raw_forces)
            elif raw_forces.shape[1] == 6:
                # Dual foot, interleaved
                left_forces = raw_forces[:, :3]
                right_forces = raw_forces[:, 3:6]
            else:
                raise ValueError(f"Unexpected force data shape: {raw_forces.shape}")
        elif raw_forces.ndim == 3:
            # (N, 2, 3) format
            left_forces = raw_forces[:, 0, :]
            right_forces = raw_forces[:, 1, :]
        else:
            raise ValueError(f"Unexpected force data shape: {raw_forces.shape}")
        
        # Determine fps and timestamps
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
            if fps is None:
                fps = 1.0 / np.mean(np.diff(timestamps))
        else:
            if fps is None:
                fps = 100.0  # Default assumption
            timestamps = np.arange(len(raw_forces)) / fps
        
        # Estimate body weight if not provided
        if body_weight is None:
            body_weight = self._estimate_body_weight(left_forces, right_forces)
        
        # Apply low-pass filter
        left_filtered = self._apply_lowpass(left_forces, fps)
        right_filtered = self._apply_lowpass(right_forces, fps)
        
        # Extract components
        vertical_left = left_filtered[:, 2]
        vertical_right = right_filtered[:, 2]
        horizontal_left = left_filtered[:, :2]
        horizontal_right = right_filtered[:, :2]
        
        return ProcessedGRF(
            vertical_left=vertical_left,
            vertical_right=vertical_right,
            horizontal_left=horizontal_left,
            horizontal_right=horizontal_right,
            cop_left=None,  # Would need moment data
            cop_right=None,
            timestamps=timestamps,
            fps=fps,
            body_weight=body_weight
        )
    
    def process_with_cop(self,
                        forces: np.ndarray,
                        moments: np.ndarray,
                        timestamps: Optional[np.ndarray] = None,
                        fps: Optional[float] = None,
                        body_weight: Optional[float] = None,
                        foot_length: float = 0.26) -> ProcessedGRF:
        """
        Process GRF data with moment data to compute COP.
        
        Args:
            forces: Force data (N, 6) or (N, 2, 3)
            moments: Moment data (N, 6) or (N, 2, 3)
            timestamps: Optional timestamps
            fps: Sampling rate
            body_weight: Subject body weight
            foot_length: Foot length for COP calculation
            
        Returns:
            ProcessedGRF with COP data
        """
        # First process forces
        processed = self.process(forces, timestamps, fps, body_weight)
        
        # Compute COP from moments
        moments = np.asarray(moments)
        
        if moments.ndim == 2 and moments.shape[1] == 6:
            left_moments = moments[:, :3]
            right_moments = moments[:, 3:6]
        elif moments.ndim == 3:
            left_moments = moments[:, 0, :]
            right_moments = moments[:, 1, :]
        else:
            return processed
        
        # COP = (My/Fz, -Mx/Fz) for each foot
        # Avoid division by zero
        eps = 1e-6
        
        cop_left = np.zeros((len(forces), 2))
        cop_right = np.zeros((len(forces), 2))
        
        # Left foot COP
        fz_left = processed.vertical_left
        valid_left = np.abs(fz_left) > eps
        cop_left[valid_left, 0] = left_moments[valid_left, 1] / fz_left[valid_left]
        cop_left[valid_left, 1] = -left_moments[valid_left, 0] / fz_left[valid_left]
        
        # Right foot COP
        fz_right = processed.vertical_right
        valid_right = np.abs(fz_right) > eps
        cop_right[valid_right, 0] = right_moments[valid_right, 1] / fz_right[valid_right]
        cop_right[valid_right, 1] = -right_moments[valid_right, 0] / fz_right[valid_right]
        
        # Update processed data
        processed.cop_left = cop_left
        processed.cop_right = cop_right
        
        return processed
    
    def _apply_lowpass(self, 
                       data: np.ndarray, 
                       fps: float) -> np.ndarray:
        """Apply low-pass Butterworth filter."""
        if self.lowpass_cutoff <= 0 or self.lowpass_cutoff >= fps / 2:
            return data
        
        nyquist = fps / 2
        normalized_cutoff = self.lowpass_cutoff / nyquist
        
        # Ensure cutoff is valid
        normalized_cutoff = min(normalized_cutoff, 0.99)
        
        # Design filter
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        
        # Apply filter
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered
    
    def _estimate_body_weight(self,
                              left_forces: np.ndarray,
                              right_forces: np.ndarray) -> float:
        """Estimate body weight from GRF data."""
        total_vertical = left_forces[:, 2] + right_forces[:, 2]
        
        # Use median of peaks as body weight estimate
        # During quiet standing, total vertical force ≈ body weight
        if len(total_vertical) > 0:
            return np.median(total_vertical[total_vertical > 0])
        return 700.0  # Default assumption (70 kg * 9.81)
    
    def detect_contact_frames(self,
                              processed_grf: ProcessedGRF,
                              threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect frames where each foot is in contact.
        
        Args:
            processed_grf: Processed GRF data
            threshold: Force threshold (uses default if None)
            
        Returns:
            Tuple of (left_contact, right_contact) boolean arrays
        """
        if threshold is None:
            if self.use_normalized_threshold:
                threshold = self.DEFAULT_NORMALIZED_THRESHOLD * processed_grf.body_weight
            else:
                threshold = self.force_threshold
        
        left_contact = processed_grf.vertical_left > threshold
        right_contact = processed_grf.vertical_right > threshold
        
        return left_contact, right_contact
    
    def compute_loading_rate(self,
                            vertical_force: np.ndarray,
                            fps: float) -> np.ndarray:
        """
        Compute loading rate (rate of force increase).
        
        Args:
            vertical_force: Vertical GRF time series
            fps: Sampling rate
            
        Returns:
            Loading rate time series
        """
        dt = 1.0 / fps
        loading_rate = np.gradient(vertical_force, dt)
        
        # Only positive values (loading)
        loading_rate = np.maximum(loading_rate, 0)
        
        return loading_rate
    
    def compute_peak_forces(self,
                           processed_grf: ProcessedGRF) -> Dict[str, float]:
        """
        Compute peak force values.
        
        Returns:
            Dictionary with peak forces for each foot
        """
        return {
            'peak_vertical_left': np.max(processed_grf.vertical_left),
            'peak_vertical_right': np.max(processed_grf.vertical_right),
            'peak_horizontal_left': np.max(np.linalg.norm(processed_grf.horizontal_left, axis=1)),
            'peak_horizontal_right': np.max(np.linalg.norm(processed_grf.horizontal_right, axis=1)),
            'peak_total_vertical': np.max(processed_grf.get_total_vertical()),
        }
    
    def find_force_onset(self,
                        vertical_force: np.ndarray,
                        threshold: Optional[float] = None,
                        min_duration_frames: int = 5) -> List[int]:
        """
        Find onset frames where force increases above threshold.
        
        Args:
            vertical_force: Vertical GRF time series
            threshold: Force threshold
            min_duration_frames: Minimum duration for valid contact
            
        Returns:
            List of onset frame indices
        """
        if threshold is None:
            threshold = self.force_threshold
        
        above_threshold = vertical_force > threshold
        
        # Find rising edges
        onsets = []
        in_contact = False
        contact_start = 0
        
        for i, above in enumerate(above_threshold):
            if above and not in_contact:
                in_contact = True
                contact_start = i
            elif not above and in_contact:
                if i - contact_start >= min_duration_frames:
                    onsets.append(contact_start)
                in_contact = False
        
        if in_contact and len(above_threshold) - contact_start >= min_duration_frames:
            onsets.append(contact_start)
        
        return onsets
    
    def find_force_offset(self,
                         vertical_force: np.ndarray,
                         threshold: Optional[float] = None,
                         min_duration_frames: int = 5) -> List[int]:
        """
        Find offset frames where force decreases below threshold.
        
        Args:
            vertical_force: Vertical GRF time series
            threshold: Force threshold
            min_duration_frames: Minimum duration for valid contact
            
        Returns:
            List of offset frame indices
        """
        if threshold is None:
            threshold = self.force_threshold
        
        above_threshold = vertical_force > threshold
        
        # Find falling edges
        offsets = []
        in_contact = False
        
        for i, above in enumerate(above_threshold):
            if above:
                in_contact = True
            elif not above and in_contact:
                offsets.append(i)
                in_contact = False
        
        return offsets
    
    def resample(self,
                processed_grf: ProcessedGRF,
                target_fps: float) -> ProcessedGRF:
        """
        Resample GRF data to different frame rate.
        
        Args:
            processed_grf: Input GRF data
            target_fps: Target frame rate
            
        Returns:
            Resampled ProcessedGRF
        """
        from scipy.interpolate import interp1d
        
        original_fps = processed_grf.fps
        n_original = len(processed_grf.timestamps)
        n_target = int(n_original * target_fps / original_fps)
        
        original_time = processed_grf.timestamps
        target_time = np.linspace(original_time[0], original_time[-1], n_target)
        
        # Interpolate all signals
        interp_vertical_left = interp1d(original_time, processed_grf.vertical_left, 
                                        kind='cubic', fill_value='extrapolate')
        interp_vertical_right = interp1d(original_time, processed_grf.vertical_right,
                                         kind='cubic', fill_value='extrapolate')
        
        return ProcessedGRF(
            vertical_left=interp_vertical_left(target_time),
            vertical_right=interp_vertical_right(target_time),
            horizontal_left=np.zeros((n_target, 2)),  # Simplified
            horizontal_right=np.zeros((n_target, 2)),
            cop_left=None,
            cop_right=None,
            timestamps=target_time,
            fps=target_fps,
            body_weight=processed_grf.body_weight
        )
