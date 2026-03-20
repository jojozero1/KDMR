"""
Tests for contact module.
"""

import pytest
import numpy as np

from kdmr.contact.contact_mode import (
    ContactMode,
    ContactSequence,
    DualContactSequence,
    compute_gait_parameters,
)
from kdmr.contact.grf_processor import GRFProcessor, ProcessedGRF


class TestContactMode:
    """Tests for ContactMode enum."""
    
    def test_is_stance(self):
        """Test stance phase detection."""
        assert ContactMode.HEEL.is_stance()
        assert ContactMode.FLAT.is_stance()
        assert ContactMode.TOE.is_stance()
        assert not ContactMode.SWING.is_stance()
    
    def test_is_swing(self):
        """Test swing phase detection."""
        assert ContactMode.SWING.is_swing()
        assert not ContactMode.FLAT.is_swing()
    
    def test_from_string(self):
        """Test string to enum conversion."""
        assert ContactMode.from_string('heel') == ContactMode.HEEL
        assert ContactMode.from_string('FLAT') == ContactMode.FLAT
        assert ContactMode.from_string('swing') == ContactMode.SWING
        assert ContactMode.from_string('stance') == ContactMode.FLAT  # Alias
    
    def test_get_color(self):
        """Test color retrieval for visualization."""
        color = ContactMode.HEEL.get_color()
        assert len(color) == 4  # RGBA
        assert all(0 <= c <= 1 for c in color)


class TestContactSequence:
    """Tests for ContactSequence class."""
    
    def test_creation(self):
        """Test sequence creation."""
        modes = [ContactMode.HEEL, ContactMode.FLAT, ContactMode.TOE, ContactMode.SWING]
        timestamps = np.array([0.0, 0.1, 0.2, 0.3])
        
        seq = ContactSequence(
            foot_name='left',
            modes=modes,
            timestamps=timestamps
        )
        
        assert len(seq) == 4
        assert seq.foot_name == 'left'
    
    def test_getitem(self):
        """Test indexing."""
        modes = [ContactMode.HEEL, ContactMode.FLAT]
        seq = ContactSequence(foot_name='left', modes=modes)
        
        state = seq[0]
        assert state.mode == ContactMode.HEEL
    
    def test_get_stance_phases(self):
        """Test stance phase detection."""
        modes = [
            ContactMode.HEEL, ContactMode.FLAT, ContactMode.TOE,
            ContactMode.SWING, ContactMode.SWING,
            ContactMode.HEEL, ContactMode.FLAT
        ]
        seq = ContactSequence(foot_name='left', modes=modes)
        
        phases = seq.get_stance_phases()
        
        assert len(phases) == 2  # Two stance phases
        assert phases[0] == (0, 3)  # First stance: indices 0-3
        assert phases[1] == (5, 7)  # Second stance: indices 5-7
    
    def test_get_swing_phases(self):
        """Test swing phase detection."""
        modes = [
            ContactMode.HEEL, ContactMode.FLAT,
            ContactMode.SWING, ContactMode.SWING,
            ContactMode.HEEL
        ]
        seq = ContactSequence(foot_name='left', modes=modes)
        
        phases = seq.get_swing_phases()
        
        assert len(phases) == 1
        assert phases[0] == (2, 4)
    
    def test_duty_factor(self):
        """Test duty factor computation."""
        modes = [
            ContactMode.FLAT, ContactMode.FLAT, ContactMode.FLAT,
            ContactMode.SWING, ContactMode.SWING
        ]
        seq = ContactSequence(foot_name='left', modes=modes)
        
        duty = seq.get_duty_factor()
        
        assert duty == pytest.approx(0.6, abs=0.01)
    
    def test_to_array(self):
        """Test conversion to array."""
        modes = [ContactMode.HEEL, ContactMode.FLAT, ContactMode.TOE, ContactMode.SWING]
        seq = ContactSequence(foot_name='left', modes=modes)
        
        arr = seq.to_array()
        
        assert len(arr) == 4
        assert arr[0] == 0  # HEEL
        assert arr[1] == 1  # FLAT
        assert arr[2] == 2  # TOE
        assert arr[3] == 3  # SWING
    
    def test_from_array(self):
        """Test creation from array."""
        arr = np.array([0, 1, 2, 3])
        
        seq = ContactSequence.from_array('left', arr)
        
        assert len(seq) == 4
        assert seq.modes[0] == ContactMode.HEEL
        assert seq.modes[3] == ContactMode.SWING


class TestDualContactSequence:
    """Tests for DualContactSequence class."""
    
    def test_creation(self):
        """Test dual sequence creation."""
        left_modes = [ContactMode.FLAT, ContactMode.FLAT, ContactMode.SWING]
        right_modes = [ContactMode.SWING, ContactMode.FLAT, ContactMode.FLAT]
        
        dual = DualContactSequence(
            left=ContactSequence('left', left_modes),
            right=ContactSequence('right', right_modes)
        )
        
        assert len(dual) == 3
    
    def test_get_double_support_phases(self):
        """Test double support phase detection."""
        left_modes = [ContactMode.FLAT, ContactMode.FLAT, ContactMode.SWING, ContactMode.FLAT]
        right_modes = [ContactMode.FLAT, ContactMode.SWING, ContactMode.SWING, ContactMode.FLAT]
        
        dual = DualContactSequence(
            left=ContactSequence('left', left_modes),
            right=ContactSequence('right', right_modes)
        )
        
        phases = dual.get_double_support_phases()
        
        assert len(phases) == 2  # Two double support phases
        assert phases[0] == (0, 1)  # First: index 0
        assert phases[1] == (3, 4)  # Second: index 3
    
    def test_get_single_support_phases(self):
        """Test single support phase detection."""
        left_modes = [ContactMode.FLAT, ContactMode.FLAT, ContactMode.SWING]
        right_modes = [ContactMode.SWING, ContactMode.FLAT, ContactMode.FLAT]
        
        dual = DualContactSequence(
            left=ContactSequence('left', left_modes),
            right=ContactSequence('right', right_modes)
        )
        
        phases = dual.get_single_support_phases()
        
        # Should have single support phases
        assert len(phases) >= 1


class TestGRFProcessor:
    """Tests for GRFProcessor class."""
    
    def test_process_single_foot(self):
        """Test processing single foot GRF."""
        processor = GRFProcessor()
        
        # Create synthetic GRF data
        n_samples = 100
        forces = np.zeros((n_samples, 3))
        
        # Simulate stance phase with vertical force
        forces[20:60, 2] = 700.0  # ~70 kg * g
        
        processed = processor.process(forces, fps=100.0)
        
        assert processed.vertical_left.shape == (n_samples,)
        assert processed.fps == 100.0
    
    def test_process_dual_foot(self):
        """Test processing dual foot GRF."""
        processor = GRFProcessor()
        
        n_samples = 100
        # Format: [left_x, left_y, left_z, right_x, right_y, right_z]
        forces = np.zeros((n_samples, 6))
        
        # Left foot stance
        forces[20:60, 2] = 700.0
        # Right foot stance
        forces[50:90, 5] = 700.0
        
        processed = processor.process(forces, fps=100.0)
        
        assert processed.vertical_left.shape == (n_samples,)
        assert processed.vertical_right.shape == (n_samples,)
    
    def test_detect_contact_frames(self):
        """Test contact frame detection."""
        processor = GRFProcessor(force_threshold=50.0)
        
        n_samples = 100
        forces = np.zeros((n_samples, 6))
        forces[20:60, 2] = 700.0  # Left stance
        forces[50:90, 5] = 700.0  # Right stance
        
        processed = processor.process(forces, fps=100.0)
        left_contact, right_contact = processor.detect_contact_frames(processed)
        
        # Check that contact is detected during stance
        assert left_contact[30]  # During left stance
        assert not left_contact[10]  # Before stance
        assert right_contact[70]  # During right stance
    
    def test_lowpass_filter(self):
        """Test low-pass filtering."""
        processor = GRFProcessor(lowpass_cutoff=10.0)
        
        # Create noisy signal
        n_samples = 100
        t = np.linspace(0, 1, n_samples)
        clean_signal = 700.0 * np.sin(2 * np.pi * 2 * t)  # 2 Hz signal
        noise = 50.0 * np.sin(2 * np.pi * 50 * t)  # 50 Hz noise
        
        forces = np.zeros((n_samples, 3))
        forces[:, 2] = np.abs(clean_signal) + noise
        
        processed = processor.process(forces, fps=100.0)
        
        # Filtered signal should be smoother
        original_std = np.std(forces[:, 2])
        filtered_std = np.std(processed.vertical_left)
        
        # Filtered should have lower high-frequency content
        assert filtered_std < original_std


class TestComputeGaitParameters:
    """Tests for gait parameter computation."""
    
    def test_normal_walking(self):
        """Test gait parameters for normal walking pattern."""
        # Create typical walking pattern
        fps = 100
        n_frames = 200  # 2 seconds
        
        left_modes = []
        right_modes = []
        
        for i in range(n_frames):
            t = i / fps
            # Alternating stance/swing
            if (t % 1.0) < 0.6:  # 60% stance
                left_modes.append(ContactMode.FLAT)
            else:
                left_modes.append(ContactMode.SWING)
            
            if ((t + 0.5) % 1.0) < 0.6:  # 50% phase shift
                right_modes.append(ContactMode.FLAT)
            else:
                right_modes.append(ContactMode.SWING)
        
        dual = DualContactSequence(
            left=ContactSequence('left', left_modes),
            right=ContactSequence('right', right_modes)
        )
        
        params = compute_gait_parameters(dual, fps)
        
        assert 'stride_time' in params
        assert 'duty_factor' in params
        assert 'cadence' in params
        
        # Duty factor should be around 0.6
        assert 0.5 < params['duty_factor'] < 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
