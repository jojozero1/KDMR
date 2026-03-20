"""
Tests for optimizer module.
"""

import pytest
import numpy as np

from kdmr.core.cost_functions import (
    CostFunctions,
    CostConfig,
    TrackingCost,
    ControlEffortCost,
    SmoothnessCost,
    JointLimitCost,
)
from kdmr.core.scp_ddp_solver import (
    SCPDDPConfig,
    QuadraticCostApprox,
    LinearizedDynamics,
)


class TestTrackingCost:
    """Tests for tracking cost function."""
    
    def test_identical_states(self):
        """Test cost when states are identical."""
        cost = TrackingCost(weight_pos=100.0, weight_rot=10.0)
        
        state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        reference = state.copy()
        
        assert cost.compute(state, reference) == 0.0
    
    def test_position_error(self):
        """Test position error computation."""
        cost = TrackingCost(weight_pos=100.0, weight_rot=0.0)
        
        state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        reference = np.array([0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        c = cost.compute(state, reference)
        
        # Expected: 100 * 0.1^2 = 1.0
        assert c == pytest.approx(1.0, rel=0.1)
    
    def test_gradient(self):
        """Test gradient computation."""
        cost = TrackingCost(weight_pos=100.0, weight_rot=10.0)
        
        state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        reference = np.array([0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        grad, _ = cost.compute_gradient(state, reference)
        
        # Gradient should point toward reference
        assert grad[0] != 0  # Position gradient


class TestControlEffortCost:
    """Tests for control effort cost."""
    
    def test_zero_control(self):
        """Test cost with zero control."""
        cost = ControlEffortCost(weight=0.01)
        
        state = np.zeros(10)
        control = np.zeros(5)
        
        assert cost.compute(state, control) == 0.0
    
    def test_nonzero_control(self):
        """Test cost with non-zero control."""
        cost = ControlEffortCost(weight=0.01)
        
        state = np.zeros(10)
        control = np.ones(5) * 10.0  # Torque of 10 Nm each
        
        c = cost.compute(state, control)
        
        # Expected: 0.01 * 5 * 100 = 5.0
        assert c == pytest.approx(5.0, rel=0.1)
    
    def test_gradient(self):
        """Test gradient computation."""
        cost = ControlEffortCost(weight=0.01)
        
        state = np.zeros(10)
        control = np.ones(5) * 10.0
        
        state_grad, control_grad = cost.compute_gradient(state, control)
        
        assert np.all(state_grad == 0)
        assert control_grad is not None
        assert len(control_grad) == 5


class TestSmoothnessCost:
    """Tests for smoothness cost."""
    
    def test_constant_trajectory(self):
        """Test cost for constant (smooth) trajectory."""
        cost = SmoothnessCost(weight=0.1, order=2)
        
        # Constant trajectory
        trajectory = np.ones((10, 5))
        
        c = cost.compute(trajectory, dt=0.01)
        
        assert c == 0.0
    
    def test_varying_trajectory(self):
        """Test cost for varying trajectory."""
        cost = SmoothnessCost(weight=0.1, order=2)
        
        # Varying trajectory
        trajectory = np.sin(np.linspace(0, 2*np.pi, 10)).reshape(-1, 1)
        trajectory = np.tile(trajectory, (1, 5))
        
        c = cost.compute(trajectory, dt=0.01)
        
        assert c > 0


class TestJointLimitCost:
    """Tests for joint limit cost."""
    
    def test_within_limits(self):
        """Test cost when within limits."""
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        
        cost = JointLimitCost(lower, upper, weight=1000.0)
        
        joints = np.array([0.0, 0.0])
        
        assert cost.compute(joints) == 0.0
    
    def test_lower_violation(self):
        """Test lower limit violation."""
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        
        cost = JointLimitCost(lower, upper, weight=1000.0)
        
        joints = np.array([-1.5, 0.0])
        
        assert cost.compute(joints) > 0
    
    def test_upper_violation(self):
        """Test upper limit violation."""
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        
        cost = JointLimitCost(lower, upper, weight=1000.0)
        
        joints = np.array([0.0, 1.5])
        
        assert cost.compute(joints) > 0
    
    def test_gradient(self):
        """Test gradient computation."""
        lower = np.array([-1.0])
        upper = np.array([1.0])
        
        cost = JointLimitCost(lower, upper, weight=1000.0, margin=0.1)
        
        # At limit boundary
        joints = np.array([-0.9])  # Near lower limit
        grad = cost.compute_gradient(joints)
        
        assert grad[0] != 0


class TestCostFunctions:
    """Tests for CostFunctions container."""
    
    def test_total_cost(self):
        """Test total cost computation."""
        config = CostConfig(
            tracking_pos=100.0,
            control_effort=0.01
        )
        
        cost_funcs = CostFunctions(config)
        
        state = np.zeros(10)
        state[2] = 1.0  # Height
        state[3] = 1.0  # Quaternion w
        control = np.ones(5) * 10.0
        reference = state.copy()
        reference[0] = 0.1  # Position offset
        
        total = cost_funcs.compute_total_cost(
            state=state,
            control=control,
            reference=reference
        )
        
        assert total > 0
    
    def test_total_gradient(self):
        """Test total gradient computation."""
        config = CostConfig()
        cost_funcs = CostFunctions(config)
        
        state = np.zeros(10)
        state[2] = 1.0
        state[3] = 1.0
        control = np.ones(5) * 10.0
        reference = state.copy()
        reference[0] = 0.1
        
        state_grad, control_grad = cost_funcs.compute_total_gradient(
            state=state,
            control=control,
            reference=reference
        )
        
        assert len(state_grad) == 10
        assert control_grad is not None
        assert len(control_grad) == 5


class TestSCPDDPConfig:
    """Tests for SCP-DDP configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SCPDDPConfig()
        
        assert config.max_scp_iterations == 10
        assert config.max_ddp_iterations == 50
        assert config.regularization > 0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SCPDDPConfig(
            max_scp_iterations=20,
            max_ddp_iterations=100,
            regularization=1e-4
        )
        
        assert config.max_scp_iterations == 20
        assert config.max_ddp_iterations == 100
        assert config.regularization == 1e-4


class TestLinearizedDynamics:
    """Tests for linearized dynamics model."""
    
    def test_prediction(self):
        """Test state prediction."""
        n = 10  # State dimension
        m = 5   # Control dimension
        
        A = np.eye(n)  # Simple identity
        B = np.eye(n, m)  # Simple control matrix
        d = np.zeros(n)
        
        dynamics = LinearizedDynamics(A, B, d)
        
        x = np.ones(n)
        u = np.zeros(m)
        
        x_next = dynamics.predict(x, u)
        
        assert np.allclose(x_next, x)  # Identity dynamics


class TestQuadraticCostApprox:
    """Tests for quadratic cost approximation."""
    
    def test_creation(self):
        """Test creation of quadratic approximation."""
        n = 10
        m = 5
        
        Q_xx = np.eye(n)
        Q_uu = np.eye(m)
        Q_xu = np.zeros((n, m))
        q_x = np.ones(n)
        q_u = np.ones(m)
        
        approx = QuadraticCostApprox(Q_xx, Q_uu, Q_xu, q_x, q_u)
        
        assert approx.Q_xx.shape == (n, n)
        assert approx.Q_uu.shape == (m, m)
        assert approx.Q_ux.shape == (m, n)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
