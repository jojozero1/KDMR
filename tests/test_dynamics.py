"""
Tests for dynamics module.
"""

import pytest
import numpy as np

from kdmr.dynamics.constraints import (
    JointLimitConstraint,
    TorqueLimitConstraint,
    FrictionConeConstraint,
    ContactComplementarityConstraint,
    ConstraintSet,
)


class TestJointLimitConstraint:
    """Tests for joint limit constraint."""
    
    def test_within_limits(self):
        """Test that constraint is satisfied within limits."""
        lower = np.array([-1.0, -0.5, -1.0])
        upper = np.array([1.0, 0.5, 1.0])
        
        constraint = JointLimitConstraint(lower, upper)
        
        # Within limits
        q = np.array([0.0, 0.0, 0.0])
        assert constraint.compute(q) == 0.0
        
        q = np.array([0.5, 0.2, -0.5])
        assert constraint.compute(q) == 0.0
    
    def test_violation_lower(self):
        """Test lower limit violation."""
        lower = np.array([-1.0, -0.5, -1.0])
        upper = np.array([1.0, 0.5, 1.0])
        
        constraint = JointLimitConstraint(lower, upper)
        
        # Violate lower limit
        q = np.array([-1.5, 0.0, 0.0])
        violation = constraint.compute(q)
        assert violation > 0
    
    def test_violation_upper(self):
        """Test upper limit violation."""
        lower = np.array([-1.0, -0.5, -1.0])
        upper = np.array([1.0, 0.5, 1.0])
        
        constraint = JointLimitConstraint(lower, upper)
        
        # Violate upper limit
        q = np.array([0.0, 1.0, 0.0])
        violation = constraint.compute(q)
        assert violation > 0
    
    def test_gradient(self):
        """Test gradient computation."""
        lower = np.array([-1.0])
        upper = np.array([1.0])
        
        constraint = JointLimitConstraint(lower, upper)
        
        # At limit boundary
        q = np.array([-1.0])
        grad, _ = constraint.compute_gradient(q)
        assert grad[0] != 0
        
        # Far from limits
        q = np.array([0.0])
        grad, _ = constraint.compute_gradient(q)
        assert grad[0] == 0


class TestTorqueLimitConstraint:
    """Tests for torque limit constraint."""
    
    def test_within_limits(self):
        """Test constraint satisfaction within limits."""
        limits = np.array([10.0, 20.0, 30.0])
        constraint = TorqueLimitConstraint(limits)
        
        tau = np.array([5.0, 10.0, 15.0])
        assert constraint.compute(tau) == 0.0
    
    def test_violation(self):
        """Test torque limit violation."""
        limits = np.array([10.0, 20.0, 30.0])
        constraint = TorqueLimitConstraint(limits)
        
        tau = np.array([15.0, 10.0, 15.0])
        violation = constraint.compute(tau)
        assert violation > 0
    
    def test_negative_violation(self):
        """Test negative torque violation."""
        limits = np.array([10.0, 20.0, 30.0])
        constraint = TorqueLimitConstraint(limits)
        
        tau = np.array([-15.0, 10.0, 15.0])
        violation = constraint.compute(tau)
        assert violation > 0


class TestFrictionConeConstraint:
    """Tests for friction cone constraint."""
    
    def test_normal_force_only(self):
        """Test pure normal force (satisfies constraint)."""
        constraint = FrictionConeConstraint(friction_coef=1.0)
        
        force = np.array([100.0, 0.0, 0.0])  # Normal only
        violation = constraint.compute(force)
        assert violation == 0.0
    
    def test_within_cone(self):
        """Test force within friction cone."""
        constraint = FrictionConeConstraint(friction_coef=1.0)
        
        force = np.array([100.0, 50.0, 50.0])  # Within cone
        violation = constraint.compute(force)
        assert violation == 0.0
    
    def test_outside_cone(self):
        """Test force outside friction cone."""
        constraint = FrictionConeConstraint(friction_coef=0.5)
        
        force = np.array([100.0, 80.0, 0.0])  # Outside cone
        violation = constraint.compute(force)
        assert violation > 0
    
    def test_negative_normal(self):
        """Test negative normal force (tension)."""
        constraint = FrictionConeConstraint(friction_coef=1.0)
        
        force = np.array([-10.0, 0.0, 0.0])  # Tension
        violation = constraint.compute(force)
        assert violation > 0
    
    def test_projection(self):
        """Test projection onto friction cone."""
        constraint = FrictionConeConstraint(friction_coef=1.0)
        
        force = np.array([100.0, 150.0, 0.0])  # Outside cone
        projected = constraint.project_to_cone(force)
        
        # Check that projected force is within cone
        tangent_mag = np.linalg.norm(projected[1:3])
        assert tangent_mag <= projected[0] * 1.01  # Allow small numerical error


class TestContactComplementarityConstraint:
    """Tests for contact complementarity constraint."""
    
    def test_contact(self):
        """Test during contact (distance=0, force>0)."""
        constraint = ContactComplementarityConstraint()
        
        violation = constraint.compute(distance=0.0, force=100.0)
        assert violation == 0.0
    
    def test_separation(self):
        """Test during separation (distance>0, force=0)."""
        constraint = ContactComplementarityConstraint()
        
        violation = constraint.compute(distance=0.1, force=0.0)
        assert violation == 0.0
    
    def test_penetration(self):
        """Test penetration (distance<0)."""
        constraint = ContactComplementarityConstraint()
        
        violation = constraint.compute(distance=-0.01, force=0.0)
        assert violation > 0
    
    def test_complementarity_violation(self):
        """Test complementarity violation (both non-zero)."""
        constraint = ContactComplementarityConstraint()
        
        violation = constraint.compute(distance=0.1, force=100.0)
        assert violation > 0


class TestConstraintSet:
    """Tests for constraint set."""
    
    def test_empty_set(self):
        """Test empty constraint set."""
        cs = ConstraintSet()
        
        state = np.zeros(10)
        control = np.zeros(5)
        
        assert cs.compute_total_violation(state, control) == 0.0
    
    def test_single_constraint(self):
        """Test single constraint in set."""
        cs = ConstraintSet()
        
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        cs.add_constraint('joint_limit', JointLimitConstraint(lower, upper))
        
        # Within limits
        state = np.array([0.0, 0.0])
        assert cs.compute_total_violation(state) == 0.0
        
        # Outside limits
        state = np.array([2.0, 0.0])
        assert cs.compute_total_violation(state) > 0
    
    def test_multiple_constraints(self):
        """Test multiple constraints in set."""
        cs = ConstraintSet()
        
        lower = np.array([-1.0])
        upper = np.array([1.0])
        cs.add_constraint('joint_limit', JointLimitConstraint(lower, upper), weight=1.0)
        
        torque_limits = np.array([10.0])
        cs.add_constraint('torque_limit', TorqueLimitConstraint(torque_limits), weight=2.0)
        
        state = np.array([2.0])  # Violates joint limit
        control = np.array([15.0])  # Violates torque limit
        
        violations = cs.compute_violations(state, control)
        
        assert 'joint_limit' in violations
        assert 'torque_limit' in violations
        assert violations['joint_limit'].is_violated
        assert violations['torque_limit'].is_violated


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
