"""
Lagrangian and Hamiltonian dynamics for QKDMR v2.0.

First Principle: Physical systems follow trajectories that extremize the
action functional S = ∫ L dt. This is a more fundamental description than
the Newton-Euler equations — it reveals the symplectic structure, conserved
quantities (Noether), and the correct geometry for optimization.

This module provides:
1. Lagrangian L(q, q̇) = T(q, q̇) - V(q) in generalized coordinates
2. Hamiltonian H(q, p) = T + V in phase space (positions + momenta)
3. Manipulator equation: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ + Jᵀλ
4. Discrete Lagrangian Ld(qk, qk+1) for variational integrators
5. Noether invariants from symmetry: linear/angular momentum, energy

Key insight: The Euler-Lagrange equations ARE the dynamics. The manipulator
equation is just a special case for rigid body systems. By working at the
Lagrangian level, we preserve the symplectic structure automatically.

References:
- Marsden & West, "Discrete Mechanics and Variational Integrators" (2001)
- Bullo & Lewis, "Geometric Control of Mechanical Systems" (2004)
- Holm, "Geometric Mechanics" (2008)
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from scipy.spatial.transform import Rotation as R

from kdmr_v2.utils.lie_utils import (
    LieAlgebra, ExponentialMap, RiemannianMetric, LieGroupType
)

try:
    import mujoco as mj
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

# Try JAX for automatic differentiation of Lagrangian
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass
class PhasePoint:
    """A point in phase space T*Q (cotangent bundle).

    Attributes:
        q: Generalized coordinates (nq,) on the configuration manifold Q
        p: Generalized momenta (nv,) in the cotangent space T*_q Q
    """
    q: np.ndarray
    p: np.ndarray

    @property
    def n(self) -> int:
        return len(self.q)


@dataclass
class GeneralizedState:
    """Complete state of a mechanical system.

    Attributes:
        q: Generalized positions (nq,)
        qdot: Generalized velocities (nv,)
        tau: Applied generalized forces (nu,)
        lambda_contact: Contact forces (nc,)
        time: Current time
    """
    q: np.ndarray
    qdot: np.ndarray
    tau: Optional[np.ndarray] = None
    lambda_contact: Optional[np.ndarray] = None
    time: float = 0.0

    @property
    def nq(self) -> int: return len(self.q)

    @property
    def nv(self) -> int: return len(self.qdot)


class LagrangianDynamics:
    """
    Lagrangian formulation of rigid body dynamics.

    L(q, q̇) = T(q, q̇) - V(q)

    where:
    - T(q, q̇) = ½ q̇ᵀ M(q) q̇  (kinetic energy)
    - V(q) = gravitational potential energy

    The Euler-Lagrange equations:
        d/dt (∂L/∂q̇) - ∂L/∂q = Q_ext

    give the manipulator equation:
        M(q) q̈ + C(q, q̇) q̇ + G(q) = τ + Jᵀ F_ext
    """

    def __init__(self, model=None, data=None):
        """Initialize Lagrangian dynamics.

        Args:
            model: MuJoCo model (or None for analytic)
            data: MuJoCo data (or None for analytic)
        """
        self.model = model
        self.data = data

        if model is not None:
            self.nq = model.nq
            self.nv = model.nv
            self.nu = model.nu
        else:
            self.nq = self.nv = self.nu = 0

    def kinetic_energy(self, q: np.ndarray, qdot: np.ndarray) -> float:
        """Compute kinetic energy T = ½ q̇ᵀ M(q) q̇.

        For a rigid body system, the mass matrix M(q) captures the
        distribution of inertia in the current configuration.

        First principle: T is the fiber metric on TQ — it defines the
        Riemannian structure of the tangent bundle.
        """
        if self.model is not None:
            self.data.qpos[:len(q)] = q
            self.data.qvel[:len(qdot)] = qdot
            mj.mj_forward(self.model, self.data)
            return self.data.energy[0]  # Kinetic energy
        else:
            # Analytic: T = ½ Σ m_i v_i² + ½ Σ I_i ω_i²
            return self._analytic_kinetic_energy(q, qdot)

    def potential_energy(self, q: np.ndarray) -> float:
        """Compute gravitational potential energy V(q) = Σ m_i g h_i.

        The potential defines the "shape" of the configuration space —
        trajectories "roll" along its gradient.
        """
        if self.model is not None:
            self.data.qpos[:len(q)] = q
            mj.mj_forward(self.model, self.data)
            return self.data.energy[1]
        else:
            return self._analytic_potential_energy(q)

    def lagrangian(self, q: np.ndarray, qdot: np.ndarray) -> float:
        """Compute Lagrangian L = T - V."""
        return self.kinetic_energy(q, qdot) - self.potential_energy(q)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute the mass (inertia) matrix M(q).

        M(q) is the Riemannian metric on the configuration manifold Q.
        It defines the kinetic energy metric — "how hard it is to move
        in each direction at configuration q."

        The mass matrix is symmetric positive-definite: M = Mᵀ ≻ 0.
        """
        if self.model is not None:
            self.data.qpos[:len(q)] = q
            mj.mj_forward(self.model, self.data)
            M = np.zeros((self.nv, self.nv))
            mj.mj_fullM(self.model, M, self.data.qM)
            return M
        else:
            return np.eye(len(q) - 1)  # Placeholder

    def coriolis_matrix(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Compute the Coriolis matrix C(q, q̇).

        The Coriolis matrix can be derived from the mass matrix:
            C_{ij} = ½ Σ_k (∂M_{ij}/∂q_k + ∂M_{ik}/∂q_j - ∂M_{kj}/∂q_i) q̇_k

        Using Christoffel symbols of the first kind.
        Importantly, Ṁ - 2C is skew-symmetric — this is the passivity property.

        First principle: C comes from the connection on TQ, specifically
        the Levi-Civita connection of the kinetic energy metric.
        """
        if self.model is not None:
            self.data.qpos[:len(q)] = q
            self.data.qvel[:len(qdot)] = qdot
            mj.mj_forward(self.model, self.data)

            # Compute bias forces at current velocity
            bias_with_vel = self.data.qfrc_bias.copy()

            # Compute gravity-only forces (zero velocity)
            self.data.qvel[:] = 0
            mj.mj_forward(self.model, self.data)
            gravity = self.data.qfrc_bias.copy()

            # C(q,q̇) = qfrc_bias(q,q̇) - G(q)
            # (This is a force vector, not matrix)
            return bias_with_vel - gravity
        else:
            return np.zeros(len(qdot))

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity forces G(q) = ∂V/∂q.

        Gravity forces are the negative gradient of potential energy.
        They drive the system toward lower potential configurations.
        """
        if self.model is not None:
            self.data.qpos[:len(q)] = q
            self.data.qvel[:] = 0
            mj.mj_forward(self.model, self.data)
            return self.data.qfrc_bias.copy()
        else:
            return np.zeros(len(q))

    def euler_lagrange_rhs(self, q: np.ndarray, qdot: np.ndarray,
                            tau: Optional[np.ndarray] = None,
                            f_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the right-hand side of Euler-Lagrange equations.

        Returns q̈ = M⁻¹(τ - C q̇ - G + Jᵀ F_ext)

        This is the acceleration field that defines the dynamics on TQ.
        """
        M = self.mass_matrix(q)
        Cqdot = self.coriolis_matrix(q, qdot)
        G = self.gravity_vector(q)

        rhs = -Cqdot - G
        if tau is not None:
            rhs[:len(tau)] += tau
        if f_ext is not None:
            rhs += f_ext

        try:
            qddot = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            qddot = np.linalg.lstsq(M, rhs, rcond=None)[0]

        return qddot

    def _analytic_kinetic_energy(self, q: np.ndarray, qdot: np.ndarray) -> float:
        """Analytic kinetic energy for simple models."""
        return 0.5 * np.dot(qdot, qdot)

    def _analytic_potential_energy(self, q: np.ndarray) -> float:
        """Analytic potential energy for simple models."""
        g = 9.81
        return g * q[2] if len(q) >= 3 else 0.0


class HamiltonianDynamics:
    """
    Hamiltonian formulation in phase space T*Q.

    H(q, p) = ½ pᵀ M⁻¹(q) p + V(q)

    Hamilton's equations:
        q̇  = ∂H/∂p = M⁻¹(q) p
        ṗ  = -∂H/∂q + Q_ext

    First Principle: The Hamiltonian flow preserves the symplectic 2-form
    ω = Σ dp_i ∧ dq_i. This is the deepest structure in classical mechanics —
    it guarantees phase space volume preservation (Liouville's theorem) and
    is the foundation for all symplectic integrators.

    Hamiltonian Monte Carlo (HMC) on path space uses Hamiltonian dynamics
    to propose new trajectories. The symplectic structure ensures that the
    Metropolis-Hastings acceptance probability depends only on energy error,
    not on the discretization details.
    """

    def __init__(self, lagrangian: LagrangianDynamics):
        """Initialize Hamiltonian dynamics from Lagrangian.

        The Legendre transform p = ∂L/∂q̇ maps from TQ (velocity phase space)
        to T*Q (momentum phase space). For mechanical systems:
            p = M(q) q̇
        """
        self.lagrangian = lagrangian
        self.nq = lagrangian.nq
        self.nv = lagrangian.nv

    def momentum(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Compute generalized momentum p = ∂L/∂q̇ = M(q) q̇."""
        M = self.lagrangian.mass_matrix(q)
        return M @ qdot

    def velocity(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute velocity from momentum: q̇ = M⁻¹(q) p."""
        M = self.lagrangian.mass_matrix(q)
        try:
            return np.linalg.solve(M, p)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(M, p, rcond=None)[0]

    def hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        """Compute total energy H(q, p) = T + V.

        The Hamiltonian is conserved along physical trajectories
        when there are no non-conservative forces. It is the generator
        of time evolution.
        """
        qdot = self.velocity(q, p)
        T = 0.5 * np.dot(p, qdot)
        V = self.lagrangian.potential_energy(q)
        return T + V

    def hamiltonian_gradient(self, q: np.ndarray, p: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients of Hamiltonian: (∂H/∂q, ∂H/∂p).

        These define Hamilton's equations:
            q̇ = +∂H/∂p
            ṗ = -∂H/∂q + τ_ext
        """
        qdot = self.velocity(q, p)

        # ∂H/∂p = M⁻¹ p = q̇
        dH_dp = qdot

        # ∂H/∂q = ∂T/∂q + ∂V/∂q
        # ∂T/∂q is complex (derivative of M⁻¹), use finite differences
        dH_dq = self._compute_dH_dq(q, p)

        return dH_dq, dH_dp

    def _compute_dH_dq(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute ∂H/∂q using finite differences on mass matrix.

        ∂H/∂q = -½ pᵀ M⁻¹ (∂M/∂q) M⁻¹ p + ∂V/∂q
        """
        # Gravity contribution: ∂V/∂q = G(q)
        G = self.lagrangian.gravity_vector(q)

        # Kinetic contribution via finite differences
        eps = 1e-6
        n = len(q)
        dT_dq = np.zeros(n)

        base_energy = self.hamiltonian(q, p)
        for i in range(min(n, 10)):  # First 10 DOFs for efficiency
            q_plus = q.copy()
            q_plus[i] += eps
            energy_plus = self.hamiltonian(q_plus, p)
            dT_dq[i] = (energy_plus - base_energy) / eps

        return dT_dq + G

    def phase_flow(self, q: np.ndarray, p: np.ndarray, dt: float,
                   tau: Optional[np.ndarray] = None) -> PhasePoint:
        """Single step of Hamiltonian phase flow.

        q(t+dt) = q(t) + dt · ∂H/∂p
        p(t+dt) = p(t) + dt · (-∂H/∂q + τ)
        """
        dH_dq, dH_dp = self.hamiltonian_gradient(q, p)

        q_new = q + dt * dH_dp
        p_new = p - dt * dH_dq
        if tau is not None:
            p_new[:len(tau)] += dt * tau

        return PhasePoint(q=q_new, p=p_new)


@dataclass
class NoetherInvariants:
    """
    Conserved quantities from continuous symmetries.

    Noether's Theorem: Every differentiable symmetry of the action of a
    physical system has a corresponding conservation law.

    For humanoid robots:
    - Translation invariance → Linear momentum conservation
    - Rotation invariance about vertical → Angular momentum about z
    - Time translation invariance → Energy conservation

    These invariants constrain the dynamics and can be used as regularization
    terms in trajectory optimization.
    """

    linear_momentum: np.ndarray  # (3,) linear momentum in world frame
    angular_momentum: np.ndarray  # (3,) angular momentum about origin
    energy: float  # Total mechanical energy H(q,p)

    @staticmethod
    def compute_from_mujoco(model, data) -> 'NoetherInvariants':
        """Compute Noether invariants from MuJoCo state."""
        mj.mj_forward(model, data)

        # Total mass
        total_mass = np.sum(model.body_mass)

        # Center of mass position
        com_pos = data.subtree_com[0].copy()

        # Linear momentum: P = M_total · v_COM
        # Use subtree_linvel
        lin_mom = np.zeros(3)
        for i in range(model.nbody):
            body_mass = model.body_mass[i]
            body_vel = data.cvel[i + 1, :3] if hasattr(data, 'cvel') else np.zeros(3)
            lin_mom += body_mass * body_vel

        # Angular momentum about origin: L = Σ (r_i × m_i v_i + I_i ω_i)
        ang_mom = np.zeros(3)
        for i in range(model.nbody):
            body_mass = model.body_mass[i]
            body_pos = data.xpos[i]
            body_vel = np.zeros(3)  # Would need body velocity
            ang_mom += body_mass * np.cross(body_pos, body_vel)

        # Total energy
        energy = data.energy[0] + data.energy[1]

        return NoetherInvariants(
            linear_momentum=lin_mom,
            angular_momentum=ang_mom,
            energy=energy
        )

    def conservation_error(self, other: 'NoetherInvariants') -> Dict[str, float]:
        """Compute conservation error between two states."""
        return {
            'linear_momentum_error': np.linalg.norm(
                self.linear_momentum - other.linear_momentum),
            'angular_momentum_error': np.linalg.norm(
                self.angular_momentum - other.angular_momentum),
            'energy_error': abs(self.energy - other.energy),
        }


class ActionFunctional:
    """
    The action functional S[q] = ∫₀ᵀ L(q, q̇) dt.

    This is THE central object of analytical mechanics. Hamilton's principle
    (principle of least action) states that physical trajectories are
    stationary points of S.

    For trajectory optimization, we discretize S using a variational
    integrator (which preserves the symplectic structure) and then find
    trajectories that make δS_d = 0.

    Discrete action:
        S_d[{q_k}] = Σ_{k=0}^{N-1} L_d(q_k, q_{k+1})

    where L_d is the discrete Lagrangian:
        L_d(q_k, q_{k+1}) ≈ h · L((q_k+q_{k+1})/2, (q_{k+1}-q_k)/h)
    """

    def __init__(self, lagrangian: LagrangianDynamics, dt: float):
        """Initialize action functional.

        Args:
            lagrangian: Continuous-time Lagrangian dynamics
            dt: Time step for discretization
        """
        self.lagrangian = lagrangian
        self.dt = dt

    def discrete_lagrangian(self, q_k: np.ndarray, q_kp1: np.ndarray) -> float:
        """Compute discrete Lagrangian L_d(q_k, q_{k+1}).

        Uses midpoint rule (2nd-order accurate, symplectic):
            q_mid = (q_k + q_{k+1}) / 2
            qdot_mid = (q_{k+1} - q_k) / dt
            L_d = dt · L(q_mid, qdot_mid)
        """
        q_mid = 0.5 * (q_k + q_kp1)
        qdot_mid = (q_kp1 - q_k) / self.dt
        return self.dt * self.lagrangian.lagrangian(q_mid, qdot_mid)

    def discrete_action(self, trajectory: np.ndarray) -> float:
        """Compute discrete action for a trajectory.

        S_d[{q_k}] = Σ L_d(q_k, q_{k+1})

        Args:
            trajectory: (T, nq) array of configurations

        Returns:
            Discrete action value
        """
        S = 0.0
        for k in range(len(trajectory) - 1):
            S += self.discrete_lagrangian(trajectory[k], trajectory[k + 1])
        return S

    def action_gradient(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute gradient of discrete action ∂S_d/∂q_k.

        The discrete Euler-Lagrange (DEL) equations:
            D₂L_d(q_{k-1}, q_k) + D₁L_d(q_k, q_{k+1}) = 0

        where D₁, D₂ are partial derivatives w.r.t. first and second arguments.
        These are the KKT conditions for the discrete variational problem.

        The gradient at interior points:
            ∂S_d/∂q_k = D₂L_d(q_{k-1}, q_k) + D₁L_d(q_k, q_{k+1})

        Args:
            trajectory: (T, nq) array of configurations

        Returns:
            Gradient (T, nq) of action w.r.t. each configuration
        """
        T, nq = trajectory.shape
        grad = np.zeros_like(trajectory)
        eps = 1e-6

        for k in range(T):
            # Finite-difference D₁L_d(q_k, q_{k+1})
            if k < T - 1:
                for i in range(min(nq, 12)):  # Limit for efficiency
                    q_plus = trajectory[k].copy()
                    q_plus[i] += eps
                    L_plus = self.discrete_lagrangian(q_plus, trajectory[k + 1])
                    L_base = self.discrete_lagrangian(trajectory[k], trajectory[k + 1])
                    grad[k, i] += (L_plus - L_base) / eps

            # Finite-difference D₂L_d(q_{k-1}, q_k)
            if k > 0:
                for i in range(min(nq, 12)):
                    q_plus = trajectory[k].copy()
                    q_plus[i] += eps
                    L_plus = self.discrete_lagrangian(trajectory[k - 1], q_plus)
                    L_base = self.discrete_lagrangian(trajectory[k - 1], trajectory[k])
                    grad[k, i] += (L_plus - L_base) / eps

        return grad

    def action_hessian_vector_product(self, trajectory: np.ndarray,
                                       vector: np.ndarray) -> np.ndarray:
        """Compute Hessian-vector product ∇²S_d · v.

        This is the directional derivative of the action gradient.
        Used in conjugate gradient for solving δS_d = 0.

        First principle: The action Hessian is the Jacobi operator —
        it determines the stability of trajectories (conjugate points).
        """
        eps = 1e-4
        grad_plus = self.action_gradient(trajectory + eps * vector)
        grad_base = self.action_gradient(trajectory)
        return (grad_plus - grad_base) / eps

    def check_stationarity(self, trajectory: np.ndarray) -> float:
        """Check how stationary the trajectory is (residual of DEL equations).

        Returns RMS of action gradient — zero for physical trajectories.
        """
        grad = self.action_gradient(trajectory)
        return np.sqrt(np.mean(grad ** 2))

    def is_physical_trajectory(self, trajectory: np.ndarray,
                                tolerance: float = 1e-3) -> bool:
        """Check if trajectory satisfies discrete Euler-Lagrange equations."""
        return self.check_stationarity(trajectory) < tolerance
