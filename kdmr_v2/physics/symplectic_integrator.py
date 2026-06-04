"""
Symplectic integrators for QKDMR v2.0.

First Principle: The exact flow of a Hamiltonian system preserves the
symplectic 2-form ω = Σ dp_i ∧ dq_i and the Hamiltonian H itself. Standard
integrators (RK4, forward Euler) violate both — they inject/remove energy
artificially, leading to unstable or inaccurate long-time simulations.

Symplectic integrators ARE the exact flow of a nearby "modified Hamiltonian"
H̃ = H + hᵖ ΔH + O(h^{p+1}). This means:
- Energy error is BOUNDED for all time (no secular drift)
- Phase space volume is preserved (Liouville's theorem holds exactly)
- Angular momentum is preserved for rotationally-invariant systems

This module provides:
1. Störmer-Verlet (2nd order symplectic)
2. Yoshida composition → 4th, 6th, 8th order
3. Variational integrator from discrete Lagrangian
4. Splitting methods for separable Hamiltonians H = T(p) + V(q)

Reference: Hairer, Lubich, & Wanner, "Geometric Numerical Integration" (2006)
"""

import numpy as np
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum, auto

from kdmr_v2.physics.lagrangian_dynamics import (
    LagrangianDynamics, HamiltonianDynamics, PhasePoint
)


class IntegratorOrder(Enum):
    """Symplectic integrator orders."""
    VERLET = 2
    YOSHIDA4 = 4
    YOSHIDA6 = 6
    YOSHIDA8 = 8


# Yoshida composition coefficients for constructing higher-order
# symplectic integrators from the 2nd-order Verlet method.
#
# The composition: Φ_h^{(2n)} = Φ_{w₁h}^{(2)} ∘ Φ_{w₂h}^{(2)} ∘ ... ∘ Φ_{wₛh}^{(2)}
# yields a 2n-order method with carefully chosen weights w_i.
#
# These are the unique real solutions to the order conditions.

YOSHIDA_WEIGHTS = {
    # 4th order: 3 stages
    4: [
        1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
        -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0)),
        1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
    ],
    # 6th order: 7 stages
    6: [
         0.784513610477560,
         0.235573213359357,
        -1.177679984178870,
         1.315186320683910,
        -1.177679984178870,
         0.235573213359357,
         0.784513610477560,
    ],
    # 8th order: 15 stages
    8: [
         0.741670364350612,
        -0.409100825800032,
         0.190754710296239,
        -0.573862471116082,
         0.299064181303656,
         0.334624918245298,
         0.315293092396767,
        -0.796887939352917,
         0.315293092396767,
         0.334624918245298,
         0.299064181303656,
        -0.573862471116082,
         0.190754710296239,
        -0.409100825800032,
         0.741670364350612,
    ],
}


class SymplecticIntegrator:
    """
    Base class for symplectic integrators.

    A numerical method y_{n+1} = Φ_h(y_n) is symplectic if:
        (∂Φ_h/∂y)ᵀ J (∂Φ_h/∂y) = J

    where J = [[0, I], [-I, 0]] is the canonical symplectic matrix.

    Equivalently: the numerical flow preserves the differential 2-form
    ω = Σ dp_i ∧ dq_i exactly.
    """

    def __init__(self,
                 hamiltonian: HamiltonianDynamics,
                 order: IntegratorOrder = IntegratorOrder.VERLET):
        self.hamiltonian = hamiltonian
        self.order = order
        self.dt = 0.005  # Default time step

    def step(self, q: np.ndarray, p: np.ndarray,
             tau: Optional[np.ndarray] = None) -> PhasePoint:
        """Single integration step. Override in subclasses."""
        raise NotImplementedError

    def integrate(self, q0: np.ndarray, p0: np.ndarray,
                  n_steps: int, tau: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate for n_steps.

        Returns:
            Tuple of (positions (n_steps+1, nq), momenta (n_steps+1, nv))
        """
        nq = len(q0)
        nv = len(p0)
        q_traj = np.zeros((n_steps + 1, nq))
        p_traj = np.zeros((n_steps + 1, nv))
        q_traj[0] = q0
        p_traj[0] = p0

        q, p = q0.copy(), p0.copy()
        for k in range(n_steps):
            result = self.step(q, p, tau)
            q, p = result.q, result.p
            q_traj[k + 1] = q
            p_traj[k + 1] = p

        return q_traj, p_traj


class StormerVerlet(SymplecticIntegrator):
    """
    Störmer-Verlet integrator (2nd order symplectic).

    For separable Hamiltonian H(p,q) = T(p) + V(q):

        p_{n+1/2} = p_n - (h/2) · ∇V(q_n)
        q_{n+1}   = q_n + h · ∇T(p_{n+1/2})
        p_{n+1}   = p_{n+1/2} - (h/2) · ∇V(q_{n+1})

    This is the "velocity Verlet" formulation. It is:
    - Symplectic: preserves ω exactly
    - Symmetric: Φ_{-h} ∘ Φ_h = I (time-reversible)
    - 2nd order accurate in position
    - Explicit when T(p) = ½pᵀM⁻¹p (standard kinetic energy)
    """

    def step(self, q: np.ndarray, p: np.ndarray,
             tau: Optional[np.ndarray] = None) -> PhasePoint:
        h = self.dt
        M = self.hamiltonian.lagrangian.mass_matrix(q)

        # Half-step in momentum (gravity)
        G = self.hamiltonian.lagrangian.gravity_vector(q)
        p_half = p - 0.5 * h * G
        if tau is not None:
            p_half[:len(tau)] += 0.5 * h * tau

        # Full step in position
        qdot = self.hamiltonian.velocity(q, p_half)
        q_new = q + h * qdot

        # Half-step in momentum (gravity at new position)
        G_new = self.hamiltonian.lagrangian.gravity_vector(q_new)
        p_new = p_half - 0.5 * h * G_new
        if tau is not None:
            p_new[:len(tau)] += 0.5 * h * tau

        return PhasePoint(q=q_new, p=p_new)


class YoshidaComposition(SymplecticIntegrator):
    """
    Higher-order symplectic integrator via Yoshida composition.

    Given a symmetric 2nd-order method Φ_h^{(2)}, the composition:
        Φ_h^{(2n)} = Φ_{w₁h}^{(2)} ∘ ... ∘ Φ_{wₛh}^{(2)}

    yields a 2n-th order symplectic integrator.

    First principle: This works because symplectic maps form a group under
    composition. Any composition of symplectic maps is symplectic. By
    carefully choosing the weights w_i, we cancel lower-order error terms.

    The weights satisfy:
        Σ w_i = 1                          (consistency)
        Σ w_i^{2n+1} = 0 for n < order    (higher order)
    """

    def __init__(self,
                 hamiltonian: HamiltonianDynamics,
                 order: IntegratorOrder = IntegratorOrder.YOSHIDA4):
        super().__init__(hamiltonian, order)
        if order.value not in YOSHIDA_WEIGHTS:
            raise ValueError(f"No Yoshida weights for order {order}")
        self.weights = YOSHIDA_WEIGHTS[order.value]
        self.base_integrator = StormerVerlet(hamiltonian)
        self.base_integrator.order = IntegratorOrder.VERLET

    def step(self, q: np.ndarray, p: np.ndarray,
             tau: Optional[np.ndarray] = None) -> PhasePoint:
        """Yoshida composition step."""
        state = PhasePoint(q=q.copy(), p=p.copy())

        for w in self.weights:
            self.base_integrator.dt = w * self.dt
            state = self.base_integrator.step(state.q, state.p, tau)

        return state


class VariationalIntegrator(SymplecticIntegrator):
    """
    Variational integrator from discrete Lagrangian.

    Unlike splitting methods, variational integrators derive directly from
    the discrete Hamilton's principle δS_d = 0. They automatically preserve
    the symplectic structure because they're derived from a discrete action.

    Given discrete Lagrangian L_d(q_k, q_{k+1}), the discrete Euler-Lagrange
    (DEL) equations are:
        D₂L_d(q_{k-1}, q_k) + D₁L_d(q_k, q_{k+1}) + f_k = 0

    where f_k are external forces.

    First principle: The discrete action principle is the discrete analog of
    δ∫L dt = 0. It inherits all the geometric properties of the continuous
    variational principle — momentum maps, Noether's theorem, multisymplecticity.
    """

    def __init__(self,
                 hamiltonian: HamiltonianDynamics,
                 discrete_lagrangian: Optional[Callable] = None):
        """Initialize variational integrator.

        Args:
            hamiltonian: Hamiltonian dynamics
            discrete_lagrangian: Discrete Lagrangian L_d(q_k, q_{k+1})
        """
        super().__init__(hamiltonian, IntegratorOrder.VERLET)
        self.hamiltonian = hamiltonian
        self.discrete_lagrangian = discrete_lagrangian

        # Discrete Legendre transforms:
        # p_k⁻ = -D₁L_d(q_k, q_{k+1})  (left discrete momentum)
        # p_k⁺ =  D₂L_d(q_{k-1}, q_k)   (right discrete momentum)
        # DEL ensures p_k⁺ = p_k⁻ = p_k

    def discrete_legendre_left(self, q_k: np.ndarray,
                                q_kp1: np.ndarray) -> np.ndarray:
        """Left discrete Legendre transform: p_k⁻ = -D₁L_d(q_k, q_{k+1})."""
        eps = 1e-6
        n = len(q_k)
        p = np.zeros(n)

        if self.discrete_lagrangian is not None:
            L_base = self.discrete_lagrangian(q_k, q_kp1)
            for i in range(n):
                q_plus = q_k.copy()
                q_plus[i] += eps
                L_plus = self.discrete_lagrangian(q_plus, q_kp1)
                p[i] = -(L_plus - L_base) / eps
        else:
            # Midpoint rule discrete Lagrangian
            # p_k⁻ = M((q_k+q_{k+1})/2) · (q_{k+1}-q_k)/h - h/2 · ∇V(q_k)
            h = self.dt
            q_mid = 0.5 * (q_k + q_kp1)
            M = self.hamiltonian.lagrangian.mass_matrix(q_mid)
            p = M @ ((q_kp1 - q_k) / h)
            G_k = self.hamiltonian.lagrangian.gravity_vector(q_k)
            p = p - 0.5 * h * G_k

        return p

    def discrete_legendre_right(self, q_km1: np.ndarray,
                                 q_k: np.ndarray) -> np.ndarray:
        """Right discrete Legendre transform: p_k⁺ = D₂L_d(q_{k-1}, q_k)."""
        eps = 1e-6
        n = len(q_k)
        p = np.zeros(n)

        if self.discrete_lagrangian is not None:
            L_base = self.discrete_lagrangian(q_km1, q_k)
            for i in range(n):
                q_plus = q_k.copy()
                q_plus[i] += eps
                L_plus = self.discrete_lagrangian(q_km1, q_plus)
                p[i] = (L_plus - L_base) / eps
        else:
            h = self.dt
            q_mid = 0.5 * (q_km1 + q_k)
            M = self.hamiltonian.lagrangian.mass_matrix(q_mid)
            p = M @ ((q_k - q_km1) / h)
            G_k = self.hamiltonian.lagrangian.gravity_vector(q_k)
            p = p + 0.5 * h * G_k

        return p

    def step(self, q: np.ndarray, p: np.ndarray,
             tau: Optional[np.ndarray] = None) -> PhasePoint:
        """Variational integrator step by solving DEL equations.

        Given (q_k, p_k), solve for q_{k+1} from:
            p_k = -D₁L_d(q_k, q_{k+1}) + (h/2) τ_k

        This requires solving a nonlinear equation — we use Newton's method.
        """
        h = self.dt
        n = len(q)

        def residual(q_next):
            p_left = self.discrete_legendre_left(q, q_next)
            rhs = -p
            if tau is not None:
                rhs[:len(tau)] += 0.5 * h * tau
            return p_left - rhs

        # Newton iteration
        q_next = q + h * self.hamiltonian.velocity(q, p)  # Initial guess
        for _ in range(5):  # Newton steps
            r = residual(q_next)
            if np.linalg.norm(r) < 1e-8:
                break

            # Approximate Jacobian via finite differences
            J = np.zeros((n, n))
            eps = 1e-6
            for i in range(min(n, 16)):
                q_plus = q_next.copy()
                q_plus[i] += eps
                J[:, i] = (residual(q_plus) - r) / eps

            try:
                dq = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(J, -r, rcond=None)[0]

            # Line search
            alpha = 1.0
            for _ in range(5):
                q_try = q_next + alpha * dq
                r_try = residual(q_try)
                if np.linalg.norm(r_try) < np.linalg.norm(r):
                    q_next = q_try
                    break
                alpha *= 0.5
            else:
                q_next = q_next + alpha * dq

        # Compute momentum at new state
        p_new = self.discrete_legendre_right(q, q_next)
        if tau is not None:
            p_new[:len(tau)] += 0.5 * h * tau

        return PhasePoint(q=q_next, p=p_new)


class SymplecticEuler(SymplecticIntegrator):
    """
    Symplectic Euler method (1st order, explicit).

    q_{n+1} = q_n + h · ∂T/∂p(p_n)
    p_{n+1} = p_n - h · ∂V/∂q(q_{n+1})

    This is a splitting of H = T + V: first drift (T), then kick (V).
    It is the simplest symplectic method.
    """

    def step(self, q: np.ndarray, p: np.ndarray,
             tau: Optional[np.ndarray] = None) -> PhasePoint:
        h = self.dt

        # Position update: q_{n+1} = q_n + h · M⁻¹ p_n
        qdot = self.hamiltonian.velocity(q, p)
        q_new = q + h * qdot

        # Momentum update: p_{n+1} = p_n - h · ∇V(q_{n+1})
        G_new = self.hamiltonian.lagrangian.gravity_vector(q_new)
        p_new = p - h * G_new
        if tau is not None:
            p_new[:len(tau)] += h * tau

        return PhasePoint(q=q_new, p=p_new)


def create_integrator(hamiltonian: HamiltonianDynamics,
                       integrator_type: str = "verlet",
                       order: int = 2) -> SymplecticIntegrator:
    """Factory function for creating symplectic integrators.

    Args:
        hamiltonian: Hamiltonian dynamics model
        integrator_type: "symplectic_euler", "verlet", "yoshida", "variational"
        order: Integration order (2, 4, 6, 8)

    Returns:
        SymplecticIntegrator instance
    """
    type_map = {
        "symplectic_euler": SymplecticEuler,
        "verlet": StormerVerlet,
        "yoshida": YoshidaComposition,
        "variational": VariationalIntegrator,
    }

    if integrator_type not in type_map:
        raise ValueError(f"Unknown integrator: {integrator_type}")

    if integrator_type == "yoshida":
        order_enum = {
            2: IntegratorOrder.VERLET,
            4: IntegratorOrder.YOSHIDA4,
            6: IntegratorOrder.YOSHIDA6,
            8: IntegratorOrder.YOSHIDA8,
        }.get(order, IntegratorOrder.YOSHIDA4)
        return YoshidaComposition(hamiltonian, order_enum)
    else:
        return type_map[integrator_type](hamiltonian)
