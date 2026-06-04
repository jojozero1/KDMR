"""
QKDMR Physics Package — First-principles mechanics for motion retargeting.

This package implements the variational (Lagrangian/Hamiltonian) formulation
of rigid body dynamics that underpins QKDMR v2.0.

Modules:
- lagrangian_dynamics:   L(q, q̇) = T - V, Hamilton's principle, DEL equations
- symplectic_integrator: Energy-conserving numerical integration (Verlet, Yoshida)
- complementarity:       Signorini contact via NCP functions (Fischer-Burmeister)
- energy_shaping:        IDA-PBC passivity-based control, barrier certificates

First Principle: All physical dynamics derive from the principle of least action.
By working at this level (not the Newton-Euler or manipulator equation level),
we inherit the symplectic structure that guarantees energy conservation and
geometric correctness.
"""

from kdmr_v2.physics.lagrangian_dynamics import (
    LagrangianDynamics,
    HamiltonianDynamics,
    ActionFunctional,
    PhasePoint,
    GeneralizedState,
    NoetherInvariants,
)

from kdmr_v2.physics.symplectic_integrator import (
    SymplecticIntegrator,
    StormerVerlet,
    YoshidaComposition,
    VariationalIntegrator,
    SymplecticEuler,
    create_integrator,
    IntegratorOrder,
    YOSHIDA_WEIGHTS,
)

from kdmr_v2.physics.complementarity import (
    ContactImplicitSolver,
    FischerBurmeisterNCP,
    SoftContactModel,
    ContactPoint,
    ContactConstraint,
    NCPFunctionType,
)

from kdmr_v2.physics.energy_shaping import (
    IDAPBC,
    DesiredEnergy,
    QuadraticPotential,
    GravitationalPotential,
    ContactPotential,
    BarrierCertificate,
    create_desired_energy_from_trajectory,
)

__all__ = [
    # Lagrangian/Hamiltonian
    "LagrangianDynamics",
    "HamiltonianDynamics",
    "ActionFunctional",
    "PhasePoint",
    "GeneralizedState",
    "NoetherInvariants",

    # Integrators
    "SymplecticIntegrator",
    "StormerVerlet",
    "YoshidaComposition",
    "VariationalIntegrator",
    "SymplecticEuler",
    "create_integrator",
    "IntegratorOrder",
    "YOSHIDA_WEIGHTS",

    # Contact
    "ContactImplicitSolver",
    "FischerBurmeisterNCP",
    "SoftContactModel",
    "ContactPoint",
    "ContactConstraint",
    "NCPFunctionType",

    # Energy shaping
    "IDAPBC",
    "DesiredEnergy",
    "QuadraticPotential",
    "GravitationalPotential",
    "ContactPotential",
    "BarrierCertificate",
    "create_desired_energy_from_trajectory",
]
