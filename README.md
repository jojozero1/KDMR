# QKDMR — Quantum Kinetodynamic Motion Retargeting v2.0

<p align="center">
  <img src="https://img.shields.io/badge/Physics-First_Principles-blue?style=for-the-badge" alt="Physics">
  <img src="https://img.shields.io/badge/Hamiltonian-Optimization-purple?style=for-the-badge" alt="Hamiltonian">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MuJoCo-3.0+-FF6B6B?style=for-the-badge" alt="MuJoCo">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Symplectic_Integrator-✓-success?style=flat-square" alt="Symplectic">
  <img src="https://img.shields.io/badge/Contact_Implicit-✓-success?style=flat-square" alt="Contact-Implicit">
  <img src="https://img.shields.io/badge/Riemannian_Opt-✓-success?style=flat-square" alt="Riemannian">
  <img src="https://img.shields.io/badge/Diffusion_Prior-✓-success?style=flat-square" alt="Diffusion">
  <img src="https://img.shields.io/badge/Energy_Shaping-✓-success?style=flat-square" alt="Energy">
</p>

---

## Preface: From Engineering to First Principles

KDMR v0.1 (arXiv:2603.09956) works. It uses SCP-DDP — iteratively linearize dynamics, convexify constraints, solve with DDP. It produces dynamically feasible trajectories. But it is an *engineering* solution: it patches over the fundamental structure of the problem rather than exploiting it.

**The universe does not optimize cost functions. The universe extremizes action.**

This insight — the Principle of Least Action — is the foundation of **QKDMR v2.0**. By reformulating kinodynamic motion retargeting as a stationary-action problem on the cotangent bundle (phase space), we unlock:

1. **Natural symplectic structure** — trajectories automatically conserve energy when unforced
2. **Contact as complementarity** — contact forces emerge from Karush-Kuhn-Tucker conditions, not heuristic thresholds
3. **Riemannian natural gradients** — optimization respects the curved geometry of SO(3)×R³×... (the Lie group configuration space)
4. **Noether invariants** — linear/angular momentum conservation constrains the search space
5. **Energy shaping** — the retargeted motion is a passive dynamic system, not a forced trajectory

---

## Architecture

```
QKDMR v2.0
│
├── HamiltonianActionOptimizer     ← Replaces SCP-DDP
│   ├── Action functional: S[q] = ∫ L(q, q̇) dt
│   ├── Stationary condition: δS = 0 → Euler-Lagrange
│   └── Monte Carlo over path space with Langevin dynamics
│
├── ContactImplicitSolver          ← Replaces GRF heuristic
│   ├── Complementarity: 0 ≤ λ ⟂ φ(q) ≥ 0
│   ├── NCP functions (Fischer-Burmeister)
│   └── Interior-point barrier with adaptive stiffness
│
├── SymplecticIntegrator           ← Exact energy conservation
│   ├── Störmer-Verlet (2nd order)
│   ├── Yoshida composition (4th, 6th, 8th order)
│   └── Variational integrator from discrete Lagrangian
│
├── RiemannianOptimizer            ← Geometry-aware descent
│   ├── Exponential map on Lie groups
│   ├── Natural gradient (Fisher-Rao metric)
│   └── Geodesic interpolation on SO(3)
│
├── EnergyShaper                   ← Passivity-based design
│   ├── Interconnection & damping assignment
│   ├── Desired Hamiltonian Hd shapes closed-loop dynamics
│   └── Barrier certificates for safety
│
└── DiffusionTrajectoryPrior       ← Learned warm-start
    ├── Score-based diffusion on trajectory manifold
    ├── Morphology-conditional generation
    └── Classifier-free guidance for contact modes
```

### From SCP-DDP to HMCTO (Hamiltonian Monte Carlo Trajectory Optimization)

```
v0.1 SCP-DDP:                    v2.0 HMCTO:
──────────────                    ────────────
1. Linearize dynamics      →     1. Lagrangian L = T(q,q̇) - V(q)
2. Convexify constraints   →     2. Discrete action: Sd = Σ Ld(qk, qk+1)
3. DDP backward pass       →     3. Natural gradient on ∇Sd = 0
4. Line-search forward     →     4. Symplectic momentum refresh
5. Trust-region update     →     5. Langevin: q ← q - ε·G⁻¹∇Sd + √(2ε)·ξ
```

The key difference: **v0.1 linearizes the dynamics** (destroying symplecticity), while **v2.0 works directly with the variational structure** (preserving it).

---

## Quick Start

```python
from kdmr_v2 import QKDMR
from kdmr_v2.utils.data_loader import DataLoader

# Load human motion
loader = DataLoader()
human_motion = loader.load_smplx_motion("path/to/motion.npz")

# Create QKDMR with Hamiltonian optimizer
qkdmr = QKDMR(
    robot_xml_path="assets/unitree_g1/g1_mocap_29dof.xml",
    optimizer="hamiltonian",        # New: Hamiltonian Monte Carlo
    contact_mode="implicit",        # New: Contact-implicit (no GRF needed!)
    integrator="yoshida6",          # New: 6th-order symplectic
    use_diffusion_prior=True        # New: Learned warm-start
)

# Run retargeting — GRF is now OPTIONAL
result = qkdmr.retarget(human_motion)

# The result.trajectory is dynamically feasible BY CONSTRUCTION
# because it satisfies the stationary-action principle.
```

### Command Line

```bash
# Run QKDMR
python scripts/run_qkdmr.py \
    --motion_file path/to/motion.npz \
    --robot unitree_g1 \
    --optimizer hamiltonian \
    --integrator yoshida6 \
    --output output/trajectory.npz

# Compare v0.1 vs v2.0
python scripts/compare_versions.py \
    --motion_file path/to/motion.npz \
    --robot unitree_g1
```

---

## Key Innovations (with Physics Justification)

### 1. Hamiltonian Monte Carlo Trajectory Optimization

**First principle**: The physical trajectory `q(t)` is a stationary point of the action functional `S[q] = ∫₀ᵀ L(q, q̇) dt` where `L = T - V` is the Lagrangian. The Euler-Lagrange equations `d/dt(∂L/∂q̇) - ∂L/∂q = τ_ext + Jᵀλ` are the necessary condition.

**Implementation**: We discretize the action using variational integrators and find stationary trajectories via Langevin dynamics on the path space — the natural gradient of the discrete action, with symplectic momentum refreshments.

### 2. Contact-Implicit Optimization

**First principle**: Rigid body contact is a **Signorini condition**: (1) non-penetration `φ(q) ≥ 0`, (2) non-negative normal force `λn ≥ 0`, (3) complementarity `λn·φ(q) = 0`. This is a Nonlinear Complementarity Problem (NCP).

**Implementation**: We use an interior-point method with the Fischer-Burmeister NCP function `ψ(a,b) = √(a² + b²) - a - b`, which is smooth and satisfies `ψ(a,b) = 0 ⇔ a ≥ 0, b ≥ 0, ab = 0`. This eliminates the need for GRF-based contact estimation entirely.

### 3. Riemannian Natural Gradient

**First principle**: The configuration space of a humanoid is `SE(3) × (S¹)ⁿ` — a product of Lie groups. Euclidean optimization (adding quaternion updates, then normalizing) is a hack that destroys the geometric structure.

**Implementation**: We use the **exponential map** for SO(3) updates and the **Fisher-Rao natural gradient** `G⁻¹∇L` where `G` is the Riemannian metric. This ensures: (a) quaternions stay on the unit sphere naturally, (b) optimization follows geodesics, (c) convergence is parametrization-invariant.

### 4. Symplectic Integration

**First principle**: Continuous-time Hamiltonian flow is symplectic (preserves the canonical 2-form `ω = Σ dpᵢ ∧ dqᵢ`). Standard integrators (RK4, Euler) are NOT symplectic — they artificially inject/remove energy, causing unstable trajectories.

**Implementation**: Yoshida-composed Störmer-Verlet integrators (2nd, 4th, 6th, 8th order) that exactly preserve a *modified* Hamiltonian `H̃ = H + O(Δtᵖ)`, guaranteeing bounded energy error for exponentially long times.

### 5. Energy Shaping via IDA-PBC

**First principle**: A dynamically feasible motion is one where the robot's closed-loop dynamics match those of a *passive* system. By the passivity theorem, any system of the form `M(q)q̈ + C(q,q̇)q̇ + ∂V/∂q = 0` is stable and physically realizable.

**Implementation**: Interconnection and Damping Assignment Passivity-Based Control (IDA-PBC) shapes the total energy `Hd = ½pᵀMd⁻¹p + Vd(q)` so that the desired trajectory is a minimum-energy path. The required torques emerge as `τ = ...` — no explicit trajectory optimization needed.

### 6. Diffusion Trajectory Prior

**First principle**: Optimization should exploit the fact that most human-to-robot motion pairs lie on a low-dimensional manifold. A score-based generative model can provide both a warm-start AND an informative prior for regularization.

**Implementation**: A diffusion model trained on successful retargeting examples provides `∇ log p(q)` — the score function — which guides the Hamiltonian optimizer toward physically plausible regions.

---

## Comparison: KDMR v0.1 vs QKDMR v2.0

| Dimension | v0.1 (SCP-DDP) | v2.0 (HMCTO) | Physics Rationale |
|-----------|---------------|-------------|-------------------|
| **Optimization** | Sequential linearization | Stationary action on cotangent bundle | Least Action Principle |
| **Dynamics** | Finite-diff Jacobians | Analytic symplectic gradients | Noether's theorem |
| **Contact** | GRF heuristic | Complementarity NCP | Signorini conditions |
| **Geometry** | Euclidean on quaternions | Riemannian on Lie groups | Natural metric on SO(3) |
| **Integration** | MuJoCo Euler (1st order) | Yoshida (up to 8th order) | Symplectic structure |
| **Prior** | Kinematic IK guess | Diffusion model + energy shaping | Manifold hypothesis |
| **Guarantees** | Dynamic feasibility (penalized) | Dynamic feasibility (constraint) | KKT conditions |
| **GRF dependency** | Required for contact | Optional (auto-detected) | NCP solves for contact |
| **Energy conservation** | Not preserved | Preserved to O(Δt⁶) | Modified Hamiltonian |
| **Convergence rate** | Linear (SCP) | Superlinear (natural gradient) | Fisher information |

---

## Installation

```bash
git clone https://github.com/jojozero1/KDMR.git
cd KDMR_v2
pip install -e ".[all]"
```

### Requirements

- Python >= 3.11
- MuJoCo >= 3.0.0
- PyTorch >= 2.0 (for diffusion prior)
- JAX >= 0.4 (for automatic differentiation of dynamics)
- Potenial (for Lie group optimization)

---

## Configuration

```yaml
# configs/qkdmr_config.yaml
optimizer:
  type: hamiltonian          # "hamiltonian" | "scp_ddp" (legacy)
  max_iterations: 200
  temperature: 0.01          # Langevin temperature
  friction: 0.1              # Momentum damping
  
contact:
  mode: implicit             # "implicit" | "grf_heuristic" (legacy)
  barrier_stiffness: 1e3     # Initial interior-point stiffness
  barrier_adaptation: true   # Auto-tune stiffness

integrator:
  type: yoshida6             # "verlet" | "yoshida4" | "yoshida6" | "yoshida8"
  dt: 0.005                  # Integration time step
  
geometry:
  type: riemannian           # "riemannian" | "euclidean" (legacy)
  lie_group: auto            # Auto-detect from model

diffusion_prior:
  enabled: true
  model_path: null           # Use default pretrained
  guidance_scale: 1.0        # Classifier-free guidance weight

energy_shaping:
  enabled: true
  damping_injection: 1.0     # Damping coefficient
  potential_scaling: 1.0     # Energy shaping strength
```

---

## Physics Appendix: From First Principles to Algorithm

### A. The Action Principle

Given a Lagrangian `L(q, q̇) = ½q̇ᵀM(q)q̇ - V(q)`, the physical trajectory satisfies:

```
δS = δ∫₀ᵀ L dt = 0
  → d/dt(∂L/∂q̇) - ∂L/∂q = τ_ext + Jᵀλ
  → M(q)q̈ + Ċ(q,q̇)q̇ + G(q) = τ + Jᵀλ     ← manipulator equation
```

### B. Contact as Complementarity

For each contact point `i` with distance `φᵢ(q)`:

```
0 ≤ φᵢ(q) ⟂ λᵢ ≥ 0        (Signorini condition)
|λᵢ,τ| ≤ μ λᵢ,ₙ            (Coulomb friction cone)
```

The Fischer-Burmeister NCP function gives a smooth equivalent:
`ψ(φ, λ) = √(φ² + λ² + ε²) - φ - λ = 0`

### C. Symplectic Integration

The continuous flow `Φₜ` of Hamiltonian H satisfies `Φₜ* ω = ω` (preserves the symplectic 2-form). A numerical integrator `Φₕ` is symplectic iff it is the exact flow of a **modified** Hamiltonian `H̃ = H + hᵖΔH + ...`.

The Störmer-Verlet method (2nd order) for `Mq̈ = -∇V`:
```
q_{k+1} = q_k + h v_k - ½h² M⁻¹∇V(q_k)
v_{k+1} = v_k - ½h [M⁻¹∇V(q_k) + M⁻¹∇V(q_{k+1})]
```

### D. Natural Gradient on Lie Groups

For a loss `f(q)` with `q ∈ SO(3)`:
```
∇_{natural} f(q) = G(q)⁻¹ ∇_{Euclidean} f(q)
```
where `G(q)` is the Fisher-Rao metric `Gᵢⱼ = tr(JᵢᵀJⱼ)` for the Jacobians of the exponential map.

The update: `q ← q ∘ exp(-η · ∇_{natural} f)` where `∘` is group composition and `exp` is the Lie algebra exponential.

---

## Citation

If you use QKDMR in your research, please cite:

```bibtex
@article{zhang2026kdmr,
  title={Kinodynamic Motion Retargeting for Humanoid Locomotion via Multi-Contact Whole-Body Trajectory Optimization},
  author={Zhang, Xiaoyu and Haener, Steven and Madabushi, Varun and Tucker, Maegan},
  journal={arXiv preprint arXiv:2603.09956},
  year={2026}
}

@software{qkdmr2026,
  title={QKDMR: Next-Generation Kinetodynamic Motion Retargeting},
  note={Derivative work extending KDMR v0.1 with first-principles physics methods},
  year={2026}
}
```

## License

MIT License
