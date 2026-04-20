<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&duration=3000&pause=500&center=true&vCenter=true&width=600&lines=KDMR;Kinodynamic+Motion+Retargeting" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-2603.09956-b31b1b.svg?style=for-the-badge" alt="arXiv">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MuJoCo-3.0+-FF6B6B?style=for-the-badge&logo=google&logoColor=white" alt="MuJoCo">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Dynamics_Constrained-✓-success?style=flat-square" alt="Dynamics">
  <img src="https://img.shields.io/badge/Multi_Contact-✓-success?style=flat-square" alt="Multi-Contact">
  <img src="https://img.shields.io/badge/SCP_DDP-✓-success?style=flat-square" alt="SCP-DDP">
  <img src="https://img.shields.io/badge/GRF_Integration-✓-success?style=flat-square" alt="GRF">
</p>


A dynamics-constrained motion retargeting framework for humanoid locomotion, based on the paper **"Kinodynamic Motion Retargeting for Humanoid Locomotion via Multi-Contact Whole-Body Trajectory Optimization"** (arXiv:2603.09956).

## Overview

KDMR extends pure kinematic retargeting methods (like GMR) by incorporating rigid-body dynamics and contact complementarity constraints. This produces dynamically feasible reference trajectories that are essential for:

- Imitation learning with better sample efficiency
- Real robot deployment without physics violations
- Natural multi-contact locomotion patterns

### Key Features

- **SCP-DDP Solver**: Sequential Convex Programming with Differential Dynamic Programming
- **GRF Integration**: Uses Ground Reaction Force data for contact detection
- **Multi-Contact Support**: Heel, flat-foot, toe, and swing phase modeling
- **Physics Constraints**: Dynamics equations, friction cones, joint/torque limits
- **Multi-Robot Support**: Unitree G1, H1, Booster T1, and more

## Installation

```bash
# Clone the repository
git clone https://github.com/kdmr/kdmr.git
cd kdmr

# Create conda environment
conda create -n kdmr python=3.10
conda activate kdmr

# Install dependencies
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

### Requirements

- Python >= 3.10
- MuJoCo >= 3.0.0
- NumPy, SciPy
- CVXPY (for convex optimization)
- Mink (for IK)

## Quick Start

### Basic Usage

```python
from kdmr import KDMR
from kdmr.utils.data_loader import DataLoader

# Load human motion data
loader = DataLoader()
human_motion = loader.load_smplx_motion("path/to/motion.npz")

# Optional: Load GRF data
grf_data = loader.load_grf_data("path/to/grf.csv")

# Create KDMR instance
kdmr = KDMR(
    robot_xml_path="assets/unitree_g1/g1_mocap_29dof.xml",
    ik_config_path="configs/ik/smplx_to_g1.json"
)

# Run retargeting
result = kdmr.retarget(human_motion, grf_data)

# Access results
trajectory = result.trajectory  # RobotTrajectory object
contact_sequence = result.contact_sequence  # DualContactSequence
```

### Command Line

```bash
# Run KDMR on motion file
python scripts/run_kdmr.py \
    --motion_file path/to/motion.npz \
    --robot unitree_g1 \
    --output output/trajectory.npz \
    --visualize

# With GRF data
python scripts/run_kdmr.py \
    --motion_file path/to/motion.npz \
    --grf_file path/to/grf.csv \
    --robot unitree_g1

# Compare with GMR baseline
python scripts/compare_with_gmr.py \
    --motion_file path/to/motion.npz \
    --robot unitree_g1 \
    --output_dir comparison_results
```

## Architecture

```
KDMR/
├── kdmr/
│   ├── core/               # Core optimization
│   │   ├── scp_ddp_solver.py    # SCP-DDP algorithm
│   │   ├── trajectory_optimizer.py
│   │   └── cost_functions.py
│   ├── dynamics/           # Dynamics computations
│   │   ├── rigid_body_dynamics.py
│   │   ├── contact_dynamics.py
│   │   └── constraints.py
│   ├── contact/            # Contact estimation
│   │   ├── grf_processor.py
│   │   ├── contact_estimator.py
│   │   └── contact_mode.py
│   ├── retargeting/        # Retargeting pipeline
│   │   ├── kinematic_retarget.py
│   │   └── kdmr_retaret.py
│   └── utils/              # Utilities
│       ├── math_utils.py
│       ├── visualization.py
│       └── data_loader.py
├── configs/
│   ├── robots/             # Robot configurations
│   └── solver/             # Solver parameters
├── scripts/                # Command-line tools
└── tests/                  # Unit tests
```

## Algorithm

### SCP-DDP Framework

```
1. Initialize:
   - Get initial guess from kinematic retargeting (GMR)
   - Estimate contact sequence from GRF data

2. SCP Outer Loop:
   a) Linearize dynamics around current trajectory
   b) Convexify contact constraints
   c) Solve DDP subproblem

3. DDP Inner Loop:
   a) Backward pass: Compute feedback gains
   b) Forward pass: Update trajectory with line search

4. Check convergence and return optimized trajectory
```

### Contact Sequence Estimation

From GRF vertical force profile:

```
     Force
       ∧
      / \      Double-peak pattern
     /   \    
    /     \  
   /       \ 
──┴─────────┴──→ Time
  H   F   T
  
H = Heel contact
F = Flat-foot contact  
T = Toe contact
```

## Comparison with GMR

| Feature | GMR | KDMR |
|---------|-----|------|
| Optimization | Kinematic only | Kinematic + Dynamic |
| Contact handling | Simplified | GRF-based detection |
| Dynamics constraints | None | Full rigid-body dynamics |
| Output guarantee | None | Dynamically feasible |
| Best for | Animation, simulation | Real robot control |

## Supported Robots

- Unitree G1
- Unitree H1
- Booster T1
- Booster T1 (29 DoF)
- And more (see `configs/robots/`)

## Configuration

### Solver Configuration

```yaml
# configs/solver/scp_ddp_config.yaml
scp:
  max_iterations: 10
  convergence_threshold: 1.0e-4

ddp:
  max_iterations: 50
  regularization: 1.0e-5
```

### Robot Configuration

```yaml
# configs/robots/unitree_g1.yaml
name: unitree_g1
nq: 29
nv: 28
nu: 23

contact_bodies:
  left_foot: left_toe_link
  right_foot: right_toe_link

joint_limits:
  lower: [...]
  upper: [...]
```

## API Reference

### Main Classes

- `KDMR`: Main retargeting class
- `TrajectoryOptimizer`: Trajectory optimization
- `SCPDDPSolver`: SCP-DDP solver
- `ContactEstimator`: Contact sequence estimation
- `GRFProcessor`: GRF data processing

### Data Classes

- `HumanMotionData`: Human motion container
- `GRFData`: Ground reaction force data
- `RobotTrajectory`: Robot trajectory output
- `ContactSequence`: Contact mode sequence

## Citation

If you use KDMR in your research, please cite:

```bibtex
@article{zhang2026kdmr,
  title={Kinodynamic Motion Retargeting for Humanoid Locomotion via Multi-Contact Whole-Body Trajectory Optimization},
  author={Zhang, Xiaoyu and Haener, Steven and Madabushi, Varun and Tucker, Maegan},
  journal={arXiv preprint arXiv:2603.09956},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- GMR project for kinematic retargeting foundation
- MuJoCo for physics simulation
- Mink for inverse kinematics
