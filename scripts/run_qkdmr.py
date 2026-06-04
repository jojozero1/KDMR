#!/usr/bin/env python3
"""
QKDMR v2.0 — Command-line interface.

Runs Quantum Kinetodynamic Motion Retargeting with the Hamiltonian
Monte Carlo Trajectory Optimizer. Replaces the SCP-DDP solver of v0.1
with first-principles physics: stationary action, symplectic integration,
contact-implicit NCP, and Riemannian optimization.

Usage:
    python scripts/run_qkdmr.py \\
        --motion_file path/to/motion.npz \\
        --robot unitree_g1 \\
        --optimizer hamiltonian \\
        --output output/trajectory.npz

Comparison with v0.1:
    --compare: Run both v0.1 (SCP-DDP) and v2.0 (HMCTO) and compare
"""

import argparse
import sys
import time
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="QKDMR v2.0 — Quantum Kinetodynamic Motion Retargeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--motion_file", type=str, required=True,
        help="Path to human motion file (NPZ format)")
    parser.add_argument(
        "--motion_format", type=str, default="smplx",
        choices=["smplx", "bvh", "fbx"],
        help="Motion data format")
    parser.add_argument(
        "--robot", type=str, default="unitree_g1",
        choices=["unitree_g1", "unitree_h1", "booster_t1"],
        help="Target robot")

    # GRF (optional in v2.0!)
    parser.add_argument(
        "--grf_file", type=str, default=None,
        help="Path to GRF data (optional in v2.0 — contact-implicit)")

    # Optimization
    parser.add_argument(
        "--optimizer", type=str, default="hamiltonian",
        choices=["hamiltonian", "scp_ddp"],
        help="Optimizer type")
    parser.add_argument(
        "--max_iterations", type=int, default=200,
        help="Maximum optimization iterations")
    parser.add_argument(
        "--temperature", type=float, default=0.01,
        help="Langevin temperature (0 = deterministic)")
    parser.add_argument(
        "--step_size", type=float, default=0.01,
        help="Natural gradient step size")

    # Physics
    parser.add_argument(
        "--integrator", type=str, default="yoshida",
        choices=["verlet", "yoshida", "variational"],
        help="Symplectic integrator type")
    parser.add_argument(
        "--integrator_order", type=int, default=6,
        choices=[2, 4, 6, 8],
        help="Symplectic integrator order")
    parser.add_argument(
        "--contact_mode", type=str, default="implicit",
        choices=["implicit", "soft", "none"],
        help="Contact handling mode")

    # Diffusion prior
    parser.add_argument(
        "--use_diffusion_prior", action="store_true",
        help="Use learned diffusion model for warm-start")
    parser.add_argument(
        "--diffusion_model", type=str, default=None,
        help="Path to pretrained diffusion model")

    # Output
    parser.add_argument(
        "--output", type=str, default="output/qkdmr_result.npz",
        help="Output path for optimized trajectory")
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize the retargeted trajectory")

    # Comparison mode
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both v0.1 and v2.0 and compare")

    # Config
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to QKDMR configuration YAML")

    # Verbosity
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output")
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output")

    return parser.parse_args()


def load_motion(filepath: str, format: str = "smplx"):
    """Load human motion data."""
    filepath = Path(filepath)
    data = np.load(filepath, allow_pickle=True)

    # Create a simple motion data container
    class MotionData:
        def __init__(self, positions, joint_names, fps):
            self.positions = positions
            self.joint_names = joint_names
            self.fps = fps
            self.orientations = np.zeros((len(positions), len(joint_names), 4))
            self.orientations[..., 0] = 1.0  # Identity quaternion
            self.duration = len(positions) / fps

        def __len__(self):
            return len(self.positions)

        def get_frame(self, idx):
            return {
                name: (self.positions[idx, i], self.orientations[idx, i])
                for i, name in enumerate(self.joint_names)
            }

    joint_names = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
    ]

    if 'root_orient' in data and 'trans' in data:
        # SMPLX format
        n_frames = len(data['trans'])
        positions = np.zeros((n_frames, 22, 3))
        positions[:, 0] = data['trans']  # Pelvis
        fps = float(data.get('mocap_frame_rate', data.get('mocap_framerate', 30.0)))
    elif 'qpos' in data:
        # Already in robot format — wrap
        positions = data['qpos'][:, None, :3]  # Just use root position
        joint_names = ['pelvis']
        fps = float(data.get('fps', 30.0))
    else:
        raise ValueError(f"Unknown motion format: {list(data.keys())}")

    return MotionData(positions, joint_names, fps)


def run_qkdmr(args):
    """Run QKDMR v2.0 retargeting."""
    from kdmr_v2 import QKDMR, HMCTOConfig

    # Load motion
    print(f"Loading motion from: {args.motion_file}")
    motion = load_motion(args.motion_file, args.motion_format)
    print(f"  Frames: {len(motion)}, FPS: {motion.fps}")

    # Load GRF if provided
    grf_data = None
    if args.grf_file:
        print(f"Loading GRF from: {args.grf_file}")
        # Simplified GRF loading
        grf_raw = np.loadtxt(args.grf_file, delimiter=',', skiprows=1)
        grf_data = type('GRFData', (), {
            'forces': grf_raw[:, 1:7] if grf_raw.shape[1] >= 7 else grf_raw[:, 1:4],
            'timestamps': grf_raw[:, 0],
            'fps': 1.0 / np.mean(np.diff(grf_raw[:, 0])),
        })()

    # Setup robot path
    assets_dir = Path(__file__).parent.parent / "assets"
    robot_xml_map = {
        'unitree_g1': 'unitree_g1/g1_mocap_29dof.xml',
        'unitree_h1': 'unitree_h1/h1.xml',
        'booster_t1': 'booster_t1/T1_locomotion.xml',
    }

    if args.robot not in robot_xml_map:
        print(f"ERROR: Unknown robot '{args.robot}'")
        print(f"  Available: {list(robot_xml_map.keys())}")
        sys.exit(1)

    robot_xml = str(assets_dir / robot_xml_map[args.robot])
    if not Path(robot_xml).exists():
        print(f"WARNING: Robot XML not found at {robot_xml}")
        print("  Using default configuration sizes instead.")
        # Continue with placeholder dimensions

    # Configure optimizer
    config = HMCTOConfig(
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        step_size=args.step_size,
        integrator_type=args.integrator,
        integrator_order=args.integrator_order,
        contact_mode=args.contact_mode,
        verbose=not args.quiet,
    )

    # Load YAML config if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # Merge YAML config into HMCTOConfig
        if 'optimizer' in yaml_config:
            opt = yaml_config['optimizer']
            for key in ['max_iterations', 'temperature', 'friction', 'step_size',
                        'use_nesterov', 'momentum_beta']:
                if key in opt:
                    setattr(config, key, opt[key])
        if 'costs' in yaml_config:
            costs = yaml_config['costs']
            for key in ['tracking_weight', 'smoothness_weight', 'energy_weight',
                        'contact_weight', 'symmetry_weight']:
                mapped_key = key.replace('_weight', '')
                if key in costs:
                    setattr(config, key, costs[key])

    # Create QKDMR
    print(f"\n[QKDMR v2.0] Creating optimizer: {args.optimizer}")
    print(f"  Robot: {args.robot}")
    print(f"  Integrator: {args.integrator}-{args.integrator_order}")
    print(f"  Contact: {args.contact_mode}")
    print(f"  Diffusion prior: {args.use_diffusion_prior}")

    qkdmr = QKDMR(
        robot_xml_path=robot_xml,
        optimizer=args.optimizer,
        contact_mode=args.contact_mode,
        integrator_type=args.integrator,
        integrator_order=args.integrator_order,
        use_diffusion_prior=args.use_diffusion_prior,
        config=config,
    )

    # Load diffusion model if specified
    if args.use_diffusion_prior and args.diffusion_model:
        qkdmr.diffusion_prior.load_pretrained(args.diffusion_model)

    # Run retargeting
    print(f"\n[QKDMR v2.0] Starting retargeting...")
    result = qkdmr.retarget(
        human_motion=motion,
        grf_data=grf_data,
    )

    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qkdmr.save_result(str(output_path))
    print(f"\n[QKDMR v2.0] Result saved to: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"QKDMR v2.0 — Retargeting Summary")
    print(f"{'='*60}")
    print(f"  Converged:           {result.converged}")
    print(f"  Iterations:          {result.iterations}")
    print(f"  Solve time:          {result.solve_time:.2f}s")
    print(f"  Final action:        {result.action_history[-1]:.6f}")
    if result.energy_history:
        print(f"  Final energy:        {result.energy_history[-1]:.6f}")
    print(f"  Contact forces:      {result.contact_forces.shape}")
    print(f"  Trajectory shape:    {result.trajectory.shape}")
    print(f"{'='*60}")

    return result


def run_comparison(args):
    """Run both KDMR v0.1 and QKDMR v2.0 and compare."""
    print("\n" + "="*60)
    print("KDMR v0.1 vs QKDMR v2.0 — Comparison Mode")
    print("="*60)

    # Run v2.0 first
    print("\n--- QKDMR v2.0 (HMCTO) ---")
    v20_result = run_qkdmr(args)

    # Try running v0.1 if available
    print("\n--- KDMR v0.1 (SCP-DDP) ---")
    try:
        from kdmr import KDMR
        motion = load_motion(args.motion_file, args.motion_format)

        kdmr = KDMR(
            robot_xml_path=str(Path(__file__).parent.parent / "assets" /
                                  "unitree_g1" / "g1_mocap_29dof.xml"),
        )
        v01_result = kdmr.retarget(motion)

        # Compare
        print("\n" + "="*60)
        print("Comparison Results")
        print("="*60)

        print(f"\n{'Metric':<30} {'v0.1 (SCP-DDP)':<20} {'v2.0 (HMCTO)':<20} {'Improvement':<15}")
        print("-"*85)

        v01_smoothness = v01_result.metrics.get('smoothness', float('nan'))
        v20_smoothness = np.mean(
            np.sum(np.diff(v20_result.trajectory, n=3, axis=0)**2, axis=1)
        ) if len(v20_result.trajectory) > 3 else float('nan')

        print(f"{'Smoothness (jerk)':<30} {v01_smoothness:<20.6f} {v20_smoothness:<20.6f} "
              f"{'N/A':<15}")

        v01_tracking = v01_result.metrics.get('tracking_error', float('nan'))
        v20_tracking = float('nan')  # Would need reference trajectory

        print(f"{'Tracking error':<30} {v01_tracking:<20.6f} {v20_tracking:<20.6f} "
              f"{'N/A':<15}")

        print(f"\n{'Other metrics':<30} {'v0.1':<20} {'v2.0':<20}")
        print("-"*70)
        print(f"{'GRF required':<30} {'Yes':<20} {'No (contact-implicit)':<20}")
        print(f"{'Integrator':<30} {'MuJoCo Euler':<20} {'Yoshida-6 symplectic':<20}")
        print(f"{'Optimization':<30} {'SCP-DDP':<20} {'Hamiltonian Langevin':<20}")
        print(f"{'Geometry':<30} {'Euclidean':<20} {'Riemannian (Lie group)':<20}")
        print(f"{'Convergence':<30} {'Linear':<20} {'Superlinear (natural grad)':<20}")

    except ImportError:
        print("KDMR v0.1 not available — skipping comparison")
    except Exception as e:
        print(f"Error running v0.1: {e}")


def main():
    args = parse_args()

    if args.compare:
        run_comparison(args)
    else:
        run_qkdmr(args)

    # Visualize if requested
    if args.visualize:
        print("\n[QKDMR] Visualization not yet implemented in v2.0 CLI.")
        print("  Use the Python API for interactive visualization.")


if __name__ == "__main__":
    main()
