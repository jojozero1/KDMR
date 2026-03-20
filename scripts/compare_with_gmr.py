#!/usr/bin/env python3
"""
Compare KDMR with GMR baseline.

This script runs both KDMR and GMR on the same motion data
and compares the results.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kdmr import KDMR
from kdmr.retargeting.kdmr_retaret import KDMRConfig, create_kdmr
from kdmr.utils.data_loader import DataLoader
from kdmr.utils.visualization import GRFPlotter


def parse_args():
    parser = argparse.ArgumentParser(description='Compare KDMR with GMR')
    
    parser.add_argument('--motion_file', type=str, required=True)
    parser.add_argument('--grf_file', type=str, default=None)
    parser.add_argument('--robot', type=str, default='unitree_g1')
    parser.add_argument('--output_dir', type=str, default='comparison_results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    assets_dir = project_root / 'assets'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    loader = DataLoader(data_root=project_root)
    
    motion_path = Path(args.motion_file)
    if motion_path.suffix == '.npz':
        human_motion = loader.load_smplx_motion(args.motion_file)
    elif motion_path.suffix == '.bvh':
        human_motion = loader.load_bvh_motion(args.motion_file)
    else:
        raise ValueError(f"Unsupported format: {motion_path.suffix}")
    
    grf_data = None
    if args.grf_file:
        grf_data = loader.load_grf_data(args.grf_file)
    
    print(f"Loaded {len(human_motion)} frames")
    
    # Run GMR (kinematic only)
    print("\n=== Running GMR (Kinematic Retargeting) ===")
    
    try:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR
        
        gmr = GMR(
            src_human='smplx',
            tgt_robot=args.robot,
            actual_human_height=1.75
        )
        
        gmr_trajectory = []
        for t in range(len(human_motion)):
            frame = human_motion.get_frame(t)
            qpos = gmr.retarget(frame)
            gmr_trajectory.append(qpos)
        
        gmr_trajectory = np.array(gmr_trajectory)
        print(f"GMR completed: {len(gmr_trajectory)} frames")
        
    except ImportError:
        print("GMR not available, using kinematic retargeting only")
        gmr_trajectory = None
    
    # Run KDMR
    print("\n=== Running KDMR (Dynamics-Constrained Retargeting) ===")
    
    config = KDMRConfig(
        verbose=True,
        max_scp_iterations=10
    )
    
    kdmr = create_kdmr(
        robot_name=args.robot,
        config=config,
        assets_dir=str(assets_dir)
    )
    
    result = kdmr.retarget(
        human_motion=human_motion,
        grf_data=grf_data,
        initial_trajectory=gmr_trajectory
    )
    
    print(f"KDMR completed in {result.total_time:.2f}s")
    
    # Compare
    print("\n=== Comparison Results ===")
    
    if gmr_trajectory is not None:
        comparison = kdmr.compare_with_gmr(gmr_trajectory, human_motion)
        
        print("\nGMR Metrics:")
        for key, value in comparison['GMR'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\nKDMR Metrics:")
        for key, value in comparison['KDMR'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\nImprovement:")
        for key, value in comparison['improvement'].items():
            print(f"  {key}: {value:.2f}%")
    
    # Save results
    np.savez(
        output_dir / 'kdmr_trajectory.npz',
        qpos=result.trajectory.qpos,
        qvel=result.trajectory.qvel,
        tau=result.trajectory.tau,
        fps=result.trajectory.fps,
        metrics=result.metrics
    )
    
    if gmr_trajectory is not None:
        np.savez(
            output_dir / 'gmr_trajectory.npz',
            qpos=gmr_trajectory,
            fps=human_motion.fps
        )
    
    print(f"\nResults saved to: {output_dir}")
    
    # Generate plots
    try:
        plotter = GRFPlotter()
        
        # Trajectory comparison
        if gmr_trajectory is not None:
            fig, axes = plotter.plot_trajectory_comparison(
                {'root_z': gmr_trajectory[:, 2]},
                {'root_z': result.trajectory.qpos[:, 2]},
                np.arange(len(human_motion)) / human_motion.fps,
                ['root_z']
            )
            plotter.save_figure(fig, output_dir / 'trajectory_comparison.png')
        
        print(f"Plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"Plotting error: {e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
