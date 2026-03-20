#!/usr/bin/env python3
"""
KDMR - Kinodynamic Motion Retargeting

Main script for running KDMR motion retargeting.

Usage:
    python run_kdmr.py --motion_file path/to/motion.npz --robot unitree_g1
    python run_kdmr.py --motion_file path/to/motion.npz --grf_file path/to/grf.csv --robot unitree_g1
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kdmr import KDMR
from kdmr.retargeting.kdmr_retaret import KDMRConfig, create_kdmr
from kdmr.utils.data_loader import DataLoader, HumanMotionData, GRFData
from kdmr.utils.visualization import KDMRVisualizer, GRFPlotter


def parse_args():
    parser = argparse.ArgumentParser(
        description='KDMR - Kinodynamic Motion Retargeting'
    )
    
    # Input files
    parser.add_argument(
        '--motion_file',
        type=str,
        required=True,
        help='Path to human motion file (SMPLX NPZ or BVH)'
    )
    
    parser.add_argument(
        '--grf_file',
        type=str,
        default=None,
        help='Path to GRF data file (optional)'
    )
    
    # Robot selection
    parser.add_argument(
        '--robot',
        type=str,
        choices=['unitree_g1', 'unitree_h1', 'booster_t1'],
        default='unitree_g1',
        help='Target robot'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for retargeted trajectory'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize result'
    )
    
    parser.add_argument(
        '--record_video',
        action='store_true',
        help='Record visualization video'
    )
    
    parser.add_argument(
        '--video_path',
        type=str,
        default='kdmr_output.mp4',
        help='Path for output video'
    )
    
    # Optimization parameters
    parser.add_argument(
        '--tracking_weight',
        type=float,
        default=100.0,
        help='Tracking cost weight'
    )
    
    parser.add_argument(
        '--smoothness_weight',
        type=float,
        default=0.1,
        help='Smoothness cost weight'
    )
    
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=10,
        help='Maximum SCP iterations'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    assets_dir = project_root / 'assets'
    
    # Create data loader
    loader = DataLoader(data_root=project_root)
    
    # Load human motion
    print(f"[KDMR] Loading motion from: {args.motion_file}")
    
    motion_path = Path(args.motion_file)
    if motion_path.suffix == '.npz':
        human_motion = loader.load_smplx_motion(args.motion_file)
    elif motion_path.suffix == '.bvh':
        human_motion = loader.load_bvh_motion(args.motion_file)
    else:
        raise ValueError(f"Unsupported motion format: {motion_path.suffix}")
    
    print(f"[KDMR] Loaded {len(human_motion)} frames at {human_motion.fps} FPS")
    
    # Load GRF data if provided
    grf_data = None
    if args.grf_file:
        print(f"[KDMR] Loading GRF data from: {args.grf_file}")
        grf_data = loader.load_grf_data(args.grf_file)
        print(f"[KDMR] Loaded {len(grf_data.timestamps)} GRF samples")
    
    # Create KDMR configuration
    config = KDMRConfig(
        tracking_weight=args.tracking_weight,
        smoothness_weight=args.smoothness_weight,
        max_scp_iterations=args.max_iterations,
        verbose=args.verbose
    )
    
    # Create KDMR instance
    print(f"[KDMR] Initializing for robot: {args.robot}")
    kdmr = create_kdmr(
        robot_name=args.robot,
        config=config,
        assets_dir=str(assets_dir)
    )
    
    # Run retargeting
    print("[KDMR] Running kinodynamic motion retargeting...")
    result = kdmr.retarget(
        human_motion=human_motion,
        grf_data=grf_data
    )
    
    print(f"[KDMR] Retargeting complete in {result.total_time:.2f}s")
    print(f"[KDMR] Metrics:")
    for key, value in result.metrics.items():
        print(f"  - {key}: {value:.4f}")
    
    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_path,
            qpos=result.trajectory.qpos,
            qvel=result.trajectory.qvel,
            tau=result.trajectory.tau,
            fps=result.trajectory.fps,
            metrics=result.metrics
        )
        print(f"[KDMR] Saved trajectory to: {args.output}")
    
    # Visualize
    if args.visualize:
        print("[KDMR] Starting visualization...")
        
        robot_xml = assets_dir / kdmr.trajectory_optimizer.robot_xml_path.name
        
        try:
            visualizer = KDMRVisualizer(str(robot_xml))
            visualizer.launch_viewer()
            
            if args.record_video:
                visualizer.start_recording(args.video_path)
            
            # Playback
            for t in range(len(result.trajectory)):
                qpos = result.trajectory.qpos[t]
                
                # Get contact modes
                left_mode = result.contact_sequence.left.modes[t]
                right_mode = result.contact_sequence.right.modes[t]
                
                visualizer.step(
                    qpos=qpos,
                    contact_modes={
                        'left_toe_link': left_mode,
                        'right_toe_link': right_mode
                    }
                )
            
            visualizer.close()
            
        except Exception as e:
            print(f"[KDMR] Visualization error: {e}")
    
    print("[KDMR] Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
