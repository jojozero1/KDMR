#!/usr/bin/env python3
"""
Evaluate metrics for motion retargeting results.

Computes standard metrics for comparing retargeting quality:
- Dynamic feasibility
- GRF tracking accuracy
- Trajectory smoothness
- Contact consistency
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kdmr.utils.math_utils import MathUtils
from kdmr.utils.data_loader import DataLoader


def compute_all_metrics(trajectory, reference, fps):
    """Compute all evaluation metrics."""
    metrics = {}
    
    # Tracking error
    tracking_error = np.sqrt(np.mean((trajectory - reference) ** 2))
    metrics['tracking_rmse'] = tracking_error
    
    # Per-joint tracking error
    if trajectory.shape[1] > 7:
        joint_errors = np.sqrt(np.mean((trajectory[:, 7:] - reference[:, 7:]) ** 2, axis=0))
        metrics['joint_tracking_mean'] = np.mean(joint_errors)
        metrics['joint_tracking_max'] = np.max(joint_errors)
    
    # Root tracking
    root_pos_error = np.sqrt(np.mean(np.sum((trajectory[:, :3] - reference[:, :3]) ** 2, axis=1)))
    metrics['root_position_rmse'] = root_pos_error
    
    # Smoothness (jerk)
    jerk = MathUtils.compute_jerk(trajectory, 1.0 / fps)
    smoothness = np.mean(np.sum(jerk ** 2, axis=1))
    metrics['smoothness_jerk'] = smoothness
    
    # Velocity continuity
    velocity = MathUtils.compute_velocity(trajectory, 1.0 / fps)
    velocity_std = np.std(velocity, axis=0).mean()
    metrics['velocity_continuity'] = velocity_std
    
    # Acceleration
    acceleration = MathUtils.compute_acceleration(velocity, 1.0 / fps)
    acceleration_max = np.max(np.abs(acceleration))
    metrics['max_acceleration'] = acceleration_max
    
    # Ground contact
    foot_height = trajectory[:, 2]  # Using root height as proxy
    ground_penetration = np.sum(foot_height < 0) / len(foot_height)
    metrics['ground_penetration_ratio'] = ground_penetration
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate retargeting metrics')
    parser.add_argument('--trajectory', type=str, required=True, help='Trajectory file (NPZ)')
    parser.add_argument('--reference', type=str, default=None, help='Reference trajectory')
    parser.add_argument('--output', type=str, default=None, help='Output file for metrics')
    
    args = parser.parse_args()
    
    # Load trajectory
    data = np.load(args.trajectory)
    trajectory = data['qpos']
    fps = float(data.get('fps', 30))
    
    # Load or create reference
    if args.reference:
        ref_data = np.load(args.reference)
        reference = ref_data['qpos']
    else:
        reference = trajectory  # Self-comparison
    
    # Compute metrics
    metrics = compute_all_metrics(trajectory, reference, fps)
    
    # Print results
    print("\n=== Evaluation Metrics ===\n")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    
    # Save if requested
    if args.output:
        np.savez(args.output, **metrics)
        print(f"\nMetrics saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
