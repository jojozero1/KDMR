"""
KDMR - Kinodynamic Motion Retargeting for Humanoid Locomotion

A dynamics-constrained motion retargeting framework that extends GMR by
incorporating rigid-body dynamics and contact complementarity constraints.

Based on paper: arXiv:2603.09956
"""

__version__ = "0.1.0"
__author__ = "KDMR Team"

from kdmr.retargeting.kdmr_retaret import KDMR
from kdmr.contact.contact_estimator import ContactEstimator
from kdmr.core.trajectory_optimizer import TrajectoryOptimizer

__all__ = [
    "KDMR",
    "ContactEstimator", 
    "TrajectoryOptimizer",
]
