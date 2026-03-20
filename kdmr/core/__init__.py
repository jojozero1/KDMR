"""
Core optimization module for KDMR.

This module contains:
- SCPDDPSolver: Sequential Convex Programming DDP solver
- TrajectoryOptimizer: Main trajectory optimization class
- CostFunctions: Cost function definitions
"""

from kdmr.core.scp_ddp_solver import SCPDDPSolver
from kdmr.core.trajectory_optimizer import TrajectoryOptimizer
from kdmr.core.cost_functions import CostFunctions

__all__ = [
    "SCPDDPSolver",
    "TrajectoryOptimizer", 
    "CostFunctions",
]
