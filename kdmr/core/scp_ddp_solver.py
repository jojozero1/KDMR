"""
SCP-DDP Solver for KDMR.

This module implements the Sequential Convex Programming DDP (SCP-DDP) algorithm
for kinodynamic motion retargeting.

The algorithm combines:
1. Sequential Convex Programming (SCP): Iteratively linearize non-convex constraints
2. Differential Dynamic Programming (DDP): Efficient trajectory optimization

Reference: arXiv:2603.09956
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

from kdmr.core.cost_functions import CostFunctions, CostConfig
from kdmr.utils.math_utils import MathUtils


@dataclass
class SCPDDPConfig:
    """Configuration for SCP-DDP solver."""
    # SCP parameters
    max_scp_iterations: int = 10
    scp_convergence_threshold: float = 1e-4
    trust_region_radius: float = 0.1
    trust_region_increase: float = 1.5
    trust_region_decrease: float = 0.5
    
    # DDP parameters
    max_ddp_iterations: int = 50
    ddp_convergence_threshold: float = 1e-6
    regularization: float = 1e-5
    regularization_factor: float = 2.0
    
    # Line search
    line_search_steps: int = 10
    line_search_factor: float = 0.5
    min_step_size: float = 1e-4
    
    # Verbosity
    verbose: bool = True
    print_interval: int = 10


@dataclass
class DDPGains:
    """DDP feedback gains."""
    K: np.ndarray  # State feedback gain (n, n)
    k: np.ndarray  # Feedforward term (n,)


@dataclass
class SCPDDPResult:
    """Result from SCP-DDP optimization."""
    trajectory: np.ndarray      # Optimized trajectory (T, n)
    controls: np.ndarray        # Control sequence (T-1, m)
    cost_history: List[float]   # Cost at each iteration
    scp_iterations: int         # Number of SCP iterations
    ddp_iterations: int         # Total DDP iterations
    converged: bool             # Whether optimization converged
    solve_time: float           # Total solve time (seconds)


class LinearizedDynamics:
    """
    Linearized dynamics model.
    
    x_{t+1} = A_t @ x_t + B_t @ u_t + d_t
    
    where:
    - A_t: State Jacobian (∂f/∂x)
    - B_t: Control Jacobian (∂f/∂u)
    - d_t: Affine term
    """
    
    def __init__(self, 
                 A: np.ndarray,
                 B: np.ndarray,
                 d: np.ndarray):
        """
        Initialize linearized dynamics.
        
        Args:
            A: State Jacobian (n, n)
            B: Control Jacobian (n, m)
            d: Affine term (n,)
        """
        self.A = A
        self.B = B
        self.d = d
    
    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict next state."""
        return self.A @ x + self.B @ u + self.d


class QuadraticCostApprox:
    """
    Quadratic approximation of cost function.
    
    l(x, u) ≈ 0.5 * [x, u]^T @ Q @ [x, u] + q^T @ [x, u] + const
    
    where Q = [[Q_xx, Q_xu], [Q_ux, Q_uu]]
    """
    
    def __init__(self,
                 Q_xx: np.ndarray,
                 Q_uu: np.ndarray,
                 Q_xu: np.ndarray,
                 q_x: np.ndarray,
                 q_u: np.ndarray,
                 const: float = 0.0):
        """
        Initialize quadratic cost approximation.
        
        Args:
            Q_xx: State-state Hessian (n, n)
            Q_uu: Control-control Hessian (m, m)
            Q_xu: State-control Hessian (n, m)
            q_x: State gradient (n,)
            q_u: Control gradient (m,)
            const: Constant term
        """
        self.Q_xx = Q_xx
        self.Q_uu = Q_uu
        self.Q_xu = Q_xu
        self.Q_ux = Q_xu.T
        self.q_x = q_x
        self.q_u = q_u
        self.const = const


class SCPDDPSolver:
    """
    Sequential Convex Programming DDP Solver.
    
    Solves trajectory optimization problems of the form:
    
    minimize  sum_{t=0}^{T-1} l_t(x_t, u_t) + l_T(x_T)
    subject to x_{t+1} = f(x_t, u_t)     (dynamics)
               g(x_t, u_t) <= 0          (constraints)
    
    by iteratively:
    1. Linearizing dynamics and constraints (SCP)
    2. Solving convex subproblem with DDP
    3. Updating trust region
    """
    
    def __init__(self,
                 dynamics_func: Callable,
                 cost_functions: CostFunctions,
                 config: Optional[SCPDDPConfig] = None):
        """
        Initialize SCP-DDP solver.
        
        Args:
            dynamics_func: Function f(x, u) -> x_next
            cost_functions: Cost function container
            config: Solver configuration
        """
        self.dynamics_func = dynamics_func
        self.cost_functions = cost_functions
        self.config = config or SCPDDPConfig()
        
        # Problem dimensions (set during solve)
        self.n = None  # State dimension
        self.m = None  # Control dimension
        self.T = None  # Horizon length
        
        # Current solution
        self.x_traj = None  # State trajectory
        self.u_traj = None  # Control trajectory
        
        # Linearized models
        self.linear_dynamics: List[LinearizedDynamics] = []
        self.cost_approx: List[QuadraticCostApprox] = []
        
        # DDP gains
        self.gains: List[DDPGains] = []
        
        # Regularization
        self.regularization = self.config.regularization
    
    def solve(self,
              initial_trajectory: np.ndarray,
              reference_trajectory: np.ndarray,
              dt: float,
              contact_modes: Optional[List[Any]] = None) -> SCPDDPResult:
        """
        Solve trajectory optimization problem.
        
        Args:
            initial_trajectory: Initial guess for trajectory (T, n)
            reference_trajectory: Reference trajectory to track (T, n)
            dt: Time step
            contact_modes: Contact mode sequence (optional)
            
        Returns:
            SCPDDPResult with optimized trajectory
        """
        start_time = time.time()
        
        # Initialize
        self.x_traj = initial_trajectory.copy()
        self.T, self.n = initial_trajectory.shape
        self.m = self.n - 7  # Assume control dimension = joint DoFs
        
        # Initialize controls (zero)
        self.u_traj = np.zeros((self.T - 1, self.m))
        
        # Cost history
        cost_history = []
        total_ddp_iters = 0
        converged = False
        
        # SCP outer loop
        for scp_iter in range(self.config.max_scp_iterations):
            # Linearize dynamics around current trajectory
            self._linearize_dynamics(dt)
            
            # Quadratic approximation of costs
            self._approximate_costs(reference_trajectory, contact_modes)
            
            # Solve DDP subproblem
            ddp_iters, ddp_converged = self._solve_ddp_subproblem(dt)
            total_ddp_iters += ddp_iters
            
            # Compute cost
            current_cost = self._compute_total_cost(reference_trajectory)
            cost_history.append(current_cost)
            
            if self.config.verbose and scp_iter % self.config.print_interval == 0:
                print(f"SCP iter {scp_iter}: cost = {current_cost:.6f}, "
                      f"DDP iters = {ddp_iters}")
            
            # Check convergence
            if len(cost_history) >= 2:
                cost_change = abs(cost_history[-1] - cost_history[-2])
                if cost_change < self.config.scp_convergence_threshold:
                    converged = True
                    if self.config.verbose:
                        print(f"SCP converged at iteration {scp_iter}")
                    break
        
        solve_time = time.time() - start_time
        
        return SCPDDPResult(
            trajectory=self.x_traj,
            controls=self.u_traj,
            cost_history=cost_history,
            scp_iterations=scp_iter + 1,
            ddp_iterations=total_ddp_iters,
            converged=converged,
            solve_time=solve_time
        )
    
    def _linearize_dynamics(self, dt: float):
        """
        Linearize dynamics around current trajectory.
        
        Uses finite differences for Jacobian computation.
        """
        self.linear_dynamics = []
        
        eps = 1e-5
        
        for t in range(self.T - 1):
            x = self.x_traj[t]
            u = self.u_traj[t] if t < len(self.u_traj) else np.zeros(self.m)
            
            # Compute nominal next state
            x_next_nom = self.dynamics_func(x, u)
            
            # Compute state Jacobian A = ∂f/∂x
            A = np.zeros((self.n, self.n))
            for i in range(self.n):
                x_plus = x.copy()
                x_plus[i] += eps
                x_next_plus = self.dynamics_func(x_plus, u)
                A[:, i] = (x_next_plus - x_next_nom) / eps
            
            # Compute control Jacobian B = ∂f/∂u
            B = np.zeros((self.n, self.m))
            for i in range(self.m):
                u_plus = u.copy()
                u_plus[i] += eps
                x_next_plus = self.dynamics_func(x, u_plus)
                B[:, i] = (x_next_plus - x_next_nom) / eps
            
            # Affine term
            d = x_next_nom - A @ x - B @ u
            
            self.linear_dynamics.append(LinearizedDynamics(A, B, d))
    
    def _approximate_costs(self,
                          reference_trajectory: np.ndarray,
                          contact_modes: Optional[List[Any]]):
        """
        Compute quadratic approximation of costs.
        """
        self.cost_approx = []
        
        for t in range(self.T):
            x = self.x_traj[t]
            x_ref = reference_trajectory[t]
            
            # Running cost for state
            l_x, _ = self.cost_functions.compute_total_gradient(x, None, x_ref)
            
            # Approximate Hessian (Gauss-Newton approximation)
            Q_xx = np.diag(np.ones(self.n)) * self.cost_functions.config.tracking_pos
            Q_uu = np.diag(np.ones(self.m)) * self.cost_functions.config.control_effort
            Q_xu = np.zeros((self.n, self.m))
            
            if t < self.T - 1:
                u = self.u_traj[t]
                _, l_u = self.cost_functions.compute_total_gradient(x, u, x_ref)
                q_u = l_u
            else:
                q_u = np.zeros(self.m)
            
            self.cost_approx.append(QuadraticCostApprox(
                Q_xx=Q_xx,
                Q_uu=Q_uu,
                Q_xu=Q_xu,
                q_x=l_x,
                q_u=q_u
            ))
    
    def _solve_ddp_subproblem(self, dt: float) -> Tuple[int, bool]:
        """
        Solve DDP subproblem for current convex approximation.
        
        Returns:
            Tuple of (iterations, converged)
        """
        converged = False
        
        for ddp_iter in range(self.config.max_ddp_iterations):
            # Backward pass
            self._backward_pass()
            
            # Forward pass with line search
            improvement = self._forward_pass(dt)
            
            # Check convergence
            if improvement < self.config.ddp_convergence_threshold:
                converged = True
                break
            
            # Update regularization
            if improvement < 0:
                self.regularization *= self.config.regularization_factor
            else:
                self.regularization = max(
                    self.regularization / self.config.regularization_factor,
                    self.config.regularization
                )
        
        return ddp_iter + 1, converged
    
    def _backward_pass(self):
        """
        DDP backward pass.
        
        Computes feedback gains by backward value iteration.
        """
        self.gains = []
        
        # Terminal cost-to-go
        V_xx = self.cost_approx[-1].Q_xx
        V_x = self.cost_approx[-1].q_x
        
        # Backward recursion
        for t in range(self.T - 2, -1, -1):
            cost_approx = self.cost_approx[t]
            dyn = self.linear_dynamics[t]
            
            # Q-function
            Q_xx = cost_approx.Q_xx + dyn.A.T @ V_xx @ dyn.A
            Q_uu = cost_approx.Q_uu + dyn.B.T @ V_xx @ dyn.B
            Q_xu = cost_approx.Q_xu + dyn.A.T @ V_xx @ dyn.B
            Q_ux = Q_xu.T
            Q_x = cost_approx.q_x + dyn.A.T @ V_x
            Q_u = cost_approx.q_u + dyn.B.T @ V_x
            
            # Add regularization
            Q_uu_reg = Q_uu + self.regularization * np.eye(self.m)
            
            # Compute gains
            try:
                Q_uu_inv = np.linalg.inv(Q_uu_reg)
            except np.linalg.LinAlgError:
                Q_uu_inv = np.linalg.pinv(Q_uu_reg)
            
            K = -Q_uu_inv @ Q_ux
            k = -Q_uu_inv @ Q_u
            
            self.gains.append(DDPGains(K=K, k=k))
            
            # Update value function
            V_xx = Q_xx + Q_xu @ K + K.T @ Q_ux + K.T @ Q_uu @ K
            V_x = Q_x + Q_xu @ k + K.T @ Q_u + K.T @ Q_uu @ k
        
        # Reverse gains (stored backwards)
        self.gains = self.gains[::-1]
    
    def _forward_pass(self, dt: float) -> float:
        """
        DDP forward pass with line search.
        
        Returns:
            Improvement in cost (negative if worse)
        """
        # Compute current cost
        old_cost = self._compute_trajectory_cost()
        
        # Line search
        for ls_step in range(self.config.line_search_steps):
            alpha = self.config.line_search_factor ** ls_step
            
            if alpha < self.config.min_step_size:
                break
            
            # Rollout with current step size
            new_x, new_u = self._rollout(alpha)
            
            # Compute new cost
            new_cost = self._compute_trajectory_cost_for(new_x, new_u)
            
            if new_cost < old_cost:
                # Accept step
                self.x_traj = new_x
                self.u_traj = new_u
                return old_cost - new_cost
        
        return 0.0  # No improvement
    
    def _rollout(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rollout trajectory with feedback control.
        
        Args:
            alpha: Step size for feedforward term
            
        Returns:
            Tuple of (new trajectory, new controls)
        """
        new_x = np.zeros_like(self.x_traj)
        new_u = np.zeros_like(self.u_traj)
        
        new_x[0] = self.x_traj[0]  # Keep initial state
        
        for t in range(self.T - 1):
            # Feedback control
            dx = new_x[t] - self.x_traj[t]
            du = self.gains[t].K @ dx + alpha * self.gains[t].k
            new_u[t] = self.u_traj[t] + du
            
            # Forward dynamics
            new_x[t+1] = self.linear_dynamics[t].predict(new_x[t], new_u[t])
        
        return new_x, new_u
    
    def _compute_total_cost(self, reference: np.ndarray) -> float:
        """Compute total trajectory cost."""
        return self._compute_trajectory_cost_for(self.x_traj, self.u_traj)
    
    def _compute_trajectory_cost_for(self,
                                     x_traj: np.ndarray,
                                     u_traj: np.ndarray) -> float:
        """Compute cost for given trajectory."""
        total = 0.0
        
        for t in range(self.T):
            cost = self.cost_functions.compute_total_cost(
                state=x_traj[t],
                control=u_traj[t] if t < len(u_traj) else None
            )
            total += cost
        
        return total


def create_scp_ddp_solver(dynamics_func: Callable,
                         cost_config: Optional[CostConfig] = None,
                         solver_config: Optional[SCPDDPConfig] = None) -> SCPDDPSolver:
    """
    Factory function to create SCP-DDP solver.
    
    Args:
        dynamics_func: Dynamics function
        cost_config: Cost function configuration
        solver_config: Solver configuration
        
    Returns:
        Configured SCPDDPSolver
    """
    cost_functions = CostFunctions(cost_config)
    return SCPDDPSolver(dynamics_func, cost_functions, solver_config)
