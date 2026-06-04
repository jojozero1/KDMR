"""
Lie group and Riemannian geometry utilities for QKDMR v2.0.

The configuration space of a humanoid robot is a product of Lie groups:
    Q = SE(3) × SO(3) × ... × SO(2) × R

where:
- SE(3): root pose (position + orientation)
- SO(3): spherical joints (hip, shoulder)
- SO(2): revolute joints (knee, elbow, ankle)
- R: prismatic joints (if any)

First Principle: Optimization on manifolds requires respecting the Riemannian
structure. Euclidean gradient descent on quaternions followed by normalization
is a geometric hack — it does not follow geodesics.

This module provides:
- Exponential and logarithm maps for each Lie group
- The Riemannian metric (Fisher-Rao for rotation groups)
- Natural gradient computation: ∇_R f = G^{-1} ∇_E f
- Geodesic interpolation between configurations
- Parallel transport for correct momentum updates
"""

import numpy as np
from typing import Tuple, Optional, List, Union, Dict
from dataclasses import dataclass
from enum import Enum, auto
from scipy.spatial.transform import Rotation as R
import warnings


class LieGroupType(Enum):
    """Classification of Lie group types in configuration space."""
    SE3 = auto()     # Full pose: R³ × SO(3)
    R3 = auto()      # Translation only
    SO3 = auto()     # Rotation only (spherical joint)
    SO2 = auto()     # Revolute joint (circle group)
    R1 = auto()      # Prismatic joint (real line)


@dataclass
class ManifoldPoint:
    """A point on the configuration manifold.

    Attributes:
        position: R³ component (translation)
        orientation: SO(3) component (quaternion, wxyz scalar-first)
        group_type: Type of Lie group this point lives in
    """
    position: np.ndarray       # (3,) for SE3, None for SO3
    orientation: np.ndarray    # (4,) quaternion (w,x,y,z), scalar-first
    group_type: LieGroupType

    def __post_init__(self):
        self.position = np.asarray(self.position) if self.position is not None else np.zeros(3)
        self.orientation = np.asarray(self.orientation)
        # Ensure unit quaternion
        norm = np.linalg.norm(self.orientation)
        if abs(norm - 1.0) > 1e-10:
            self.orientation = self.orientation / norm


class LieAlgebra:
    """
    Lie algebra elements corresponding to Lie groups.

    se(3) ≈ R⁶: (v, ω) where v ∈ R³ (linear velocity), ω ∈ R³ (angular velocity)
    so(3) ≈ R³: ω (angular velocity vector)
    so(2) ≈ R¹: θ̇ (scalar angular velocity)
    r¹ ≈ R¹: v (scalar velocity)
    """

    @staticmethod
    def se3_from_twist(v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Create se(3) element from linear and angular velocity.

        Args:
            v: Linear velocity (3,)
            w: Angular velocity (3,)

        Returns:
            Twist vector (6,): [v_x, v_y, v_z, ω_x, ω_y, ω_z]
        """
        return np.concatenate([np.asarray(v).ravel()[:3],
                                np.asarray(w).ravel()[:3]])

    @staticmethod
    def twist_to_vw(twist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose twist into linear and angular components."""
        return twist[:3], twist[3:6]

    @staticmethod
    def hat_so3(w: np.ndarray) -> np.ndarray:
        """Hat map: R³ → so(3), creates skew-symmetric matrix.

        [ω]× = [[0, -ωz, ωy],
                [ωz, 0, -ωx],
                [-ωy, ωx, 0]]
        """
        w = np.asarray(w).ravel()[:3]
        return np.array([
            [0,     -w[2],  w[1]],
            [w[2],   0,    -w[0]],
            [-w[1],  w[0],  0   ]
        ])

    @staticmethod
    def vee_so3(S: np.ndarray) -> np.ndarray:
        """Vee map: so(3) → R³, inverse of hat map."""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    @staticmethod
    def hat_se3(twist: np.ndarray) -> np.ndarray:
        """Hat map: R⁶ → se(3) ⊂ R^{4×4}."""
        v, w = twist[:3], twist[3:6]
        S = np.zeros((4, 4))
        S[:3, :3] = LieAlgebra.hat_so3(w)
        S[:3, 3] = v
        return S

    @staticmethod
    def adjoint_SE3(g: np.ndarray) -> np.ndarray:
        """Adjoint map Ad_g: se(3) → se(3).

        For g = (R, p) ∈ SE(3):
        Ad_g = [[R, [p]×R],
                [0, R     ]]
        """
        R_mat = g[:3, :3]
        p = g[:3, 3]
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R_mat
        Ad[:3, 3:] = LieAlgebra.hat_so3(p) @ R_mat
        Ad[3:, 3:] = R_mat
        return Ad


class ExponentialMap:
    """
    Exponential and logarithm maps for Lie groups.

    The exponential map exp: g → G maps from the Lie algebra to the group.
    It sends straight lines through the origin in g to geodesics through
    the identity in G.

    For SO(3): exp(ω) = I + sin(θ)/θ [ω]× + (1-cos(θ))/θ² [ω]×²
               where θ = ||ω||

    For SE(3): exp(ξ) = [exp(ω), V·v]
               where V = I + (1-cos(θ))/θ² [ω]× + (θ-sin(θ))/θ³ [ω]×²
    """

    EPS = 1e-12

    @staticmethod
    def exp_SO3(omega: np.ndarray) -> np.ndarray:
        """SO(3) exponential map: so(3) → SO(3) represented as quaternion.

        Args:
            omega: Angular velocity vector (3,) — the Lie algebra element

        Returns:
            Unit quaternion (4,) in wxyz scalar-first format
        """
        omega = np.asarray(omega).ravel()[:3]
        theta = np.linalg.norm(omega)

        if theta < ExponentialMap.EPS:
            # exp(0) = identity quaternion
            return np.array([1.0, 0.0, 0.0, 0.0])

        axis = omega / theta
        half_theta = theta / 2.0
        w = np.cos(half_theta)
        xyz = np.sin(half_theta) * axis
        return np.array([w, xyz[0], xyz[1], xyz[2]])

    @staticmethod
    def log_SO3(q: np.ndarray) -> np.ndarray:
        """SO(3) logarithm map: SO(3) → so(3).

        Args:
            q: Unit quaternion (4,) wxyz scalar-first

        Returns:
            Angular velocity vector (3,) — the Lie algebra element
        """
        q = np.asarray(q).ravel()
        w, v = q[0], q[1:4]

        # Ensure quaternion is normalized
        norm = np.linalg.norm(q)
        if norm < ExponentialMap.EPS:
            return np.zeros(3)
        w = w / norm
        v = v / norm

        # Clamp w to [-1, 1] for numerical stability
        w = np.clip(w, -1.0, 1.0)

        theta = 2.0 * np.arctan2(np.linalg.norm(v), w)

        if theta < ExponentialMap.EPS:
            return np.zeros(3)

        # log(q) = θ * v/||v||
        return theta * v / np.linalg.norm(v)

    @staticmethod
    def exp_SE3(twist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SE(3) exponential map: se(3) → SE(3).

        Args:
            twist: (6,) [v; ω] — linear and angular velocity

        Returns:
            Tuple of (position (3,), quaternion (4,))
        """
        v = twist[:3]
        omega = twist[3:6]
        theta = np.linalg.norm(omega)

        # Orientation (SO(3) exponential)
        quat = ExponentialMap.exp_SO3(omega)

        if theta < ExponentialMap.EPS:
            # Pure translation
            pos = v
        else:
            omega_hat = LieAlgebra.hat_so3(omega) / theta
            omega_hat2 = omega_hat @ omega_hat

            # V matrix for SE(3) integration
            V = (np.eye(3) +
                 (1.0 - np.cos(theta)) / (theta * theta) * omega_hat +
                 (theta - np.sin(theta)) / (theta * theta * theta) * omega_hat2)

            pos = V @ v * theta  # V @ v scales with step size

        return pos, quat

    @staticmethod
    def log_SE3(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """SE(3) logarithm map: SE(3) → se(3).

        Args:
            position: (3,) translation
            quaternion: (4,) orientation quaternion

        Returns:
            Twist (6,) [v; ω]
        """
        omega = ExponentialMap.log_SO3(quaternion)
        theta = np.linalg.norm(omega)

        if theta < ExponentialMap.EPS:
            return np.concatenate([position, np.zeros(3)])

        omega_hat = LieAlgebra.hat_so3(omega) / theta
        omega_hat2 = omega_hat @ omega_hat

        # V inverse for SE(3)
        V_inv = (np.eye(3) -
                 0.5 * theta * omega_hat +
                 (1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta)))) * omega_hat2)

        v = V_inv @ position / theta

        return np.concatenate([v, omega])

    @staticmethod
    def exp_SO2(theta: float) -> np.ndarray:
        """SO(2) exponential map: R → SO(2) as unit complex number."""
        return np.array([np.cos(theta), np.sin(theta)])

    @staticmethod
    def log_SO2(z: np.ndarray) -> float:
        """SO(2) logarithm: unit complex → angle."""
        return np.arctan2(z[1], z[0])

    @staticmethod
    def left_multiply_SE3(pos: np.ndarray, quat: np.ndarray,
                           delta_pos: np.ndarray, delta_quat: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Left-multiply SE(3) transformation: g_new = g_delta ∘ g.

        Applies a local update to a global pose.

        Args:
            pos: Current global position (3,)
            quat: Current global orientation quaternion (4,) wxyz
            delta_pos: Local position update (3,)
            delta_quat: Local orientation update quaternion (4,) wxyz

        Returns:
            Tuple of (new_position, new_orientation)
        """
        # Rotate delta position by current orientation
        R_cur = R.from_quat(quat[1:4].tolist() + [quat[0]], scalar_first=True)
        # Actually, let's use consistent scalar_first=True convention
        R_cur = R.from_quat([quat[1], quat[2], quat[3], quat[0]], scalar_first=True)
        new_pos = pos + R_cur.apply(delta_pos)

        # Multiply orientations
        q_cur = quat.copy()
        q_delta = delta_quat.copy()

        w1, x1, y1, z1 = q_delta
        w2, x2, y2, z2 = q_cur
        new_quat = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

        norm = np.linalg.norm(new_quat)
        if norm > 0:
            new_quat = new_quat / norm

        return new_pos, new_quat


class RiemannianMetric:
    """
    Riemannian metrics on Lie groups for natural gradient computation.

    First Principle: The natural gradient G⁻¹∇f is the direction of steepest
    descent in the Riemannian sense — it is parametrization-invariant and
    follows geodesics.

    For SO(3), the Fisher-Rao metric is G = tr(JᵀJ) where J is the Jacobian
    of the exponential map. This gives:
        G_SO3(ω) = I + (1-cos(θ))/θ² [ω]×ᵀ[ω]× + θ⁻²(1 - sin(θ)/θ) ωωᵀ
    """

    @staticmethod
    def metric_SO3(omega: np.ndarray) -> np.ndarray:
        """Compute the Riemannian metric on SO(3) at a given tangent vector.

        Args:
            omega: Tangent vector in so(3) (angular velocity)

        Returns:
            Metric tensor (3, 3) symmetric positive-definite
        """
        omega = np.asarray(omega).ravel()[:3]
        theta = np.linalg.norm(omega)

        if theta < 1e-10:
            return np.eye(3)

        omega_hat = LieAlgebra.hat_so3(omega)
        omega_hat_T = omega_hat.T

        # Fisher-Rao metric for SO(3)
        coeff1 = (1.0 - np.cos(theta)) / (theta * theta)
        coeff2 = (1.0 / (theta * theta)) * (1.0 - np.sin(theta) / theta)

        G = (np.eye(3) +
             coeff1 * (omega_hat_T @ omega_hat) +
             coeff2 * np.outer(omega, omega))

        # Ensure symmetry
        G = 0.5 * (G + G.T)

        return G

    @staticmethod
    def metric_inverse_SO3(omega: np.ndarray) -> np.ndarray:
        """Compute inverse Riemannian metric on SO(3)."""
        G = RiemannianMetric.metric_SO3(omega)
        try:
            return np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(G)

    @staticmethod
    def metric_SE3(twist: np.ndarray) -> np.ndarray:
        """Compute Riemannian metric on SE(3).

        SE(3) metric is block-diagonal with:
        - R³: Euclidean metric (identity, scaled by mass)
        - SO(3): Fisher-Rao metric
        """
        omega = twist[3:6]
        G = np.eye(6)
        G[:3, :3] = np.eye(3)  # R³ Euclidean
        G[3:, 3:] = RiemannianMetric.metric_SO3(omega)
        return G

    @staticmethod
    def metric_inverse_SE3(twist: np.ndarray) -> np.ndarray:
        """Compute inverse Riemannian metric on SE(3)."""
        G = RiemannianMetric.metric_SE3(twist)
        try:
            return np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(G)

    @staticmethod
    def natural_gradient_SO3(q: np.ndarray,
                              euclidean_grad: np.ndarray) -> np.ndarray:
        """Compute the natural gradient on SO(3).

        ∇_R f = G^{-1} ∇_E f

        Args:
            q: Current quaternion (4,) wxyz
            euclidean_grad: Euclidean gradient (4,) w.r.t. quaternion

        Returns:
            Natural gradient as Lie algebra element (3,)
        """
        q = np.asarray(q).ravel()
        euclidean_grad = np.asarray(euclidean_grad).ravel()

        # Project Euclidean gradient onto tangent space at q
        # T_q S³: remove component along q (radial direction)
        # Then map to so(3) via the isometric identification
        radial_component = np.dot(euclidean_grad, q)
        tangent_grad = euclidean_grad - radial_component * q

        # Map tangent vector at q to Lie algebra so(3)
        # The mapping is: ω = 2 * (q^{-1} * tangent_grad)_{1:4}
        # where q^{-1} = (w, -x, -y, -z) for unit quaternion
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])

        # Quaternion multiply q_inv * tangent_grad
        w1, x1, y1, z1 = q_inv
        w2, x2, y2, z2 = tangent_grad
        prod = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

        # Lie algebra element = 2 * imaginary part
        omega_euclidean = 2.0 * prod[1:4]

        # Apply metric inverse for natural gradient
        G_inv = RiemannianMetric.metric_inverse_SO3(omega_euclidean)
        return G_inv @ omega_euclidean

    @staticmethod
    def natural_gradient_SE3(pos: np.ndarray, quat: np.ndarray,
                              grad_pos: np.ndarray, grad_quat: np.ndarray
                              ) -> np.ndarray:
        """Compute natural gradient on SE(3).

        Returns:
            Natural gradient twist (6,) in Lie algebra se(3)
        """
        # Position: Euclidean = Riemannian for R³
        v_natural = grad_pos[:3]

        # Rotation: Fisher-Rao natural gradient
        omega_natural = RiemannianMetric.natural_gradient_SO3(quat, grad_quat)

        return np.concatenate([v_natural, omega_natural])


class GeodesicInterpolation:
    """
    Geodesic interpolation on Lie groups.

    In Euclidean space, linear interpolation is straight lines.
    On curved manifolds, geodesics are the "straightest possible" curves.

    For SO(3): geodesic = constant angular velocity → SLERP
    For SE(3): geodesic = screw motion (constant twist)
    """

    @staticmethod
    def geodesic_SO3(q1: np.ndarray, q2: np.ndarray, s: float) -> np.ndarray:
        """Geodesic interpolation on SO(3) — SLERP.

        Args:
            q1: Start quaternion (4,)
            q2: End quaternion (4,)
            s: Interpolation parameter in [0, 1]

        Returns:
            Interpolated quaternion (4,)
        """
        q1 = np.asarray(q1).ravel()
        q2 = np.asarray(q2).ravel()

        # Normalize
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        # Ensure shortest path
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot

        # Clamp for numerical stability
        dot = np.clip(dot, -1.0, 1.0)

        # Compute geodesic
        if dot > 0.9995:
            # Quaternions are close — linear interpolation
            result = q1 + s * (q2 - q1)
            return result / np.linalg.norm(result)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        w1 = np.sin((1 - s) * theta) / sin_theta
        w2 = np.sin(s * theta) / sin_theta

        return w1 * q1 + w2 * q2

    @staticmethod
    def geodesic_SE3(pos1: np.ndarray, quat1: np.ndarray,
                     pos2: np.ndarray, quat2: np.ndarray,
                     s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Geodesic interpolation on SE(3) — screw motion.

        Uses the logarithm to map to se(3), linear interpolation in Lie
        algebra, then exponential back to SE(3).
        """
        # Compute relative transformation: g2 = g_delta ∘ g1
        # g_delta = g2 ∘ g1^{-1}

        # Invert g1
        quat1_inv = np.array([quat1[0], -quat1[1], -quat1[2], -quat1[3]])
        quat1_inv = quat1_inv / np.linalg.norm(quat1_inv)

        # Compute delta: (g1^{-1} applied to pos2)
        R1_inv = R.from_quat([quat1_inv[1], quat1_inv[2], quat1_inv[3], quat1_inv[0]],
                             scalar_first=True)
        delta_pos = R1_inv.apply(pos2 - pos1)

        # Delta orientation
        w1, x1, y1, z1 = quat1_inv
        w2, x2, y2, z2 = quat2
        delta_quat = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        delta_quat = delta_quat / np.linalg.norm(delta_quat)

        # Lie algebra of delta
        twist = ExponentialMap.log_SE3(delta_pos, delta_quat)

        # Interpolate in Lie algebra (straight line)
        twist_s = s * twist

        # Map back to SE(3)
        delta_pos_s, delta_quat_s = ExponentialMap.exp_SE3(twist_s)

        # Apply to start: g(s) = g_delta(s) ∘ g1
        return ExponentialMap.left_multiply_SE3(pos1, quat1, delta_pos_s, delta_quat_s)

    @staticmethod
    def geodesic_distance_SO3(q1: np.ndarray, q2: np.ndarray) -> float:
        """Geodesic (angular) distance on SO(3)."""
        q1 = np.asarray(q1).ravel()
        q2 = np.asarray(q2).ravel()
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)

    @staticmethod
    def parallel_transport_SO3(q: np.ndarray, v: np.ndarray,
                                q_target: np.ndarray) -> np.ndarray:
        """Parallel transport a tangent vector from T_q SO(3) to T_{q_target} SO(3).

        This is needed to correctly transport momentum vectors between
        different points on the manifold during optimization.

        Args:
            q: Source quaternion (4,)
            v: Tangent vector at q (4,) — must satisfy v·q = 0
            q_target: Target quaternion (4,)

        Returns:
            Parallel transported vector at q_target (4,)
        """
        # Use the isometric identification of tangent spaces
        # via the Lie group left-translation
        q = np.asarray(q).ravel()
        q = q / np.linalg.norm(q)
        q_target = np.asarray(q_target).ravel()
        q_target = q_target / np.linalg.norm(q_target)

        # Map v to Lie algebra: ω = 2 * (q^{-1} * v)_{1:4}
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
        w1, x1, y1, z1 = q_inv
        w2, x2, y2, z2 = v
        v_lie = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        omega = 2.0 * v_lie[1:4]

        # Transport by left-translation: v' = q_target * (0, ω/2)
        half_omega_quat = np.array([0.0, omega[0]/2, omega[1]/2, omega[2]/2])
        w1, x1, y1, z1 = q_target
        w2, x2, y2, z2 = half_omega_quat
        v_transported = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

        return v_transported
