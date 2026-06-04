"""
Diffusion trajectory prior for QKDMR v2.0.

First Principle: Most human-to-robot motion pairs lie on a low-dimensional
manifold in the space of trajectories. A score-based generative model
learns this manifold and provides:

1. Warm-start trajectories that are already "close" to the optimum
2. The score function ∇log p(q) as a regularizer (pulls toward manifold)
3. Morphology-conditional generation (different robots, different styles)

The diffusion model is trained on pairs of (human_motion, robot_trajectory)
and learns the conditional distribution p(robot_traj | human_motion).

At inference time, the model "denoises" from Gaussian noise to a
physically plausible trajectory — this replaces the kinematic IK + SCP-DDP
initialization with a single learned forward pass.

References:
- Song & Ermon, "Score-based generative modeling" (2019)
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis" (2022)
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DiffusionConfig:
    """Configuration for diffusion trajectory prior."""
    # Architecture
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1

    # Diffusion process
    num_diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"  # "linear" or "cosine"

    # Conditioning
    condition_dim: int = 256   # Human motion embedding dimension
    use_cross_attention: bool = True

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 1000
    ema_decay: float = 0.995


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TrajectoryTransformer(nn.Module):
    """Transformer-based denoiser for trajectory data.

    Trajectories are sequences of (T, nq) where nq includes both
    positions and orientations. The transformer processes the full
    sequence with temporal attention, capturing both local (smoothness)
    and global (periodicity, gait) structure.
    """

    def __init__(self,
                 nq: int,
                 T: int,
                 config: DiffusionConfig):
        super().__init__()
        self.nq = nq
        self.T = T
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(nq + config.condition_dim, config.hidden_dim)

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_heads,
        )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, nq)

        # Position embedding
        self.pos_emb = nn.Parameter(
            torch.randn(1, T, config.hidden_dim) * 0.02
        )

    def forward(self,
                x_t: torch.Tensor,       # (B, T, nq) noisy trajectory
                t: torch.Tensor,          # (B,) diffusion timestep
                condition: torch.Tensor   # (B, T, condition_dim) human motion
                ) -> torch.Tensor:
        """Predict the noise added to the trajectory.

        Args:
            x_t: Noisy trajectory at timestep t
            t: Diffusion timestep (0 = clean, T = pure noise)
            condition: Human motion encoding (same temporal resolution)

        Returns:
            Predicted noise (B, T, nq)
        """
        B = x_t.shape[0]

        # Concatenate trajectory with condition
        x_cond = torch.cat([x_t, condition], dim=-1)  # (B, T, nq + cond_dim)

        # Project to hidden dimension
        h = self.input_proj(x_cond)  # (B, T, hidden_dim)

        # Add position embedding
        h = h + self.pos_emb

        # Add time embedding
        t_emb = self.time_emb(t)[:, None, :]  # (B, 1, hidden_dim)
        h = h + t_emb

        # Transformer
        h = self.transformer(h)  # (B, T, hidden_dim)

        # Output projection
        noise_pred = self.output_proj(h)  # (B, T, nq)

        return noise_pred


class DiffusionTrajectoryPrior:
    """
    Diffusion model for trajectory generation.

    The forward (noising) process:
        q(x_t | x_0) = N(√(ᾱ_t) · x_0, (1 - ᾱ_t) · I)

    The reverse (denoising) process:
        p_θ(x_{t-1} | x_t, c) = N(μ_θ(x_t, t, c), σ_t² · I)

    where μ_θ is the learned denoising network (TrajectoryTransformer)
    parameterized by θ, and c is the human motion conditioning.

    At inference time, we start from x_T ~ N(0, I) and iteratively denoise
    to get x_0 ~ p(robot_traj | human_motion).
    """

    def __init__(self,
                 nq: int,
                 T: int,
                 config: Optional[DiffusionConfig] = None):
        """Initialize diffusion prior.

        Args:
            nq: Robot configuration dimension
            T: Trajectory length
            config: Diffusion configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for diffusion prior")

        self.nq = nq
        self.T = T
        self.config = config or DiffusionConfig()

        # Build noise schedule
        self.betas = self._build_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), self.alphas_cumprod[:-1]
        ])

        # Model (created if needed)
        self.model: Optional[TrajectoryTransformer] = None

        # Score function cache
        self._cached_trajectory = None
        self._cached_score = None

    def _build_beta_schedule(self) -> torch.Tensor:
        """Build noise schedule (β₁, ..., β_T)."""
        T = self.config.num_diffusion_steps

        if self.config.beta_schedule == "linear":
            return torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                T
            )
        elif self.config.beta_schedule == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021)
            s = 0.008
            steps = T + 1
            x = torch.linspace(0, T, steps) / T
            alphas_cumprod = torch.cos((x + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {self.config.beta_schedule}")

    def forward_diffusion(self, x_0: torch.Tensor, t: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise: x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε.

        Args:
            x_0: Clean trajectory (B, T, nq)
            t: Timestep indices (B,)

        Returns:
            Tuple of (x_t, noise ε)
        """
        noise = torch.randn_like(x_0)
        alpha_bar = self.alphas_cumprod[t].to(x_0.device)
        alpha_bar = alpha_bar[:, None, None]  # (B, 1, 1)

        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * noise
        return x_t, noise

    def reverse_diffusion(self,
                           x_t: torch.Tensor,
                           t: torch.Tensor,
                           condition: torch.Tensor,
                           noise_pred: torch.Tensor
                           ) -> torch.Tensor:
        """One reverse step: x_{t-1} = μ_θ(x_t, t, c) + σ_t · z.

        Uses DDPM sampling (Ho et al., 2020).
        """
        alpha = self.alphas[t].to(x_t.device)[:, None, None]
        alpha_bar = self.alphas_cumprod[t].to(x_t.device)[:, None, None]
        alpha_bar_prev = self.alphas_cumprod_prev[t].to(x_t.device)[:, None, None]
        beta = self.betas[t].to(x_t.device)[:, None, None]

        # Predicted x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

        # Clamp to [-1, 1] range (assuming normalized data)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Mean: μ̃_t = √(ᾱ_{t-1})·β_t/(1-ᾱ_t)·x_0_pred + √(α_t)·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t
        coef1 = torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar)
        coef2 = torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
        mean = coef1 * x_0_pred + coef2 * x_t

        # Variance
        if t[0] > 0:
            var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(var) * noise
        else:
            return mean

    @torch.no_grad()
    def sample(self,
               human_motion: np.ndarray,
               robot_dims: Tuple[int, int],
               condition_encoder: Optional[Callable] = None,
               guidance_scale: float = 0.0) -> np.ndarray:
        """Sample a robot trajectory conditioned on human motion.

        Args:
            human_motion: Human motion data (T, n_joints, 3)
            robot_dims: (T, nq) output dimensions
            condition_encoder: Function to encode human motion
            guidance_scale: Classifier-free guidance weight

        Returns:
            Sampled robot trajectory (T, nq)
        """
        if self.model is None:
            # No trained model — return random initialization
            T_out, nq = robot_dims
            return np.random.randn(T_out, nq) * 0.01

        T_out, nq = robot_dims
        B = 1  # Batch size 1 for inference
        device = next(self.model.parameters()).device

        # Encode human motion
        if condition_encoder is not None:
            condition = condition_encoder(human_motion)  # (B, T, cond_dim)
        else:
            # Simple encoding: flatten and project
            human_flat = human_motion.reshape(human_motion.shape[0], -1)
            condition = torch.tensor(human_flat, dtype=torch.float32)
            condition = condition[None, :, :]  # (1, T, J*3)
            # Pad or truncate to match condition_dim
            cond_dim = self.config.condition_dim
            if condition.shape[-1] < cond_dim:
                padding = torch.zeros(B, T_out, cond_dim - condition.shape[-1])
                condition = torch.cat([condition, padding], dim=-1)
            condition = condition[:, :, :cond_dim]
            condition = torch.tensor(condition, dtype=torch.float32, device=device)

        # Start from pure noise
        x_t = torch.randn(B, T_out, nq, device=device)

        # Denoise iteratively
        for step in reversed(range(self.config.num_diffusion_steps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(x_t, t, condition)

            # Classifier-free guidance
            if guidance_scale > 0:
                null_condition = torch.zeros_like(condition)
                noise_uncond = self.model(x_t, t, null_condition)
                noise_pred = (noise_uncond +
                               guidance_scale * (noise_pred - noise_uncond))

            # Reverse step
            x_t = self.reverse_diffusion(x_t, t, condition, noise_pred)

        return x_t[0].cpu().numpy()  # (T, nq)

    def score(self, trajectory: np.ndarray,
              human_motion: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the score function ∇_q log p(q | c).

        The score points toward higher-density regions of the learned
        trajectory manifold. It can be used as a regularizer in the
        Hamiltonian optimizer:

            S_total = S_physics + S_task + λ · (-log p(q|c))

        where -log p is the negative log-likelihood under the prior.

        First principle: This is the information-theoretic regularizer —
        it penalizes trajectories that are "unlikely" under the learned
        distribution of physically valid motion.

        Args:
            trajectory: Robot trajectory (T, nq)
            human_motion: Conditioning human motion

        Returns:
            Score (T, nq) — gradient of log-density
        """
        if self.model is None:
            return np.zeros_like(trajectory)

        # Cache check
        if (self._cached_trajectory is not None and
                np.allclose(trajectory, self._cached_trajectory)):
            return self._cached_score

        device = next(self.model.parameters()).device
        x = torch.tensor(trajectory[None], dtype=torch.float32, device=device)
        x.requires_grad_(True)

        # Encode human motion
        if human_motion is not None:
            condition = torch.tensor(
                human_motion.reshape(1, human_motion.shape[0], -1),
                dtype=torch.float32, device=device
            )
            cond_dim = self.config.condition_dim
            if condition.shape[-1] < cond_dim:
                padding = torch.zeros(1, condition.shape[1],
                                       cond_dim - condition.shape[-1],
                                       device=device)
                condition = torch.cat([condition, padding], dim=-1)
            condition = condition[:, :, :cond_dim]
        else:
            condition = torch.zeros(1, len(trajectory),
                                     self.config.condition_dim,
                                     device=device)

        # Score from denoising score matching:
        # s_θ(x, c) ≈ -ε_θ(x, t, c) / √(1-ᾱ_t)
        # For score at clean data, use small t
        t = torch.tensor([1], device=device)  # Small timestep
        alpha_bar = self.alphas_cumprod[t[0]]

        # Get noise prediction for a slightly noised version
        x_t, _ = self.forward_diffusion(x, t)
        noise_pred = self.model(x_t, t, condition)

        # Score approximation
        score_tensor = -noise_pred / torch.sqrt(1.0 - alpha_bar + 1e-8)

        # Backpropagate to get gradient w.r.t. x
        score_norm = torch.sum(score_tensor)
        grad = torch.autograd.grad(score_norm, x, create_graph=False)[0]

        score_np = grad[0].detach().cpu().numpy()

        # Cache
        self._cached_trajectory = trajectory.copy()
        self._cached_score = score_np

        return score_np

    def load_pretrained(self, model_path: str):
        """Load pretrained model weights."""
        if not TORCH_AVAILABLE:
            return

        self.model = TrajectoryTransformer(
            self.nq, self.T, self.config
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        self.model.eval()


class MorphologyEncoder:
    """
    Encode robot morphology for morphology-conditioned generation.

    Different robots have different kinematics (number of joints, link
    lengths, joint limits). A morphology-aware diffusion model can
    generate trajectories for any robot in the training distribution.

    First principle: The mapping from human to robot motion depends on
    the robot's embodiment. By conditioning the diffusion model on a
    morphology embedding, we can handle multiple robots with a single model.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

        # Morphology features to encode:
        # - Number of joints per limb
        # - Link lengths (normalized)
        # - Joint types (revolute/prismatic/spherical)
        # - Mass distribution
        # - Joint limits

    def encode(self,
               n_joints: int,
               link_lengths: np.ndarray,
               joint_types: List[str],
               mass: float = 35.0) -> np.ndarray:
        """Encode robot morphology into fixed-dimensional embedding.

        Uses random Fourier features for a continuous embedding space.
        """
        # Collect features
        features = [
            float(n_joints),
            mass / 100.0,  # Normalize by ~100kg
            float(len([j for j in joint_types if 'revolute' in j])) / n_joints,
            float(len([j for j in joint_types if 'prismatic' in j])) / n_joints,
        ]

        # Add link length statistics
        if len(link_lengths) > 0:
            features.extend([
                np.mean(link_lengths),
                np.std(link_lengths),
                np.min(link_lengths),
                np.max(link_lengths),
            ])

        features = np.array(features)

        # Random Fourier features
        if hasattr(self, 'B'):
            projection = self.B @ features
            embedding = np.concatenate([
                np.cos(projection),
                np.sin(projection)
            ])
            return embedding[:self.embedding_dim]

        return features[:self.embedding_dim]
