"""
Visualization utilities for KDMR.

This module provides visualization tools for:
- MuJoCo robot visualization
- GRF curve plotting
- Contact mode visualization
- Comparison plots between GMR and KDMR
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

try:
    import mujoco as mj
    import mujoco.viewer as mjv
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from kdmr.contact.contact_mode import ContactMode


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    fps: float = 30.0
    camera_distance: float = 3.0
    camera_elevation: float = -20.0
    camera_azimuth: float = 180.0
    show_human: bool = True
    show_grf: bool = True
    show_contact_modes: bool = True
    video_width: int = 640
    video_height: int = 480


class KDMRVisualizer:
    """
    Visualization class for KDMR results.
    
    Provides both MuJoCo-based 3D visualization and matplotlib-based
    2D plotting capabilities.
    """
    
    def __init__(self, 
                 robot_xml_path: str,
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            robot_xml_path: Path to MuJoCo XML file
            config: Visualization configuration
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required for 3D visualization")
        
        self.robot_xml_path = Path(robot_xml_path)
        self.config = config or VisualizationConfig()
        
        # Load MuJoCo model
        self.model = mj.MjModel.from_xml_path(str(self.robot_xml_path))
        self.data = mj.MjData(self.model)
        
        # Initialize viewer
        self.viewer = None
        self.renderer = None
        
        # Video recording
        self.recording = False
        self.video_frames = []
    
    def launch_viewer(self):
        """Launch MuJoCo passive viewer."""
        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False
        )
        
        # Set camera
        self.viewer.cam.distance = self.config.camera_distance
        self.viewer.cam.elevation = self.config.camera_elevation
        self.viewer.cam.azimuth = self.config.camera_azimuth
    
    def update_robot(self, 
                     qpos: np.ndarray,
                     root_pos: Optional[np.ndarray] = None,
                     root_quat: Optional[np.ndarray] = None,
                     joint_angles: Optional[np.ndarray] = None):
        """
        Update robot configuration.
        
        Args:
            qpos: Full joint position vector, or
            root_pos: Root position (3,)
            root_quat: Root quaternion (4,)
            joint_angles: Joint angles (nq-7,)
        """
        if qpos is not None:
            self.data.qpos[:] = qpos
        else:
            if root_pos is not None:
                self.data.qpos[:3] = root_pos
            if root_quat is not None:
                self.data.qpos[3:7] = root_quat
            if joint_angles is not None:
                self.data.qpos[7:] = joint_angles
        
        mj.mj_forward(self.model, self.data)
    
    def step(self, 
             qpos: np.ndarray,
             human_data: Optional[Dict] = None,
             grf_data: Optional[Dict] = None,
             contact_modes: Optional[Dict] = None,
             sync: bool = True):
        """
        Perform one visualization step.
        
        Args:
            qpos: Robot joint positions
            human_data: Human motion data for overlay
            grf_data: GRF data for visualization
            contact_modes: Contact mode data
            sync: Whether to sync viewer
        """
        self.update_robot(qpos)
        
        if human_data is not None and self.config.show_human:
            self._draw_human_skeleton(human_data)
        
        if grf_data is not None and self.config.show_grf:
            self._draw_grf_vectors(grf_data)
        
        if contact_modes is not None and self.config.show_contact_modes:
            self._draw_contact_indicators(contact_modes)
        
        if sync and self.viewer is not None:
            self.viewer.sync()
        
        if self.recording:
            self._capture_frame()
    
    def _draw_human_skeleton(self, human_data: Dict):
        """Draw human skeleton overlay."""
        if self.viewer is None:
            return
        
        # Clear previous geometry
        self.viewer.user_scn.ngeom = 0
        
        # Draw skeleton points
        for joint_name, (pos, quat) in human_data.items():
            self._draw_sphere(pos, radius=0.02, rgba=[0, 1, 0, 0.5])
    
    def _draw_grf_vectors(self, grf_data: Dict):
        """Draw GRF force vectors."""
        if self.viewer is None:
            return
        
        for foot_name, force in grf_data.items():
            # Get foot position
            foot_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, foot_name)
            if foot_id >= 0:
                foot_pos = self.data.xpos[foot_id]
                self._draw_arrow(foot_pos, force, scale=0.001, rgba=[1, 0, 0, 0.8])
    
    def _draw_contact_indicators(self, contact_modes: Dict):
        """Draw contact mode indicators."""
        if self.viewer is None:
            return
        
        colors = {
            ContactMode.HEEL: [1, 0, 0, 0.8],      # Red
            ContactMode.FLAT: [0, 1, 0, 0.8],      # Green
            ContactMode.TOE: [0, 0, 1, 0.8],       # Blue
            ContactMode.SWING: [0.5, 0.5, 0.5, 0.3] # Gray
        }
        
        for foot_name, mode in contact_modes.items():
            foot_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, foot_name)
            if foot_id >= 0:
                foot_pos = self.data.xpos[foot_id]
                rgba = colors.get(mode, [1, 1, 1, 0.5])
                self._draw_sphere(foot_pos + [0, 0, 0.1], radius=0.03, rgba=rgba)
    
    def _draw_sphere(self, pos: np.ndarray, radius: float, rgba: List[float]):
        """Draw a sphere at given position."""
        if self.viewer is None:
            return
        
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=rgba
        )
        self.viewer.user_scn.ngeom += 1
    
    def _draw_arrow(self, 
                    start: np.ndarray, 
                    direction: np.ndarray,
                    scale: float = 1.0,
                    rgba: List[float] = None):
        """Draw an arrow from start in given direction."""
        if self.viewer is None:
            return
        
        rgba = rgba or [1, 0, 0, 1]
        end = start + direction * scale
        
        mj.mjv_connector(
            self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.01,
            from_=start,
            to=end
        )
        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom].rgba = rgba
        self.viewer.user_scn.ngeom += 1
    
    def start_recording(self, output_path: str):
        """Start video recording."""
        self.recording = True
        self.video_path = Path(output_path)
        self.video_frames = []
        
        # Initialize renderer
        self.renderer = mj.Renderer(
            self.model,
            height=self.config.video_height,
            width=self.config.video_width
        )
    
    def _capture_frame(self):
        """Capture current frame for video."""
        if self.renderer is not None:
            self.renderer.update_scene(self.data)
            img = self.renderer.render()
            self.video_frames.append(img)
    
    def stop_recording(self):
        """Stop recording and save video."""
        if not self.recording:
            return
        
        self.recording = False
        
        if self.video_frames:
            try:
                import imageio
                self.video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(
                    str(self.video_path),
                    self.video_frames,
                    fps=self.config.fps
                )
            except ImportError:
                print("imageio required for video recording")
        
        self.video_frames = []
    
    def close(self):
        """Close viewer."""
        if self.recording:
            self.stop_recording()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class GRFPlotter:
    """
    Plotting utilities for GRF data analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")
        
        self.figsize = figsize
    
    def plot_grf_comparison(self,
                           grf_human: np.ndarray,
                           grf_gmr: np.ndarray,
                           grf_kdmr: np.ndarray,
                           timestamps: np.ndarray,
                           title: str = "GRF Comparison"):
        """
        Plot GRF comparison between human, GMR, and KDMR.
        
        Args:
            grf_human: Human GRF data, shape (N, 3)
            grf_gmr: GMR-estimated GRF, shape (N, 3)
            grf_kdmr: KDMR-estimated GRF, shape (N, 3)
            timestamps: Time stamps, shape (N,)
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        
        labels = ['Fx (Anterior)', 'Fy (Lateral)', 'Fz (Vertical)']
        colors = {'human': 'blue', 'gmr': 'red', 'kdmr': 'green'}
        
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(timestamps, grf_human[:, i], 
                   color=colors['human'], label='Human', linewidth=2)
            ax.plot(timestamps, grf_gmr[:, i], 
                   color=colors['gmr'], label='GMR', linestyle='--', linewidth=1.5)
            ax.plot(timestamps, grf_kdmr[:, i], 
                   color=colors['kdmr'], label='KDMR', linestyle='-.', linewidth=1.5)
            
            ax.set_ylabel(f'{label} Force (N)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(title)
        fig.tight_layout()
        
        return fig, axes
    
    def plot_contact_modes(self,
                          contact_modes: List[ContactMode],
                          timestamps: np.ndarray,
                          grf_vertical: Optional[np.ndarray] = None):
        """
        Plot contact mode sequence with optional GRF overlay.
        
        Args:
            contact_modes: List of contact modes
            timestamps: Time stamps
            grf_vertical: Optional vertical GRF for overlay
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Plot contact modes
        ax1 = axes[0]
        mode_colors = {
            ContactMode.HEEL: 'red',
            ContactMode.FLAT: 'green',
            ContactMode.TOE: 'blue',
            ContactMode.SWING: 'gray'
        }
        
        for i, (mode, t) in enumerate(zip(contact_modes, timestamps[:-1])):
            color = mode_colors.get(mode, 'black')
            ax1.axvspan(t, timestamps[i+1], alpha=0.3, color=color)
        
        ax1.set_ylabel('Contact Mode')
        ax1.set_yticks([])
        ax1.set_title('Contact Mode Sequence')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='Heel'),
            Patch(facecolor='green', alpha=0.3, label='Flat'),
            Patch(facecolor='blue', alpha=0.3, label='Toe'),
            Patch(facecolor='gray', alpha=0.3, label='Swing')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot GRF if provided
        ax2 = axes[1]
        if grf_vertical is not None:
            ax2.plot(timestamps[:len(grf_vertical)], grf_vertical, 
                    color='black', linewidth=2)
            ax2.set_ylabel('Vertical GRF (N)')
            ax2.set_xlabel('Time (s)')
            ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig, axes
    
    def plot_trajectory_comparison(self,
                                  traj_gmr: Dict[str, np.ndarray],
                                  traj_kdmr: Dict[str, np.ndarray],
                                  timestamps: np.ndarray,
                                  joint_names: List[str]):
        """
        Plot trajectory comparison between GMR and KDMR.
        
        Args:
            traj_gmr: GMR trajectory data
            traj_kdmr: KDMR trajectory data
            timestamps: Time stamps
            joint_names: Joint names to plot
            
        Returns:
            Tuple of (figure, axes)
        """
        n_joints = len(joint_names)
        fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2*n_joints), sharex=True)
        
        if n_joints == 1:
            axes = [axes]
        
        for ax, joint_name in zip(axes, joint_names):
            if joint_name in traj_gmr:
                ax.plot(timestamps, traj_gmr[joint_name], 
                       label='GMR', color='red', linestyle='--')
            if joint_name in traj_kdmr:
                ax.plot(timestamps, traj_kdmr[joint_name], 
                       label='KDMR', color='green', linestyle='-.')
            
            ax.set_ylabel(f'{joint_name}\n(rad)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout()
        
        return fig, axes
    
    def plot_metrics_comparison(self,
                               metrics_gmr: Dict[str, float],
                               metrics_kdmr: Dict[str, float],
                               metric_names: List[str]):
        """
        Create bar chart comparing metrics between GMR and KDMR.
        
        Args:
            metrics_gmr: GMR metrics
            metrics_kdmr: KDMR metrics
            metric_names: Names of metrics to compare
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        gmr_values = [metrics_gmr.get(m, 0) for m in metric_names]
        kdmr_values = [metrics_kdmr.get(m, 0) for m in metric_names]
        
        bars1 = ax.bar(x - width/2, gmr_values, width, label='GMR', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, kdmr_values, width, label='KDMR', color='green', alpha=0.7)
        
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        fig.tight_layout()
        return fig, ax
    
    def save_figure(self, fig: 'Figure', filepath: str, dpi: int = 150):
        """Save figure to file."""
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
