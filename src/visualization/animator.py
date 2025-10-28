"""
High-performance parallel animation generator
Supports multi-threaded parallel rendering and out-of-order composition
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import io
from PIL import Image
from tqdm import tqdm
import imageio
from multiprocessing import Pool, cpu_count
import gc

from src.core.models import TimeSeriesData, DragonConfig, SimulationState
from src.visualization.renderer import DragonRenderer


def render_single_frame(args: Tuple[int, Dict[str, Any]]) -> Tuple[int, np.ndarray]:
    """
    Render a single frame (top-level function for multiprocessing support)
    
    Args:
        args: (frame_index, render_params)
        
    render_params contains:
        - state: SimulationState
        - config: DragonConfig
        - xlim: tuple
        - ylim: tuple
        - highlight_indices: Optional[List[int]]
        - dpi: int
        - figsize: tuple
        - path_handler: Optional[PathHandler] - 路径处理器
        - path_range: Optional[tuple] - 路径参数范围
        
    Returns:
        (frame_index, frame_array)
    """
    frame_idx, params = args
    
    state = params['state']
    config = params['config']
    xlim = params['xlim']
    ylim = params['ylim']
    highlight_indices = params.get('highlight_indices')
    dpi = params['dpi']
    figsize = params['figsize']
    path_handler = params.get('path_handler')
    path_range = params.get('path_range')
    
    # Create renderer
    renderer = DragonRenderer(config)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Render path first (so it's in background)
    if path_handler is not None and path_range is not None:
        renderer.render_path(ax, path_handler, path_range, num_points=2000, label='路径')
    
    # Render dragon
    renderer.render_dragon(state, ax, highlight_indices, show_handles=True)
    
    # Setup axes
    renderer.setup_axes(ax, xlim, ylim, title=f'Time: {state.time:.2f}s')
    
    # Convert to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    frame = np.array(img)[:, :, :3]  # RGB channels only
    buf.close()
    
    # Ensure dimensions are even (required for H.264 encoding)
    h, w = frame.shape[:2]
    if h % 2 != 0:
        frame = frame[:-1, :, :]  # Remove last row
    if w % 2 != 0:
        frame = frame[:, :-1, :]  # Remove last column
    
    # Cleanup
    plt.close(fig)
    
    return frame_idx, frame


class ParallelAnimationGenerator:
    """
    Parallel animation generator
    Supports multi-threaded parallel rendering for significantly faster generation
    """
    
    def __init__(self, config: DragonConfig, fps: int = 10, dpi: int = 100, 
                 n_workers: Optional[int] = None):
        """
        Initialize parallel animation generator
        
        Args:
            config: Dragon configuration
            fps: Frame rate
            dpi: Resolution
            n_workers: Number of worker processes (None = CPU count - 1)
        """
        self.config = config
        self.fps = fps
        self.dpi = dpi
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
        print(f"Parallel animator initialized: {self.n_workers} workers")
    
    def _compute_bounds(self, data: TimeSeriesData) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute coordinate bounds"""
        all_positions = np.vstack([s.positions for s in data.states])
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        
        # Add margin
        margin = max(x_max - x_min, y_max - y_min) * 0.1
        xlim = (x_min - margin, x_max + margin)
        ylim = (y_min - margin, y_max + margin)
        
        return xlim, ylim
    
    def _prepare_frame_params(self, data: TimeSeriesData, 
                              frame_indices: List[int],
                              xlim: Tuple[float, float],
                              ylim: Tuple[float, float],
                              highlight_indices: Optional[List[int]] = None,
                              figsize: Tuple[float, float] = (10, 10),
                              path_handler=None,
                              path_range: Optional[Tuple[float, float]] = None) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Prepare rendering parameters for all frames
        
        Returns:
            List[(frame_index, render_params)]
        """
        frame_params = []
        
        for i, idx in enumerate(frame_indices):
            state = data.states[idx]
            
            params = {
                'state': state,
                'config': self.config,
                'xlim': xlim,
                'ylim': ylim,
                'highlight_indices': highlight_indices,
                'dpi': self.dpi,
                'figsize': figsize,
                'path_handler': path_handler,
                'path_range': path_range
            }
            
            frame_params.append((i, params))
        
        return frame_params
    
    def generate_mp4(self, data: TimeSeriesData,
                     output_path: Path,
                     highlight_indices: Optional[List[int]] = None,
                     frame_interval: int = 1,
                     figsize: Tuple[float, float] = (10, 10),
                     codec: str = 'libx264',
                     quality: int = 8,
                     path_handler=None,
                     path_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Generate MP4 animation in parallel
        
        Args:
            data: Time series data
            output_path: Output file path
            highlight_indices: Indices of handles to highlight
            frame_interval: Frame interval (sample every N frames)
            figsize: Figure size
            codec: Video codec
            quality: Quality (1-10, 10 is best)
            path_handler: 路径处理器（可选）
            path_range: 路径参数范围（可选）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Select frames
        frame_indices = list(range(0, len(data), frame_interval))
        n_frames = len(frame_indices)
        
        print(f"\n{'='*60}")
        print(f"Generating MP4 animation in parallel")
        print(f"{'='*60}")
        print(f"Total frames: {n_frames}")
        print(f"Workers: {self.n_workers}")
        print(f"FPS: {self.fps}")
        print(f"DPI: {self.dpi}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")
        
        # Compute bounds
        xlim, ylim = self._compute_bounds(data)
        
        # Prepare parameters
        print("Preparing render parameters...")
        frame_params = self._prepare_frame_params(
            data, frame_indices, xlim, ylim, 
            highlight_indices, figsize, path_handler, path_range
        )
        
        # Parallel rendering
        print(f"\nRendering {n_frames} frames in parallel...")
        frames: List[Optional[np.ndarray]] = [None] * n_frames
        
        with Pool(processes=self.n_workers) as pool:
            # Use imap_unordered for out-of-order completion with index tracking
            results = list(tqdm(
                pool.imap_unordered(render_single_frame, frame_params),
                total=n_frames,
                desc="Rendering",
                unit="frame"
            ))
        
        # Sort by index
        print("\nOrdering frames...")
        for frame_idx, frame in results:
            frames[frame_idx] = frame
        
        # Check for missing frames
        if any(f is None for f in frames):
            raise RuntimeError("Some frames failed to render")
        
        # Cast to non-optional list
        valid_frames: List[np.ndarray] = [f for f in frames if f is not None]
        
        # Compose video
        print(f"\nComposing MP4 video...")
        writer = imageio.get_writer(
            str(output_path),
            fps=self.fps,
            codec=codec,
            quality=quality,
            macro_block_size=1
        )
        
        for frame in tqdm(valid_frames, desc="Writing", unit="frame"):
            writer.append_data(frame)
        
        writer.close()
        
        # Cleanup
        del frames
        del results
        gc.collect()
        
        print(f"\n✓ MP4 saved to: {output_path}")
        print(f"{'='*60}\n")
    
    def generate_gif(self, data: TimeSeriesData,
                     output_path: Path,
                     highlight_indices: Optional[List[int]] = None,
                     frame_interval: int = 1,
                     figsize: Tuple[float, float] = (8, 8),
                     loop: int = 0,
                     path_handler=None,
                     path_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Generate GIF animation in parallel
        
        Args:
            data: Time series data
            output_path: Output file path
            highlight_indices: Indices to highlight
            frame_interval: Frame interval
            figsize: Figure size
            loop: Loop count (0 = infinite)
            path_handler: 路径处理器（可选）
            path_range: 路径参数范围（可选）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Select frames
        frame_indices = list(range(0, len(data), frame_interval))
        n_frames = len(frame_indices)
        
        print(f"\n{'='*60}")
        print(f"Generating GIF animation in parallel")
        print(f"{'='*60}")
        print(f"Total frames: {n_frames}")
        print(f"Workers: {self.n_workers}")
        print(f"FPS: {self.fps}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")
        
        # Compute bounds
        xlim, ylim = self._compute_bounds(data)
        
        # Prepare parameters
        print("Preparing render parameters...")
        frame_params = self._prepare_frame_params(
            data, frame_indices, xlim, ylim,
            highlight_indices, figsize, path_handler, path_range
        )
        
        # Parallel rendering
        print(f"\nRendering {n_frames} frames in parallel...")
        frames: List[Optional[np.ndarray]] = [None] * n_frames
        
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(render_single_frame, frame_params),
                total=n_frames,
                desc="Rendering",
                unit="frame"
            ))
        
        # Sort
        print("\nOrdering frames...")
        for frame_idx, frame in results:
            frames[frame_idx] = frame
        
        if any(f is None for f in frames):
            raise RuntimeError("Some frames failed to render")
        
        # Cast to non-optional list for writing
        valid_frames = [f for f in frames if f is not None]
        
        # Compose GIF
        print(f"\nComposing GIF...")
        imageio.mimsave(
            str(output_path),
            valid_frames,  # type: ignore[arg-type]
            fps=self.fps,
            loop=loop
        )
        
        # Cleanup
        del valid_frames
        del frames
        del results
        gc.collect()
        
        print(f"\n✓ GIF saved to: {output_path}")
        print(f"{'='*60}\n")
    
    def generate_frames(self, data: TimeSeriesData,
                       output_dir: Path,
                       highlight_indices: Optional[List[int]] = None,
                       frame_interval: int = 1,
                       figsize: Tuple[float, float] = (10, 10),
                       prefix: str = "frame") -> List[Path]:
        """
        Generate individual frame images (PNG) in parallel
        
        Args:
            data: Time series data
            output_dir: Output directory
            highlight_indices: Indices to highlight
            frame_interval: Frame interval
            figsize: Figure size
            prefix: Filename prefix
            
        Returns:
            List of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select frames
        frame_indices = list(range(0, len(data), frame_interval))
        n_frames = len(frame_indices)
        
        print(f"\n{'='*60}")
        print(f"Generating frame images in parallel")
        print(f"{'='*60}")
        print(f"Total frames: {n_frames}")
        print(f"Workers: {self.n_workers}")
        print(f"Output dir: {output_dir}")
        print(f"{'='*60}\n")
        
        # Compute bounds
        xlim, ylim = self._compute_bounds(data)
        
        # Prepare parameters
        frame_params = self._prepare_frame_params(
            data, frame_indices, xlim, ylim,
            highlight_indices, figsize
        )
        
        # Parallel rendering
        print(f"\nRendering {n_frames} frames in parallel...")
        frames: List[Optional[np.ndarray]] = [None] * n_frames
        
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(render_single_frame, frame_params),
                total=n_frames,
                desc="Rendering",
                unit="frame"
            ))
        
        # Sort
        for frame_idx, frame in results:
            frames[frame_idx] = frame
        
        # Cast to non-optional list
        valid_frames: List[np.ndarray] = [f for f in frames if f is not None]
        
        # Save as PNG
        print(f"\nSaving images...")
        file_paths = []
        for i, frame in enumerate(tqdm(valid_frames, desc="Saving", unit="file")):
            filepath = output_dir / f"{prefix}_{i:06d}.png"
            imageio.imwrite(str(filepath), frame)
            file_paths.append(filepath)
        
        # Cleanup
        del valid_frames
        del frames
        del results
        gc.collect()
        
        print(f"\n✓ {len(file_paths)} frames saved to: {output_dir}")
        print(f"{'='*60}\n")
        
        return file_paths


class AnimationGenerator:
    """
    Animation generator (compatibility wrapper using parallel generator internally)
    """
    
    def __init__(self, config: DragonConfig, fps: int = 10, dpi: int = 100):
        """Initialize animation generator"""
        self.parallel_gen = ParallelAnimationGenerator(config, fps, dpi)
    
    def create_frames_mp4(self, data: TimeSeriesData,
                         output_path: Path,
                         highlight_indices: Optional[List[int]] = None,
                         frame_interval: int = 1,
                         **kwargs):
        """Generate MP4 (compatibility interface)"""
        self.parallel_gen.generate_mp4(
            data, output_path, highlight_indices, frame_interval
        )
    
    def create_frames_gif(self, data: TimeSeriesData,
                         output_path: Path,
                         highlight_indices: Optional[List[int]] = None,
                         frame_interval: int = 1,
                         **kwargs):
        """Generate GIF (compatibility interface)"""
        self.parallel_gen.generate_gif(
            data, output_path, highlight_indices, frame_interval
        )
