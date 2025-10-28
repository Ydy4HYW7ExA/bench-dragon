"""
图表绘制器
生成各种分析图表
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional

from ..core.models import TimeSeriesData, DragonConfig


class Plotter:
    """
    图表绘制器
    生成位置、速度等分析图表
    """
    
    def __init__(self, config: DragonConfig, output_dir: Path, dpi: int = 300):
        """
        初始化绘制器
        
        config: 龙的配置
        output_dir: 输出目录
        dpi: 图像分辨率
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
    def plot_trajectory(self, data: TimeSeriesData, 
                       indices: Optional[List[int]] = None,
                       filename: str = 'trajectory.png',
                       path_handler=None,
                       path_range: Optional[tuple] = None):
        """
        绘制轨迹图
        
        data: 时间序列数据
        indices: 要绘制的把手索引
        filename: 输出文件名
        path_handler: 路径处理器（可选）
        path_range: 路径参数范围（可选）
        """
        if indices is None:
            indices = [0, 1, 51, 101, 151, 201, self.config.num_segments]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 先绘制路径（背景）
        if path_handler is not None and path_range is not None:
            from ..visualization.renderer import DragonRenderer
            renderer = DragonRenderer(self.config)
            renderer.render_path(ax, path_handler, path_range, num_points=2000, label='路径')
        
        labels = self._get_labels(indices)
        
        for idx, label in zip(indices, labels):
            xs = [s.positions[idx, 0] for s in data.states]
            ys = [s.positions[idx, 1] for s in data.states]
            ax.plot(xs, ys, '-', label=label, linewidth=1.5)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('龙队运动轨迹')
        ax.legend()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"轨迹图已保存到: {filepath}")
    
    def plot_position_vs_time(self, data: TimeSeriesData,
                             indices: Optional[List[int]] = None,
                             filename: str = 'position_vs_time.png'):
        """
        绘制位置随时间变化
        
        data: 时间序列数据
        indices: 要绘制的把手索引
        filename: 输出文件名
        """
        if indices is None:
            indices = [0, 1, 51, 101, 151, 201, self.config.num_segments]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        times = data.times
        labels = self._get_labels(indices)
        
        # X坐标
        for idx, label in zip(indices, labels):
            xs = [s.positions[idx, 0] for s in data.states]
            ax1.plot(times, xs, '-', label=label, linewidth=1.5)
        
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('x坐标 (m)')
        ax1.set_title('X坐标随时间变化')
        ax1.legend()
        
        # Y坐标
        for idx, label in zip(indices, labels):
            ys = [s.positions[idx, 1] for s in data.states]
            ax2.plot(times, ys, '-', label=label, linewidth=1.5)
        
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('y坐标 (m)')
        ax2.set_title('Y坐标随时间变化')
        ax2.legend()
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"位置图已保存到: {filepath}")
    
    def plot_velocity_vs_time(self, data: TimeSeriesData,
                             indices: Optional[List[int]] = None,
                             filename: str = 'velocity_vs_time.png'):
        """
        绘制速度随时间变化
        
        data: 时间序列数据
        indices: 要绘制的把手索引
        filename: 输出文件名
        """
        if indices is None:
            indices = [0, 1, 51, 101, 151, 201, self.config.num_segments]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        times = data.times
        labels = self._get_labels(indices)
        
        for idx, label in zip(indices, labels):
            vs = [s.velocities[idx] for s in data.states]
            ax.plot(times, vs, '-', label=label, linewidth=1.5)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('速度 (m/s)')
        ax.set_title('速度随时间变化')
        ax.legend()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"速度图已保存到: {filepath}")
    
    def plot_snapshot(self, state, time: float, 
                     highlight_indices: Optional[List[int]] = None,
                     filename: str = 'snapshot.png'):
        """
        绘制单个时刻的快照
        
        state: 仿真状态
        time: 时间
        highlight_indices: 高亮索引
        filename: 输出文件名
        """
        from .renderer import DragonRenderer
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        renderer = DragonRenderer(self.config)
        renderer.render_dragon(state, ax, highlight_indices, show_handles=True)
        
        # 计算范围
        positions = state.positions
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        margin = max(x_max - x_min, y_max - y_min) * 0.1
        
        renderer.setup_axes(ax, 
                          (x_min - margin, x_max + margin),
                          (y_min - margin, y_max + margin),
                          title=f'时间: {time:.2f}s')
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"快照已保存到: {filepath}")
    
    def _get_labels(self, indices: List[int]) -> List[str]:
        """获取标签"""
        labels = []
        for idx in indices:
            if idx == 0:
                labels.append('龙头前把手')
            elif idx == self.config.num_segments:
                labels.append('龙尾后把手')
            else:
                labels.append(f'第{idx}节龙身')
        return labels
