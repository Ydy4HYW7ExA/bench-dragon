"""
板凳龙渲染器
负责绘制板凳龙的图形
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Optional, Dict, Any

from ..core.models import DragonConfig, SimulationState, DragonSegment


class DragonRenderer:
    """
    板凳龙渲染器
    正确绘制板凳的形状和方向
    """
    
    def __init__(self, config: DragonConfig, colors: Optional[Dict[str, str]] = None):
        """
        初始化渲染器
        
        config: 龙的配置
        colors: 颜色方案
        """
        self.config = config
        self.colors = colors or {
            'head': '#FF0000',      # 龙头 - 红色
            'body': '#FFD700',      # 龙身 - 金色
            'tail': '#FFD700',      # 龙尾 - 金色
            'handle': '#000000',    # 把手 - 黑色
            'highlight': '#00FF00', # 高亮 - 绿色
            'path': '#0000FF',      # 路径 - 蓝色
            'boundary': '#808080'   # 边界 - 灰色
        }
    
    def render_dragon(self, state: SimulationState, ax: Axes, 
                     highlight_indices: Optional[List[int]] = None,
                     show_handles: bool = True):
        """
        渲染整个龙队
        
        state: 仿真状态
        ax: matplotlib axes
        highlight_indices: 需要高亮显示的把手索引
        show_handles: 是否显示把手
        """
        highlight_indices = highlight_indices or []
        segments = state.get_all_segments(self.config)
        
        for i, segment in enumerate(segments):
            self.render_segment(segment, ax, i in highlight_indices, show_handles)
    
    def render_segment(self, segment: DragonSegment, ax: Axes, 
                      highlight: bool = False, show_handles: bool = True):
        """
        渲染单个板凳节
        
        segment: 板凳节
        ax: matplotlib axes
        highlight: 是否高亮
        show_handles: 是否显示把手
        """
        # 选择颜色
        if segment.is_head:
            face_color = self.colors['head']
        else:
            face_color = self.colors['body']
        
        # 获取板凳四角
        corners = segment.get_corners(self.config)
        
        # 绘制板凳矩形
        polygon = Polygon(
            [(c.x, c.y) for c in corners],
            facecolor=face_color,
            edgecolor='black',
            linewidth=1.0,
            alpha=0.8,
            zorder=2
        )
        ax.add_patch(polygon)
        
        # 绘制把手
        if show_handles:
            if highlight:
                handle_color = self.colors['highlight']
                marker_size = 8
                marker = 'o'
                edge_width = 2
            else:
                handle_color = self.colors['handle']
                marker_size = 4
                marker = 'o'
                edge_width = 1
            
            # 前把手
            ax.plot(segment.front_handle.x, segment.front_handle.y,
                   marker=marker, color=handle_color, 
                   markersize=marker_size, markeredgewidth=edge_width,
                   markeredgecolor='black', zorder=10)
            
            # 后把手(只有非最后一个节点才绘制,避免重复)
            if not segment.is_tail:
                ax.plot(segment.back_handle.x, segment.back_handle.y,
                       marker=marker, color=handle_color,
                       markersize=marker_size, markeredgewidth=edge_width,
                       markeredgecolor='black', zorder=10)
    
    def render_path(self, ax: Axes, path_handler, param_range: tuple, 
                   num_points: int = 1000, label: str = ''):
        """
        渲染路径曲线
        
        ax: matplotlib axes
        path_handler: 路径处理器
        param_range: 参数范围 (start, end)
        num_points: 离散点数
        label: 标签
        """
        params = np.linspace(param_range[0], param_range[1], num_points)
        points = [path_handler.compute_position(p) for p in params]
        
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        
        ax.plot(xs, ys, '--', color=self.colors['path'], 
               linewidth=1.0, alpha=0.5, label=label, zorder=1)
    
    def render_boundary(self, ax: Axes, radius: float, center=(0, 0)):
        """
        渲染边界圆
        
        ax: matplotlib axes
        radius: 半径
        center: 圆心
        """
        circle = Circle(center, radius, fill=False, 
                       edgecolor=self.colors['boundary'],
                       linewidth=2, linestyle='--', zorder=1)
        ax.add_patch(circle)
    
    def setup_axes(self, ax: Axes, xlim: tuple, ylim: tuple, title: str = ''):
        """
        设置坐标轴
        
        ax: matplotlib axes
        xlim: x轴范围
        ylim: y轴范围
        title: 标题
        """
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        if title:
            ax.set_title(title)
