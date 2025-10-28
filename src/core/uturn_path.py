"""
调头路径处理器

实现完整的调头路径:
1. 螺线盘入段
2. 双圆弧S形调头段
3. 螺线盘出段
"""

import numpy as np
from typing import Optional

from .models import Point2D
from .paths import PathHandler, SpiralInHandler, SpiralOutHandler
from .uturn_geometry import UTurnGeometry


class DoubleArcUTurnHandler(PathHandler):
    """
    双圆弧S形调头路径处理器
    
    包含三段:
    1. 螺线盘入段: 从某个位置沿螺线盘入到调头空间边界
    2. 双圆弧调头段: 两段圆弧(R1=2*R2)相切连接
    3. 螺线盘出段: 从调头空间边界沿螺线盘出
    """
    
    def __init__(self, geometry: UTurnGeometry, pitch: float, 
                 spiral_in_length: float, clockwise_in: bool = True):
        """
        初始化调头路径处理器
        
        参数:
            geometry: 调头曲线几何参数
            pitch: 螺线螺距(m)
            spiral_in_length: 盘入螺线段长度(m), 用于确定路径起点
            clockwise_in: 盘入螺线是否顺时针
        """
        self.geometry = geometry
        self.pitch = pitch
        self.k = pitch / (2 * np.pi)
        self.spiral_in_length = spiral_in_length
        self.clockwise_in = clockwise_in
        
        # 三段路径的弧长范围
        self.uturn_start_arc_length = spiral_in_length  # 调头段起始弧长
        self.uturn_end_arc_length = spiral_in_length + geometry.total_length  # 调头段结束弧长
        self.total_length = spiral_in_length + geometry.total_length  # 暂不包含盘出段(盘出段可以无限长)
        
        # 创建螺线处理器(用于计算螺线段)
        self.spiral_in_handler = SpiralInHandler(pitch=pitch)
        self.spiral_out_handler = SpiralOutHandler(pitch=pitch)
    
    def compute_position(self, arc_length: float) -> Point2D:
        """
        根据弧长计算位置
        
        参数:
            arc_length: 从路径起点开始的累积弧长(m)
        
        返回:
            Point2D: 位置坐标
        """
        if arc_length < self.uturn_start_arc_length:
            # === 螺线盘入段 ===
            return self.spiral_in_handler.compute_position(arc_length)
        
        elif arc_length < self.uturn_end_arc_length:
            # === 双圆弧调头段 ===
            s_uturn = arc_length - self.uturn_start_arc_length
            
            if s_uturn < self.geometry.arc1_length:
                # 第一段圆弧
                angle = self.geometry.arc1_start_angle + s_uturn / self.geometry.r1
                x = self.geometry.arc1_center.x + self.geometry.r1 * np.cos(angle)
                y = self.geometry.arc1_center.y + self.geometry.r1 * np.sin(angle)
                return Point2D(x=float(x), y=float(y))
            else:
                # 第二段圆弧
                s_arc2 = s_uturn - self.geometry.arc1_length
                angle = self.geometry.arc2_start_angle + s_arc2 / self.geometry.r2
                x = self.geometry.arc2_center.x + self.geometry.r2 * np.cos(angle)
                y = self.geometry.arc2_center.y + self.geometry.r2 * np.sin(angle)
                return Point2D(x=float(x), y=float(y))
        
        else:
            # === 螺线盘出段 ===
            # 盘出段从调头结束点开始
            # 需要将弧长映射到盘出螺线的参数
            s_out = arc_length - self.uturn_end_arc_length
            
            # 盘出螺线起始点的极角
            theta_out_start = self.geometry.spiral_out_start_angle
            
            # 盘出螺线: r = -k*theta (逆时针,向外)
            # 弧长 s = integral sqrt(r^2 + (dr/dtheta)^2) dtheta
            #         = integral |k| * sqrt(theta^2 + 1) dtheta
            
            # 从弧长反推theta(数值方法)
            from scipy.optimize import fsolve
            
            def arc_length_eq(theta):
                # 计算从theta_out_start到theta的弧长
                from scipy.integrate import quad
                integrand = lambda t: abs(self.k) * np.sqrt(t**2 + 1)
                length, _ = quad(integrand, theta_out_start, theta)
                return length - s_out
            
            # 初始猜测
            theta_guess = theta_out_start + s_out / (abs(self.k) * np.sqrt(theta_out_start**2 + 1))
            theta = fsolve(arc_length_eq, theta_guess)[0]
            
            # 逆时针盘出: r = -k*theta
            r = -self.k * theta
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return Point2D(x=float(x), y=float(y))
    
    def compute_velocity_magnitude(self, arc_length: float, speed: float) -> float:
        """
        计算速度大小(恒定)
        
        参数:
            arc_length: 弧长参数(m)
            speed: 龙头速度(m/s)
        
        返回:
            float: 速度大小(m/s)
        """
        return speed
    
    def compute_param_derivative(self, arc_length: float, speed: float) -> float:
        """
        计算参数对时间的导数 ds/dt
        
        参数:
            arc_length: 当前弧长参数(m)
            speed: 速度(m/s)
        
        返回:
            float: ds/dt = speed
        """
        return speed
    
    def solve_next_param(self, current_pos: Point2D, current_arc_length: float, 
                        distance: float) -> float:
        """
        求解下一时刻的弧长参数
        
        参数:
            current_pos: 当前位置
            current_arc_length: 当前弧长参数(m)
            distance: 移动距离(m)
        
        返回:
            float: 下一时刻的弧长参数
        """
        return current_arc_length + distance
    
    def get_tangent_direction(self, arc_length: float) -> float:
        """
        获取路径切线方向(角度)
        
        参数:
            arc_length: 弧长参数(m)
        
        返回:
            float: 切线方向角(rad)
        """
        # 通过微小位移计算切线方向
        eps = 0.001
        p1 = self.compute_position(arc_length)
        p2 = self.compute_position(arc_length + eps)
        
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        
        return np.arctan2(dy, dx)
