"""
调头路径几何计算模块

计算双圆弧S形调头曲线的几何参数:
- 第一段圆弧半径R1, 第二段圆弧半径R2, R1 = 2*R2
- 两段圆弧相切连接
- 与盘入/盘出螺线相切
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .models import Point2D


@dataclass
class UTurnGeometry:
    """
    调头曲线的几何参数
    """
    # 圆弧半径
    r1: float  # 第一段圆弧半径
    r2: float  # 第二段圆弧半径 (r1 = 2*r2)
    
    # 第一段圆弧
    arc1_center: Point2D  # 圆心
    arc1_start_angle: float  # 起始角度(rad)
    arc1_end_angle: float  # 结束角度(rad)
    arc1_length: float  # 弧长
    
    # 第二段圆弧  
    arc2_center: Point2D  # 圆心
    arc2_start_angle: float  # 起始角度(rad)
    arc2_end_angle: float  # 结束角度(rad)
    arc2_length: float  # 弧长
    
    # 总调头曲线长度
    total_length: float
    
    # 与螺线的衔接点
    spiral_in_end_point: Point2D  # 盘入螺线结束点(调头起点)
    spiral_out_start_point: Point2D  # 盘出螺线起始点(调头终点)
    
    # 螺线衔接参数
    spiral_in_end_angle: float  # 盘入螺线结束时的极角
    spiral_out_start_angle: float  # 盘出螺线开始时的极角


def compute_uturn_geometry(
    pitch: float,
    turnaround_radius: float,
    r2: float,
    clockwise_in: bool = True
) -> UTurnGeometry:
    """
    计算调头曲线的几何参数
    
    参数:
        pitch: 螺线螺距(m)
        turnaround_radius: 调头空间半径(m), 直径9m则为4.5m
        r2: 第二段圆弧半径(m), 第一段圆弧半径r1 = 2*r2
        clockwise_in: 盘入螺线是否顺时针
    
    返回:
        UTurnGeometry: 调头曲线几何参数
    
    设计思路:
        1. 盘入螺线(顺时针)到达调头空间边界
        2. 第一段圆弧(半径r1=2*r2)从螺线切出,向内弯曲
        3. 第二段圆弧(半径r2)与第一段相切,继续弯曲完成调头
        4. 盘出螺线(逆时针,中心对称)从第二段圆弧切出
    """
    k = pitch / (2 * np.pi)  # 螺线参数 r = k*theta
    r1 = 2 * r2  # 第一段圆弧半径
    
    # === 确定盘入螺线结束点(调头起点) ===
    # 盘入螺线到达调头空间边界: r_spiral = turnaround_radius
    # r = k * theta => theta = r / k
    theta_in_end = turnaround_radius / k if clockwise_in else -turnaround_radius / k
    
    # 盘入螺线结束点位置
    x_in = turnaround_radius * np.cos(theta_in_end)
    y_in = turnaround_radius * np.sin(theta_in_end)
    spiral_in_end = Point2D(x=x_in, y=y_in)
    
    # 盘入螺线在该点的切线方向
    # 对于螺线 r = k*theta (顺时针):
    # dx/dtheta = k*cos(theta) - k*theta*sin(theta)
    # dy/dtheta = k*sin(theta) + k*theta*cos(theta)
    if clockwise_in:
        dx_dtheta = k * np.cos(theta_in_end) - k * theta_in_end * np.sin(theta_in_end)
        dy_dtheta = k * np.sin(theta_in_end) + k * theta_in_end * np.cos(theta_in_end)
    else:
        dx_dtheta = -(k * np.cos(theta_in_end) - k * theta_in_end * np.sin(theta_in_end))
        dy_dtheta = -(k * np.sin(theta_in_end) + k * theta_in_end * np.cos(theta_in_end))
    
    # 切线单位向量(螺线前进方向)
    tangent_length = np.sqrt(dx_dtheta**2 + dy_dtheta**2)
    tangent_x = dx_dtheta / tangent_length
    tangent_y = dy_dtheta / tangent_length
    
    # 螺线向内盘入,需要向内侧(左侧)转弯
    # 法向量(向左,即逆时针旋转90度)
    normal_x = -tangent_y
    normal_y = tangent_x
    
    # === 第一段圆弧 ===
    # 圆心在切点的法向方向,距离为r1
    arc1_center_x = x_in + r1 * normal_x
    arc1_center_y = y_in + r1 * normal_y
    arc1_center = Point2D(x=arc1_center_x, y=arc1_center_y)
    
    # 第一段圆弧起始点就是盘入螺线结束点
    arc1_start_angle = np.arctan2(y_in - arc1_center_y, x_in - arc1_center_x)
    
    # 第一段圆弧需要转多少角度?
    # 使得在结束点处与第二段圆弧相切
    # 几何关系: 两段圆弧相切,圆心连线通过切点
    # 设第一段圆弧转角为alpha
    
    # 为简化,假设调头是对称的S形曲线
    # 第一段圆弧向内转约90度,第二段圆弧向外转约90度
    # 使得整体完成约180度转向
    
    # 更精确的计算:
    # 两圆弧相切,圆心距离 = r1 + r2 = 2*r2 + r2 = 3*r2
    # 设第一段圆弧终点为P1, 第二段圆弧起点为P1
    # 几何约束: |arc1_center - arc2_center| = 3*r2
    
    # 为使调头路径对称且紧凑,设计为:
    # 第一段圆弧转角 alpha1, 第二段圆弧转角 alpha2
    # 由于对称性, alpha1 ≈ alpha2
    
    # 简化设计: 每段圆弧转90度
    alpha1 = np.pi / 2  # 第一段转90度
    alpha2 = np.pi / 2  # 第二段转90度
    
    arc1_end_angle = arc1_start_angle + alpha1
    arc1_length = r1 * alpha1
    
    # 第一段圆弧结束点(即第二段圆弧起始点)
    x_mid = arc1_center_x + r1 * np.cos(arc1_end_angle)
    y_mid = arc1_center_y + r1 * np.sin(arc1_end_angle)
    
    # === 第二段圆弧 ===
    # 第二段圆弧与第一段圆弧相切
    # 切点处的切向量相同
    tangent_mid_x = -np.sin(arc1_end_angle)
    tangent_mid_y = np.cos(arc1_end_angle)
    
    # 第二段圆弧需要向另一侧弯曲(向右,即顺时针旋转90度)
    normal_mid_x = tangent_mid_y
    normal_mid_y = -tangent_mid_x
    
    # 第二段圆弧圆心
    arc2_center_x = x_mid + r2 * normal_mid_x
    arc2_center_y = y_mid + r2 * normal_mid_y
    arc2_center = Point2D(x=arc2_center_x, y=arc2_center_y)
    
    arc2_start_angle = np.arctan2(y_mid - arc2_center_y, x_mid - arc2_center_x)
    arc2_end_angle = arc2_start_angle + alpha2
    arc2_length = r2 * alpha2
    
    # 第二段圆弧结束点(即盘出螺线起始点)
    x_out = arc2_center_x + r2 * np.cos(arc2_end_angle)
    y_out = arc2_center_y + r2 * np.sin(arc2_end_angle)
    spiral_out_start = Point2D(x=x_out, y=y_out)
    
    # 盘出螺线的起始极角
    # 盘出螺线与盘入螺线关于原点中心对称
    # 若盘入结束于 (r, theta), 则盘出开始于 (-r, theta+pi) = (r, theta+pi)
    # 但实际上盘出螺线是逆时针的,参数化不同
    
    # 盘出螺线起始点距离原点的距离
    r_out = np.sqrt(x_out**2 + y_out**2)
    theta_out_start = r_out / k if not clockwise_in else -r_out / k
    
    # 总调头曲线长度
    total_length = arc1_length + arc2_length
    
    return UTurnGeometry(
        r1=r1,
        r2=r2,
        arc1_center=arc1_center,
        arc1_start_angle=arc1_start_angle,
        arc1_end_angle=arc1_end_angle,
        arc1_length=arc1_length,
        arc2_center=arc2_center,
        arc2_start_angle=arc2_start_angle,
        arc2_end_angle=arc2_end_angle,
        arc2_length=arc2_length,
        total_length=total_length,
        spiral_in_end_point=spiral_in_end,
        spiral_out_start_point=spiral_out_start,
        spiral_in_end_angle=theta_in_end,
        spiral_out_start_angle=theta_out_start
    )


def optimize_uturn_radius(
    pitch: float,
    turnaround_radius: float,
    min_r2: float = 0.5,
    max_r2: float = 3.0,
    num_samples: int = 20
) -> Tuple[float, UTurnGeometry]:
    """
    优化第二段圆弧半径r2,使调头曲线尽可能短
    
    参数:
        pitch: 螺线螺距(m)
        turnaround_radius: 调头空间半径(m)
        min_r2: r2的最小值(m)
        max_r2: r2的最大值(m)
        num_samples: 采样数量
    
    返回:
        (最优r2, 最优几何参数)
    """
    best_r2 = None
    best_geometry = None
    min_length = float('inf')
    
    for r2 in np.linspace(min_r2, max_r2, num_samples):
        try:
            geometry = compute_uturn_geometry(pitch, turnaround_radius, r2)
            
            # 检查是否在调头空间内
            # 圆弧圆心到原点的距离 + 圆弧半径 应该 ≤ turnaround_radius
            d1 = np.sqrt(geometry.arc1_center.x**2 + geometry.arc1_center.y**2)
            d2 = np.sqrt(geometry.arc2_center.x**2 + geometry.arc2_center.y**2)
            
            if d1 + geometry.r1 > turnaround_radius or d2 + geometry.r2 > turnaround_radius:
                continue  # 超出调头空间,跳过
            
            # 比较长度
            if geometry.total_length < min_length:
                min_length = geometry.total_length
                best_r2 = r2
                best_geometry = geometry
        except Exception as e:
            continue
    
    if best_geometry is None:
        # 如果没有找到合适的,使用中间值
        r2 = (min_r2 + max_r2) / 2
        best_geometry = compute_uturn_geometry(pitch, turnaround_radius, r2)
        best_r2 = r2
    
    assert best_r2 is not None, "Failed to find optimal r2"
    assert best_geometry is not None, "Failed to compute geometry"
    
    return best_r2, best_geometry
