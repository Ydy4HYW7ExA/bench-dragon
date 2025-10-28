"""
几何计算工具和碰撞检测
"""

import numpy as np
from typing import Optional, List
from ..core.models import Point2D, DragonSegment, SimulationState, DragonConfig


class GeometryUtils:
    """几何计算工具类"""
    
    @staticmethod
    def point_to_line_distance(point: Point2D, line_start: Point2D, line_end: Point2D) -> float:
        """计算点到线段的距离"""
        # 向量
        line_vec = np.array([line_end.x - line_start.x, line_end.y - line_start.y])
        point_vec = np.array([point.x - line_start.x, point.y - line_start.y])
        
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-10:
            return float(np.linalg.norm(point_vec))
        
        line_unitvec = line_vec / line_len
        
        # 投影长度
        proj_len = np.dot(point_vec, line_unitvec)
        
        if proj_len < 0:
            # 最近点是line_start
            return float(np.linalg.norm(point_vec))
        elif proj_len > line_len:
            # 最近点是line_end
            return point.distance_to(line_end)
        else:
            # 最近点在线段上
            proj_point = line_start.to_array() + proj_len * line_unitvec
            return float(np.linalg.norm(point.to_array() - proj_point))
    
    @staticmethod
    def line_segment_intersection(seg1_start: Point2D, seg1_end: Point2D,
                                   seg2_start: Point2D, seg2_end: Point2D) -> Optional[Point2D]:
        """
        判断两条线段是否相交
        如果相交返回交点,否则返回None
        """
        x1, y1 = seg1_start.x, seg1_start.y
        x2, y2 = seg1_end.x, seg1_end.y
        x3, y3 = seg2_start.x, seg2_start.y
        x4, y4 = seg2_end.x, seg2_end.y
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            # 平行或共线
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            # 相交
            return Point2D(x=x1 + t * (x2 - x1), y=y1 + t * (y2 - y1))
        
        return None
    
    @staticmethod
    def rectangle_intersection(rect1_corners: List[Point2D], rect2_corners: List[Point2D]) -> bool:
        """
        判断两个矩形是否相交
        使用完整的分离轴定理(SAT) + 边相交检测 + 包含检测
        """
        # 方法1: 检查所有边是否相交
        for i in range(4):
            for j in range(4):
                if GeometryUtils.line_segment_intersection(
                    rect1_corners[i], rect1_corners[(i+1) % 4],
                    rect2_corners[j], rect2_corners[(j+1) % 4]
                ):
                    return True
        
        # 方法2: 检查所有顶点是否在另一个矩形内
        # 这捕获一个矩形完全包含另一个的情况
        for corner in rect1_corners:
            if GeometryUtils.point_in_polygon(corner, rect2_corners):
                return True
        
        for corner in rect2_corners:
            if GeometryUtils.point_in_polygon(corner, rect1_corners):
                return True
        
        return False
    
    @staticmethod
    def point_in_polygon(point: Point2D, polygon: List[Point2D]) -> bool:
        """判断点是否在多边形内 (射线法)"""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if point.y > min(p1y, p2y):
                if point.y <= max(p1y, p2y):
                    if point.x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point.y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point.x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


class CollisionDetector:
    """
    碰撞检测器
    检测板凳龙各节之间的碰撞
    """
    
    def __init__(self, config: DragonConfig, discretization_n: int = 10, discretization_m: int = 20):
        """
        初始化碰撞检测器
        config: 龙的配置
        discretization_n: 板凳宽度方向的离散点数
        discretization_m: 板凳长度方向的离散点数
        """
        self.config = config
        self.n = discretization_n
        self.m = discretization_m
    
    def check_segments_collision(self, seg1: DragonSegment, seg2: DragonSegment) -> bool:
        """
        检查两个板凳节是否发生碰撞
        使用离散化方法
        
        参考MATLAB代码中的find_if_intersect函数
        """
        # 获取两个板凳的中心线
        X1_1 = seg1.front_handle.to_array()
        X1_2 = seg1.back_handle.to_array()
        X2_1 = seg2.front_handle.to_array()
        X2_2 = seg2.back_handle.to_array()
        
        # 获取板凳长度
        L1 = seg1.get_length(self.config)
        L2 = seg2.get_length(self.config)
        
        # 计算斜率和垂线斜率
        dx1 = X1_1[0] - X1_2[0]
        dy1 = X1_1[1] - X1_2[1]
        dx2 = X2_1[0] - X2_2[0]
        dy2 = X2_1[1] - X2_2[1]
        
        if abs(dx1) < 1e-10 or abs(dx2) < 1e-10:
            # 处理垂直情况
            return self._check_collision_simple(seg1, seg2)
        
        k1 = dy1 / dx1
        k1_perp = -1 / k1
        k2 = dy2 / dx2
        k2_perp = -1 / k2
        
        # 中心点
        X1_center = (X1_1 + X1_2) / 2
        X2_center = (X2_1 + X2_2) / 2
        
        # 求垂线交点 (新坐标原点)
        A = np.array([[k1_perp, -1], [k2_perp, -1]])
        b = np.array([k1_perp * X1_center[0] - X1_center[1],
                      k2_perp * X2_center[0] - X2_center[1]])
        
        try:
            P = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 矩阵奇异, 使用简单方法
            return self._check_collision_simple(seg1, seg2)
        
        # 计算向量和角度
        vec1 = X1_center - P
        vec2 = X2_center - P
        
        theta1 = np.arctan2(vec1[1], vec1[0])
        theta2 = np.arctan2(vec2[1], vec2[0])
        delta_theta = theta1 - theta2
        
        d1 = np.linalg.norm(vec1)
        d2 = np.linalg.norm(vec2)
        
        # 创建网格
        x_range = np.linspace(d1 - self.config.bench_width/2, d1 + self.config.bench_width/2, self.n)
        y_range = np.linspace(-L1/2, L1/2, self.m)
        X, Y = np.meshgrid(x_range, y_range)
        
        # 旋转矩阵
        cos_delta = np.cos(delta_theta)
        sin_delta = np.sin(delta_theta)
        T = np.array([[cos_delta, -sin_delta],
                      [sin_delta, cos_delta]])
        
        # 转换所有点
        XY = np.stack([X.ravel(), Y.ravel()], axis=0)
        XY_new = T @ XY
        
        # 检查是否有点在第二个板凳的范围内
        in_range_x = np.abs(XY_new[0, :] - d2) < self.config.bench_width / 2
        in_range_y = np.abs(XY_new[1, :]) < L2 / 2
        
        collision = bool(np.any(in_range_x & in_range_y))
        
        return collision
    
    def _check_collision_simple(self, seg1: DragonSegment, seg2: DragonSegment) -> bool:
        """简单的碰撞检测: 检查矩形是否相交"""
        corners1 = seg1.get_corners(self.config)
        corners2 = seg2.get_corners(self.config)
        return GeometryUtils.rectangle_intersection(corners1, corners2)
    
    def check_state_collision(self, state: SimulationState) -> bool:
        """
        检查整个龙队是否发生碰撞
        只需要检查不相邻的板凳之间
        """
        segments = state.get_all_segments(self.config)
        n = len(segments)
        
        # 只检查相隔一定距离的板凳
        # 根据螺线特性, 主要检查头部与螺线内圈的碰撞
        for i in range(min(n // 2, 50)):  # 只检查前半部分
            seg1 = segments[i]
            
            # 计算可能发生碰撞的范围
            # 基于角度差
            angle1 = seg1.angle
            
            for j in range(i + 2, n):  # 跳过相邻的
                seg2 = segments[j]
                angle2 = seg2.angle
                
                # 检查是否在可能碰撞的角度范围内
                # 对于螺线, 主要是相差约2π的位置
                angle_diff = abs(angle1 - angle2)
                if 1.5 * np.pi < angle_diff < 2.5 * np.pi:
                    if self.check_segments_collision(seg1, seg2):
                        return True
        
        return False
    
    def find_first_collision_index(self, state: SimulationState) -> Optional[int]:
        """
        找到第一个发生碰撞的板凳索引
        返回None表示没有碰撞
        """
        segments = state.get_all_segments(self.config)
        n = len(segments)
        
        for i in range(min(n // 2, 50)):
            seg1 = segments[i]
            angle1 = seg1.angle
            
            for j in range(i + 2, n):
                seg2 = segments[j]
                angle2 = seg2.angle
                
                angle_diff = abs(angle1 - angle2)
                if 1.5 * np.pi < angle_diff < 2.5 * np.pi:
                    if self.check_segments_collision(seg1, seg2):
                        return i
        
        return None
