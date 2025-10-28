"""
路径处理接口和实现
支持螺线、圆弧和复合路径
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import fsolve, brentq
from typing import Tuple, Optional
from ..core.models import Point2D


class PathHandler(ABC):
    """路径处理器抽象基类"""
    
    @abstractmethod
    def compute_position(self, param: float) -> Point2D:
        """
        根据路径参数计算位置
        param: 对于螺线是角度theta, 对于圆弧是弧长
        """
        pass
    
    @abstractmethod
    def compute_velocity_magnitude(self, param: float, speed: float) -> float:
        """
        计算指定参数处的速度大小
        param: 路径参数
        speed: 沿路径的速度
        """
        pass
    
    @abstractmethod
    def compute_param_derivative(self, param: float, speed: float) -> float:
        """
        计算参数对时间的导数 d(param)/dt
        param: 当前参数值
        speed: 沿路径的速度
        """
        pass
    
    @abstractmethod
    def solve_next_param(self, current_pos: Point2D, current_param: float, distance: float) -> float:
        """
        求解下一个位置的路径参数
        current_pos: 当前位置
        current_param: 当前参数
        distance: 需要的距离
        """
        pass


class SpiralPathHandler(PathHandler):
    """
    等距螺线路径处理器
    螺线方程: r = k * theta
    其中 k = pitch / (2π)
    """
    
    def __init__(self, pitch: float, direction: str = 'clockwise', solver_config: Optional[dict] = None):
        """
        初始化螺线路径
        pitch: 螺距 (m)
        direction: 方向, 'clockwise' 或 'counterclockwise'
        solver_config: 求解器配置 (可选)
        """
        self.pitch = pitch
        self.k = pitch / (2 * np.pi)  # 螺线系数
        self.direction_sign = -1 if direction == 'clockwise' else 1
        
        # 求解器配置（使用默认值或用户提供的配置）
        self.solver_config = solver_config or {}
        self.brentq_config = self.solver_config.get('brentq', {
            'xtol': 1e-12,
            'rtol': 1e-12,
            'maxiter': 1000
        })
        self.fsolve_config = self.solver_config.get('fsolve', {
            'xtol': 1e-12,
            'maxfev': 10000
        })
    
    def compute_position(self, theta: float) -> Point2D:
        """计算螺线上角度theta处的位置"""
        r = self.k * theta
        return Point2D(
            x=r * np.cos(theta),
            y=r * np.sin(theta)
        )
    
    def compute_velocity_magnitude(self, theta: float, speed: float) -> float:
        """
        计算螺线上的速度大小
        对于等距螺线, 速度大小等于沿螺线的速度
        """
        return speed
    
    def compute_param_derivative(self, theta: float, speed: float) -> float:
        """
        计算 dtheta/dt
        
        对于螺线 r = k*theta:
        弧长元素 ds = sqrt(r^2 + (dr/dtheta)^2) * dtheta
                     = sqrt((k*theta)^2 + k^2) * dtheta
                     = k * sqrt(1 + theta^2) * dtheta
        
        速度 v = ds/dt, 所以:
        dtheta/dt = v / (k * sqrt(1 + theta^2))
        
        顺时针运动时, theta减小, 所以要加负号
        """
        dtheta_dt = speed / (self.k * np.sqrt(1 + theta**2))
        return self.direction_sign * dtheta_dt
    
    def solve_next_param(self, current_pos: Point2D, current_theta: float, distance: float) -> float:
        """
        求解距离current_pos为distance的下一个点的角度
        
        关键理解:
        - 对于顺时针盘入(direction_sign=-1): 龙头在前(theta小), 后续把手在后(theta大)
          所以后续把手的theta应该 > current_theta (增大)
        - 对于逆时针盘出(direction_sign=1): 龙头在前(theta大), 后续把手在后(theta小)
          所以后续把手的theta应该 < current_theta (减小)
        
        总结: 后续把手的theta变化方向与direction_sign相反!
        """
        # 确定搜索方向: 与运动方向相反
        # 顺时针运动(direction_sign=-1) -> 后续theta增大 -> search_direction=+1
        # 逆时针运动(direction_sign=1) -> 后续theta减小 -> search_direction=-1
        search_direction = -self.direction_sign
        
        # 使用迭代方法找到满足距离条件的theta
        # 从current_theta开始，沿着search_direction方向搜索
        
        # 粗略估计步长
        r_current = self.k * abs(current_theta) if abs(current_theta) > 0.1 else self.k * 0.1
        estimated_dtheta = distance / max(r_current, self.k)
        
        # 二分搜索范围
        # 最小变化：distance对应的角度变化的一半
        # 最大变化：distance对应的角度变化的3倍（考虑螺线曲率）
        min_dtheta = estimated_dtheta * 0.5
        max_dtheta = estimated_dtheta * 3.0
        
        if search_direction > 0:
            theta_min = current_theta + min_dtheta
            theta_max = current_theta + max_dtheta
        else:
            theta_min = current_theta - max_dtheta
            theta_max = current_theta - min_dtheta
        
        # 定义距离误差函数
        def distance_error(theta):
            pos = self.compute_position(theta)
            actual_dist = np.sqrt((pos.x - current_pos.x)**2 + (pos.y - current_pos.y)**2)
            return actual_dist - distance
        
        # 检查边界
        error_min = distance_error(theta_min)
        error_max = distance_error(theta_max)
        
        # 如果边界同号，需要调整范围
        max_attempts = 10
        attempt = 0
        while error_min * error_max > 0 and attempt < max_attempts:
            attempt += 1
            # 扩大搜索范围
            if search_direction > 0:
                if error_min > 0:  # 两个都太远，需要更小的theta
                    theta_min = current_theta + min_dtheta * 0.5
                    min_dtheta *= 0.5
                else:  # 两个都太近，需要更大的theta
                    theta_max = current_theta + max_dtheta * 2.0
                    max_dtheta *= 2.0
            else:
                if error_min > 0:  # 两个都太远
                    theta_max = current_theta - min_dtheta * 0.5
                    min_dtheta *= 0.5
                else:  # 两个都太近
                    theta_min = current_theta - max_dtheta * 2.0
                    max_dtheta *= 2.0
            
            error_min = distance_error(theta_min)
            error_max = distance_error(theta_max)
        
        # 使用二分法求解
        try:
            theta_solution = brentq(
                distance_error, theta_min, theta_max, 
                xtol=np.float64(self.brentq_config['xtol']),
                rtol=np.float64(self.brentq_config['rtol']),
                maxiter=self.brentq_config['maxiter']
            )
            # brentq 返回 float，但类型检查器不确定，所以明确标注
            return float(theta_solution)  # type: ignore[arg-type]
        except ValueError as e:
            # 如果二分法失败，输出诊断信息并使用fsolve作为后备
            print(f"警告: brentq失败 at theta={current_theta:.6f}, distance={distance:.6f}")
            print(f"  范围: [{theta_min:.6f}, {theta_max:.6f}]")
            print(f"  边界误差: [{error_min:.6f}, {error_max:.6f}]")
            print(f"  使用fsolve作为后备方案")
            
            initial_guess = current_theta + search_direction * estimated_dtheta
            result = fsolve(
                distance_error, initial_guess,
                xtol=self.fsolve_config['xtol'],
                maxfev=self.fsolve_config['maxfev'],
                full_output=False
            )
            # full_output=False 时 fsolve 返回 ndarray
            theta_sol = float(result[0])  # type: ignore
            
            # 验证方向
            if (theta_sol - current_theta) * search_direction < 0:
                print(f"  警告: fsolve返回了错误方向的解!")
                print(f"  theta变化: {theta_sol - current_theta:.6f}, 期望方向: {search_direction}")
            
            return theta_sol


class ArcPathHandler(PathHandler):
    """
    圆弧路径处理器
    """
    
    def __init__(self, center: Point2D, radius: float, start_angle: float, end_angle: float, 
                 direction: str = 'counterclockwise', solver_config: Optional[dict] = None):
        """
        初始化圆弧路径
        center: 圆心
        radius: 半径
        start_angle: 起始角度 (rad)
        end_angle: 终止角度 (rad)
        direction: 方向
        solver_config: 求解器配置
        """
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.direction_sign = 1 if direction == 'counterclockwise' else -1
        
        # 求解器配置
        self.solver_config = solver_config or {}
        self.fsolve_config = self.solver_config.get('fsolve', {
            'xtol': 1e-12,
            'maxfev': 10000
        })
        
        # 计算圆弧总长度
        angle_span = abs(end_angle - start_angle)
        self.total_length = radius * angle_span
    
    def compute_position(self, arc_length: float) -> Point2D:
        """
        根据弧长计算位置
        arc_length: 从起始点开始的弧长
        """
        angle = self.start_angle + self.direction_sign * arc_length / self.radius
        return Point2D(
            x=self.center.x + self.radius * np.cos(angle),
            y=self.center.y + self.radius * np.sin(angle)
        )
    
    def angle_from_arc_length(self, arc_length: float) -> float:
        """从弧长计算角度"""
        return self.start_angle + self.direction_sign * arc_length / self.radius
    
    def compute_velocity_magnitude(self, arc_length: float, speed: float) -> float:
        """圆弧上速度大小恒定"""
        return speed
    
    def compute_param_derivative(self, arc_length: float, speed: float) -> float:
        """
        对于圆弧, 参数是弧长本身
        所以 d(arc_length)/dt = speed
        """
        return speed
    
    def solve_next_param(self, current_pos: Point2D, current_arc_length: float, distance: float) -> float:
        """
        求解下一个点的弧长
        
        对于圆弧上的点，使用精确的距离方程求解
        """
        # 使用当前弧长作为初始猜测
        initial_guess = current_arc_length + distance
        
        # 精确求解：找到满足距离约束的弧长
        def distance_equation(arc_length):
            pos = self.compute_position(arc_length)
            dist_sq = (pos.x - current_pos.x)**2 + (pos.y - current_pos.y)**2
            return dist_sq - distance**2
        
        # 使用配置的求解器参数
        result = fsolve(
            distance_equation, 
            initial_guess,
            xtol=self.fsolve_config.get('xtol', 1e-12),
            maxfev=self.fsolve_config.get('maxfev', 10000),
            full_output=False
        )
        
        next_arc_length = float(result[0])
        return next_arc_length


class SpiralInHandler(SpiralPathHandler):
    """盘入螺线 (顺时针)"""
    def __init__(self, pitch: float, solver_config: Optional[dict] = None):
        super().__init__(pitch, direction='clockwise', solver_config=solver_config)


class SpiralOutHandler(SpiralPathHandler):
    """
    盘出螺线 (逆时针, 中心对称)
    盘出螺线与盘入螺线关于原点中心对称
    
    如果盘入螺线是 r = k*theta
    盘出螺线是 r = k*(theta + π), 但角度theta从-π开始
    等价于: (x, y) -> (-x, -y) 的变换
    """
    def __init__(self, pitch: float, solver_config: Optional[dict] = None):
        """
        初始化盘出螺线
        盘出螺线方程: r = k * (theta + π)
        """
        self.pitch = pitch
        self.k = pitch / (2 * np.pi)
        self.direction_sign = 1  # 逆时针
        
        # 求解器配置
        self.solver_config = solver_config or {}
        self.brentq_config = self.solver_config.get('brentq', {
            'xtol': 1e-12,
            'rtol': 1e-12,
            'maxiter': 1000
        })
        self.fsolve_config = self.solver_config.get('fsolve', {
            'xtol': 1e-12,
            'maxfev': 10000
        })
    
    def compute_position(self, theta: float) -> Point2D:
        """计算盘出螺线上的位置"""
        r = self.k * (theta + np.pi)
        return Point2D(
            x=r * np.cos(theta),
            y=r * np.sin(theta)
        )
    
    def compute_param_derivative(self, theta: float, speed: float) -> float:
        """
        计算 dtheta/dt
        
        对于盘出螺线 r = k*(theta + π):
        dr/dtheta = k
        ds = sqrt(r^2 + k^2) * dtheta
           = sqrt(k^2*(theta+π)^2 + k^2) * dtheta
           = k * sqrt((theta+π)^2 + 1) * dtheta
        
        dtheta/dt = v / (k * sqrt((theta+π)^2 + 1))
        """
        dtheta_dt = speed / (self.k * np.sqrt((theta + np.pi)**2 + 1))
        return dtheta_dt  # 逆时针, 正方向
    
    def solve_next_param(self, current_pos: Point2D, current_theta: float, distance: float) -> float:
        """求解下一个角度"""
        def distance_equation(theta):
            pos = self.compute_position(theta)
            dist_sq = (pos.x - current_pos.x)**2 + (pos.y - current_pos.y)**2
            return dist_sq - distance**2
        
        initial_guess = current_theta + 0.1  # 逆时针增加
        theta_solution = fsolve(distance_equation, initial_guess, xtol=1e-12, full_output=False)[0]
        
        # 验证
        max_theta_change = self.pitch / (2 * self.k)
        attempts = 0
        while attempts < 10:
            if theta_solution >= current_theta and abs(self.k * (theta_solution - current_theta)) <= max_theta_change:
                break
            initial_guess = theta_solution + 0.1
            theta_solution = fsolve(distance_equation, initial_guess, xtol=1e-12, full_output=False)[0]
            attempts += 1
        
        return theta_solution


class CompositePath:
    """
    复合路径
    由多段路径组成
    """
    
    def __init__(self):
        self.segments: list[tuple[PathHandler, float]] = []  # (handler, end_time)
        self.total_duration = 0.0
    
    def add_segment(self, handler: PathHandler, duration: float):
        """添加一段路径"""
        self.total_duration += duration
        self.segments.append((handler, self.total_duration))
    
    def get_handler_at_time(self, time: float) -> Tuple[PathHandler, float]:
        """
        获取指定时间对应的路径处理器和该段内的相对时间
        返回: (handler, relative_time)
        """
        if time < 0:
            return self.segments[0][0], 0.0
        
        segment_start = 0.0
        for handler, segment_end in self.segments:
            if time <= segment_end:
                return handler, time - segment_start
            segment_start = segment_end
        
        # 超出范围, 返回最后一段
        last_handler, last_end = self.segments[-1]
        return last_handler, time - (last_end - (self.segments[-1][1] - self.segments[-2][1] if len(self.segments) > 1 else last_end))
