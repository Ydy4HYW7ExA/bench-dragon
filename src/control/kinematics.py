"""
运动学计算
计算龙队各部分的运动状态
"""

import numpy as np
from typing import List, Tuple
from ..core.models import DragonConfig, SimulationState, Point2D
from ..core.paths import PathHandler


class DragonKinematics:
    """
    板凳龙运动学计算
    负责计算龙头和所有龙身/龙尾的位置和速度
    """
    
    def __init__(self, config: DragonConfig, path_handler: PathHandler):
        """
        初始化运动学计算器
        config: 龙的配置
        path_handler: 路径处理器
        """
        self.config = config
        self.path_handler = path_handler
    
    def compute_head_next_state(self, current_state: SimulationState, dt: float, speed: float) -> dict:
        """
        计算龙头在下一时刻的状态
        
        返回: {'angle': float, 'position': Point2D, 'velocity': float}
        """
        # 当前龙头角度
        current_angle = current_state.angles[0]
        
        # 计算角度变化率
        dangle_dt = self.path_handler.compute_param_derivative(current_angle, speed)
        
        # 使用欧拉法或RK4积分
        # 这里使用简单的欧拉法,可以改进
        next_angle = current_angle + dangle_dt * dt
        
        # 计算新位置
        next_position = self.path_handler.compute_position(next_angle)
        
        # 速度
        next_velocity = self.path_handler.compute_velocity_magnitude(next_angle, speed)
        
        return {
            'angle': next_angle,
            'position': next_position,
            'velocity': next_velocity
        }
    
    def compute_following_handle(self, prev_position: Point2D, prev_angle: float, 
                                  distance: float) -> dict:
        """
        计算跟随的把手位置
        
        prev_position: 前一个把手的位置
        prev_angle: 前一个把手对应的路径参数
        distance: 把手之间的距离
        
        返回: {'angle': float, 'position': Point2D}
        """
        # 使用路径处理器求解
        next_angle = self.path_handler.solve_next_param(prev_position, prev_angle, distance)
        next_position = self.path_handler.compute_position(next_angle)
        
        return {
            'angle': next_angle,
            'position': next_position
        }
    
    def compute_all_handles(self, head_state: dict, prev_state: SimulationState, speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算所有把手的位置和角度
        
        head_state: 龙头状态
        prev_state: 前一时刻的状态(用于初始猜测)
        speed: 龙头速度
        
        返回: (positions, angles)
        """
        num_handles = self.config.num_handles
        positions = np.zeros((num_handles, 2))
        angles = np.zeros(num_handles)
        
        # 龙头前把手
        positions[0] = head_state['position'].to_array()
        angles[0] = head_state['angle']
        
        # 依次计算每个后续把手
        for i in range(1, num_handles):
            # 确定把手间距
            if i == 1:
                # 龙头后把手
                distance = self.config.head_handle_distance
            else:
                # 龙身/龙尾
                distance = self.config.body_handle_distance
            
            # 计算
            prev_pos = Point2D.from_array(positions[i-1])
            prev_ang = angles[i-1]
            
            handle_state = self.compute_following_handle(prev_pos, prev_ang, distance)
            
            positions[i] = handle_state['position'].to_array()
            angles[i] = handle_state['angle']
        
        return positions, angles
    
    def compute_velocities(self, positions: np.ndarray, prev_positions: np.ndarray, dt: float) -> np.ndarray:
        """
        计算所有把手的速度
        
        使用简单的差分法: v = (x(t) - x(t-dt)) / dt
        """
        velocities = np.linalg.norm(positions - prev_positions, axis=1) / dt
        return velocities
    
    def integrate_step(self, current_state: SimulationState, dt: float, speed: float) -> SimulationState:
        """
        执行一步积分,计算下一时刻的状态
        
        current_state: 当前状态
        dt: 时间步长
        speed: 龙头速度
        
        返回: 下一时刻的状态
        """
        # 计算龙头新状态
        head_state = self.compute_head_next_state(current_state, dt, speed)
        
        # 计算所有把手
        new_positions, new_angles = self.compute_all_handles(head_state, current_state, speed)
        
        # 计算速度
        new_velocities = self.compute_velocities(new_positions, current_state.positions, dt)
        
        # 创建新状态
        new_state = SimulationState(
            time=current_state.time + dt,
            positions=new_positions,
            angles=new_angles,
            velocities=new_velocities,
            metadata=current_state.metadata.copy()
        )
        
        return new_state
