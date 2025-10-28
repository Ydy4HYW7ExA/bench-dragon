"""
仿真控制器
管理整个仿真流程
"""

import numpy as np
from typing import List, Optional, Callable
from tqdm import tqdm

from ..core.models import DragonConfig, SimulationState, TimeSeriesData, Point2D
from ..core.paths import PathHandler
from ..core.geometry import CollisionDetector
from .kinematics import DragonKinematics
from .triggers import Trigger


class SimulationController:
    """
    仿真控制器
    负责管理仿真流程、状态更新和触发器
    """
    
    def __init__(self, 
                 config: DragonConfig,
                 path_handler: PathHandler,
                 initial_state: SimulationState,
                 collision_detector: Optional[CollisionDetector] = None):
        """
        初始化仿真控制器
        
        config: 龙的配置
        path_handler: 路径处理器
        initial_state: 初始状态
        collision_detector: 碰撞检测器(可选)
        """
        self.config = config
        self.path_handler = path_handler
        self.current_state = initial_state.clone()
        self.initial_state = initial_state.clone()
        
        self.kinematics = DragonKinematics(config, path_handler)
        self.collision_detector = collision_detector
        
        self.triggers: List[Trigger] = []
        self.history: List[SimulationState] = [initial_state.clone()]
        self.times: List[float] = [initial_state.time]
        
        self.should_stop = False
    
    def add_trigger(self, trigger: Trigger):
        """添加触发器"""
        self.triggers.append(trigger)
    
    def step(self, dt: float, speed: float) -> SimulationState:
        """
        执行一步仿真
        
        dt: 时间步长
        speed: 龙头速度
        
        返回: 新状态
        """
        # 计算新状态
        new_state = self.kinematics.integrate_step(self.current_state, dt, speed)
        
        # 更新当前状态
        self.current_state = new_state
        
        # 保存历史
        self.history.append(new_state.clone())
        self.times.append(new_state.time)
        
        # 检查触发器
        for trigger in self.triggers:
            if trigger.check(new_state):
                trigger.execute(self)
        
        return new_state
    
    def run(self, duration: float, dt: float, speed: float, 
            show_progress: bool = True) -> TimeSeriesData:
        """
        运行仿真
        
        duration: 仿真时长
        dt: 时间步长
        speed: 龙头速度
        show_progress: 是否显示进度条
        
        返回: 时间序列数据
        """
        num_steps = int(duration / dt)
        
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="仿真进行中")
        
        for _ in iterator:
            if self.should_stop:
                break
            
            self.step(dt, speed)
            
            # 检查碰撞
            if self.collision_detector is not None:
                if self.collision_detector.check_state_collision(self.current_state):
                    print(f"\n检测到碰撞! 时间: {self.current_state.time:.2f}s")
                    self.should_stop = True
                    break
        
        return self.get_history()
    
    def run_until(self, condition: Callable[[SimulationState], bool],
                  dt: float, speed: float, max_time: float = 1000.0,
                  show_progress: bool = True) -> TimeSeriesData:
        """
        运行仿真直到满足条件
        
        condition: 停止条件
        dt: 时间步长
        speed: 龙头速度
        max_time: 最大仿真时间
        show_progress: 是否显示进度条
        
        返回: 时间序列数据
        """
        max_steps = int(max_time / dt)
        
        iterator = range(max_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="仿真进行中")
        
        for _ in iterator:
            if self.should_stop or condition(self.current_state):
                break
            
            self.step(dt, speed)
        
        return self.get_history()
    
    def get_history(self) -> TimeSeriesData:
        """获取历史数据"""
        return TimeSeriesData(
            times=np.array(self.times),
            states=self.history,
            metadata={
                'config': self.config.to_dict(),
                'stopped_early': self.should_stop
            }
        )
    
    def get_state(self) -> SimulationState:
        """获取当前状态"""
        return self.current_state.clone()
    
    def reset(self):
        """重置仿真"""
        self.current_state = self.initial_state.clone()
        self.history = [self.initial_state.clone()]
        self.times = [self.initial_state.time]
        self.should_stop = False
        
        for trigger in self.triggers:
            trigger.reset()
    
    def stop(self):
        """停止仿真"""
        self.should_stop = True


def create_initial_state(config: DragonConfig, path_handler: PathHandler,
                         initial_angle: float) -> SimulationState:
    """
    创建初始状态
    
    config: 龙的配置
    path_handler: 路径处理器
    initial_angle: 题目中给定的初始角度（龙头前把手的位置）
    
    返回: 初始状态
    
    注意：
    - 题目：问题1中"初始时，龙头位于螺线第16圈"，所以initial_angle是龙头的位置
    - 龙尾在龙头后面（路径后方），theta更大（顺时针）或更小（逆时针）
    - 从龙头开始正向计算到龙尾
    """
    num_handles = config.num_handles
    kinematics = DragonKinematics(config, path_handler)
    
    positions = np.zeros((num_handles, 2))
    angles = np.zeros(num_handles)
    velocities = np.zeros(num_handles)
    
    # 龙头前把手在initial_angle
    head_pos = path_handler.compute_position(initial_angle)
    positions[0] = head_pos.to_array()
    angles[0] = initial_angle
    
    # 从龙头正向计算到龙尾
    for i in range(1, num_handles):
        # 确定把手间距
        if i == 1:
            # 龙头后把手
            distance = config.head_handle_distance
        else:
            # 龙身/龙尾
            distance = config.body_handle_distance
        
        # 从前一个把手计算当前把手
        prev_pos = Point2D.from_array(positions[i-1])
        prev_ang = angles[i-1]
        
        handle_state = kinematics.compute_following_handle(prev_pos, prev_ang, distance)
        positions[i] = handle_state['position'].to_array()
        angles[i] = handle_state['angle']
    
    return SimulationState(
        time=0.0,
        positions=positions,
        angles=angles,
        velocities=velocities
    )
