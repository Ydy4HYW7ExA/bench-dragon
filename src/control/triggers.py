"""
触发器系统
允许在仿真过程中根据条件执行动作
"""

from typing import Callable, TYPE_CHECKING, Optional
from ..core.models import SimulationState

if TYPE_CHECKING:
    from .simulation import SimulationController


class Trigger:
    """
    触发器
    当满足条件时执行动作
    """
    
    def __init__(self, 
                 condition: Callable[[SimulationState], bool],
                 action: Callable[['SimulationController'], None],
                 once: bool = True,
                 name: str = ""):
        """
        初始化触发器
        
        condition: 条件函数, 接受SimulationState, 返回bool
        action: 动作函数, 接受SimulationController
        once: 是否只触发一次
        name: 触发器名称
        """
        self.condition = condition
        self.action = action
        self.once = once
        self.name = name
        self.triggered = False
    
    def check(self, state: SimulationState) -> bool:
        """检查条件是否满足"""
        if self.once and self.triggered:
            return False
        return self.condition(state)
    
    def execute(self, controller: 'SimulationController'):
        """执行动作"""
        self.action(controller)
        if self.once:
            self.triggered = True
    
    def reset(self):
        """重置触发状态"""
        self.triggered = False


# 预定义的常用触发条件
def time_trigger(target_time: float):
    """时间触发条件"""
    return lambda state: state.time >= target_time


def collision_trigger():
    """碰撞触发条件"""
    def check_collision(state: SimulationState) -> bool:
        # 这里需要访问碰撞检测器
        # 在实际使用中,会从controller获取
        return False  # 占位
    return check_collision


def position_trigger(handle_idx: int, x: Optional[float] = None, y: Optional[float] = None, radius: float = 0.1):
    """位置触发条件"""
    def check_position(state: SimulationState) -> bool:
        pos = state.positions[handle_idx]
        if x is not None and abs(pos[0] - x) > radius:
            return False
        if y is not None and abs(pos[1] - y) > radius:
            return False
        return True
    return check_position
