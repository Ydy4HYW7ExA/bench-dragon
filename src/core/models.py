"""
核心数据模型
定义系统中使用的所有数据结构
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path
import pickle


def clean_float(value: float, threshold: float = 1e-10) -> float:
    """
    清理浮点数，将接近0的值设为精确的0
    避免出现-0.000000这样的值
    """
    if abs(value) < threshold:
        return 0.0
    return value


@dataclass
class Point2D:
    """二维点"""
    x: float
    y: float
    
    def __post_init__(self):
        """清理坐标值"""
        self.x = clean_float(self.x)
        self.y = clean_float(self.y)
    
    def distance_to(self, other: 'Point2D') -> float:
        """计算到另一个点的距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_to(self, other: 'Point2D') -> float:
        """计算到另一个点的角度"""
        return np.arctan2(other.y - self.y, other.x - self.x)
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point2D':
        """从numpy数组创建"""
        return cls(x=float(arr[0]), y=float(arr[1]))
    
    def __repr__(self) -> str:
        return f"Point2D(x={self.x:.6f}, y={self.y:.6f})"


@dataclass
class DragonSegment:
    """
    板凳龙的一节
    """
    segment_id: int              # 节编号 (0=龙头, 1-221=龙身, 222=龙尾)
    front_handle: Point2D        # 前把手位置
    back_handle: Point2D         # 后把手位置
    angle: float                 # 角度 (对于螺线路径)
    velocity: float              # 速度
    is_head: bool = False        # 是否为龙头
    is_tail: bool = False        # 是否为龙尾
    
    def get_length(self, config: 'DragonConfig') -> float:
        """获取板凳长度"""
        if self.is_head:
            return config.head_length
        return config.body_length
    
    def get_handle_distance(self, config: 'DragonConfig') -> float:
        """获取把手间距"""
        if self.is_head:
            return config.head_handle_distance
        return config.body_handle_distance
    
    def get_center(self) -> Point2D:
        """获取板凳中心点"""
        return Point2D(
            x=(self.front_handle.x + self.back_handle.x) / 2,
            y=(self.front_handle.y + self.back_handle.y) / 2
        )
    
    def get_direction(self) -> float:
        """获取板凳方向角"""
        return self.front_handle.angle_to(self.back_handle)
    
    def get_corners(self, config: 'DragonConfig') -> List[Point2D]:
        """
        获取板凳的四个角点
        返回顺序: 后左, 前左, 前右, 后右 (逆时针)
        """
        direction = self.get_direction()
        handle_distance = self.get_handle_distance(config)
        bench_length = self.get_length(config)
        bench_width = config.bench_width
        
        # 板凳两端超出把手的长度
        offset = (bench_length - handle_distance) / 2
        
        # 计算四个角点
        cos_dir = np.cos(direction)
        sin_dir = np.sin(direction)
        
        corners = []
        # 后左角
        corners.append(Point2D(
            x=self.back_handle.x + offset * cos_dir - bench_width/2 * sin_dir,
            y=self.back_handle.y + offset * sin_dir + bench_width/2 * cos_dir
        ))
        # 前左角
        corners.append(Point2D(
            x=self.front_handle.x - offset * cos_dir - bench_width/2 * sin_dir,
            y=self.front_handle.y - offset * sin_dir + bench_width/2 * cos_dir
        ))
        # 前右角
        corners.append(Point2D(
            x=self.front_handle.x - offset * cos_dir + bench_width/2 * sin_dir,
            y=self.front_handle.y - offset * sin_dir - bench_width/2 * cos_dir
        ))
        # 后右角
        corners.append(Point2D(
            x=self.back_handle.x + offset * cos_dir + bench_width/2 * sin_dir,
            y=self.back_handle.y + offset * sin_dir - bench_width/2 * cos_dir
        ))
        
        return corners


@dataclass
class SimulationState:
    """
    仿真状态
    包含某个时刻整个龙队的完整状态
    """
    time: float                          # 当前时间
    positions: np.ndarray                # 所有把手的位置 (n_handles, 2)
    angles: np.ndarray                   # 所有把手对应的角度 (n_handles,)
    velocities: np.ndarray               # 所有把手的速度 (n_handles,)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """确保数据类型正确"""
        self.positions = np.asarray(self.positions, dtype=float)
        self.angles = np.asarray(self.angles, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
    
    def get_num_handles(self) -> int:
        """获取把手数量"""
        return len(self.positions)
    
    def get_segment(self, idx: int, config: 'DragonConfig') -> DragonSegment:
        """
        获取第idx节板凳
        idx: 0 到 num_segments (包含)
        """
        num_segments = config.num_segments
        
        # 前把手索引就是idx
        front_handle = Point2D.from_array(self.positions[idx])
        
        # 后把手索引
        if idx < num_segments:
            back_handle = Point2D.from_array(self.positions[idx + 1])
        else:
            # 龙尾后把手没有对应的板凳
            back_handle = front_handle
        
        return DragonSegment(
            segment_id=idx,
            front_handle=front_handle,
            back_handle=back_handle,
            angle=self.angles[idx],
            velocity=self.velocities[idx],
            is_head=(idx == 0),
            is_tail=(idx == num_segments - 1)
        )
    
    def get_all_segments(self, config: 'DragonConfig') -> List[DragonSegment]:
        """获取所有板凳节"""
        return [self.get_segment(i, config) for i in range(config.num_segments)]
    
    def clone(self) -> 'SimulationState':
        """深拷贝"""
        return SimulationState(
            time=self.time,
            positions=self.positions.copy(),
            angles=self.angles.copy(),
            velocities=self.velocities.copy(),
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'time': self.time,
            'positions': self.positions.tolist(),
            'angles': self.angles.tolist(),
            'velocities': self.velocities.tolist(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationState':
        """从字典创建"""
        return cls(
            time=data['time'],
            positions=np.array(data['positions']),
            angles=np.array(data['angles']),
            velocities=np.array(data['velocities']),
            metadata=data.get('metadata', {})
        )


@dataclass
class TimeSeriesData:
    """
    时间序列数据
    存储整个仿真过程的所有状态
    """
    times: np.ndarray                    # 时间数组
    states: List[SimulationState]        # 状态列表
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据一致性"""
        if len(self.times) != len(self.states):
            raise ValueError(f"时间数组长度({len(self.times)})与状态列表长度({len(self.states)})不匹配")
    
    def __len__(self) -> int:
        return len(self.times)
    
    def __getitem__(self, idx: int) -> tuple[float, SimulationState]:
        """获取指定索引的时间和状态"""
        return self.times[idx], self.states[idx]
    
    def get_state_at_time(self, time: float, method: str = 'nearest') -> Optional[SimulationState]:
        """
        获取指定时间的状态
        method: 'nearest' 或 'interpolate'
        """
        if method == 'nearest':
            idx = np.argmin(np.abs(self.times - time))
            return self.states[idx]
        else:
            # TODO: 实现插值
            raise NotImplementedError("插值方法暂未实现")
    
    def slice(self, start_time: float, end_time: float) -> 'TimeSeriesData':
        """切片,获取时间范围内的数据"""
        mask = (self.times >= start_time) & (self.times <= end_time)
        indices = np.where(mask)[0]
        
        return TimeSeriesData(
            times=self.times[indices],
            states=[self.states[i] for i in indices],
            metadata=self.metadata.copy()
        )
    
    def save(self, filepath: Path):
        """保存到文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'times': self.times,
                'states': [s.to_dict() for s in self.states],
                'metadata': self.metadata
            }, f)
    
    @classmethod
    def load(cls, filepath: Path) -> 'TimeSeriesData':
        """从文件加载"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return cls(
            times=data['times'],
            states=[SimulationState.from_dict(s) for s in data['states']],
            metadata=data.get('metadata', {})
        )


@dataclass
class DragonConfig:
    """
    板凳龙配置参数
    """
    # 板凳尺寸
    head_length: float               # 龙头长度 (m)
    body_length: float               # 龙身/龙尾长度 (m)
    bench_width: float               # 板凳宽度 (m)
    
    # 把手参数
    handle_offset: float             # 把手距板头距离 (m)
    handle_diameter: float           # 把手直径 (m)
    
    # 龙结构
    num_segments: int                # 总段数
    
    # 导出属性
    head_handle_distance: float = field(init=False)
    body_handle_distance: float = field(init=False)
    num_handles: int = field(init=False)
    
    def __post_init__(self):
        """计算导出属性"""
        self.head_handle_distance = self.head_length - 2 * self.handle_offset
        self.body_handle_distance = self.body_length - 2 * self.handle_offset
        self.num_handles = self.num_segments + 1  # 龙头前把手 + 每节的后把手
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DragonConfig':
        """从字典创建"""
        return cls(
            head_length=data['head_length'],
            body_length=data['body_length'],
            bench_width=data['bench_width'],
            handle_offset=data['handle_offset'],
            handle_diameter=data['handle_diameter'],
            num_segments=data['num_segments']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'head_length': self.head_length,
            'body_length': self.body_length,
            'bench_width': self.bench_width,
            'handle_offset': self.handle_offset,
            'handle_diameter': self.handle_diameter,
            'num_segments': self.num_segments,
            'head_handle_distance': self.head_handle_distance,
            'body_handle_distance': self.body_handle_distance,
            'num_handles': self.num_handles
        }
    
    def validate(self) -> bool:
        """验证配置的合理性"""
        if self.head_length <= 0 or self.body_length <= 0:
            return False
        if self.bench_width <= 0:
            return False
        if self.handle_offset < 0 or self.handle_offset >= min(self.head_length, self.body_length) / 2:
            return False
        if self.num_segments <= 0:
            return False
        return True


@dataclass
class PathConfig:
    """路径配置"""
    path_type: str                   # 路径类型: 'spiral', 'arc', 'composite'
    parameters: Dict[str, Any]       # 路径参数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'path_type': self.path_type,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PathConfig':
        """从字典创建"""
        return cls(
            path_type=data['path_type'],
            parameters=data['parameters']
        )
