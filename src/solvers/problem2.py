"""
问题2求解器 - 碰撞检测

从问题1的最终状态继续运行,检测第一次碰撞的时间点并导出结果
"""

from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

from ..core.models import DragonConfig, TimeSeriesData, SimulationState
from ..control.simulation import SimulationController
from ..core.paths import SpiralInHandler
from ..core.geometry import CollisionDetector
from ..io.export import DataExporter
from ..io.config import ConfigLoader
from .problem1 import Problem1Solver


class Problem2Solver:
    """
    问题2求解器
    
    从问题1的300s状态继续仿真,检测何时发生第一次碰撞
    """
    
    def __init__(self, config_path: str = "configs/problem2.yaml"):
        """
        初始化问题2求解器
        
        config_path: 配置文件路径
        """
        # 加载配置
        self.config_dict = ConfigLoader.load_with_inheritance(Path(config_path))
        
        # 提取龙的配置
        dragon_cfg = self.config_dict['dragon']
        self.dragon_config = DragonConfig.from_dict(dragon_cfg)
        
        # 提取问题参数
        self.spiral_pitch = self.config_dict['path']['spiral_pitch']
        self.head_speed = self.config_dict['motion']['head_speed']
        self.max_duration = self.config_dict['motion'].get('max_duration', 100.0)
        self.time_step = self.config_dict['simulation'].get('time_step', 0.01)
        
        # 输出设置
        output_cfg = self.config_dict['output']
        self.problem_dir = Path(output_cfg['problem_dir'])
        self.results_dir = self.problem_dir / output_cfg.get('results_subdir', 'results')
        self.data_dir = self.problem_dir / output_cfg.get('data_subdir', 'data')
        
        for dir_path in [self.results_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 获取求解器配置
        solver_config = self.config_dict['simulation'].get('solvers', {})
        
        # 创建路径处理器
        self.path_handler = SpiralInHandler(
            pitch=self.spiral_pitch,
            solver_config=solver_config
        )
        
        # 创建碰撞检测器
        self.collision_detector = CollisionDetector(
            self.dragon_config,
            discretization_n=self.config_dict['simulation'].get('collision_discretization_n', 10),
            discretization_m=self.config_dict['simulation'].get('collision_discretization_m', 20)
        )
        
        # 创建数据导出器
        decimal_places = self.config_dict['output'].get('decimal_places', 6)
        self.exporter = DataExporter(self.results_dir, decimal_places)
        
        # 初始状态(将从问题1导入)
        self.initial_state: Optional[SimulationState] = None
    
    def load_initial_state_from_problem1(self) -> SimulationState:
        """
        从问题1的结果加载初始状态
        
        返回: 问题1的最终状态
        """
        print("\n" + "=" * 60)
        print("从问题1加载初始状态")
        print("=" * 60)
        
        # 运行问题1获取最终状态
        problem1_solver = Problem1Solver("configs/problem1.yaml")
        data = problem1_solver.solve()
        
        # 获取最终状态
        final_state = data.states[-1]
        
        # 从状态中获取龙头位置(第一个把手)
        head_pos = final_state.positions[0]
        
        print(f"问题1最终时间: {final_state.time:.2f}s")
        print(f"问题1最终位置: ({head_pos[0]:.4f}, {head_pos[1]:.4f})")
        print(f"问题1最终角度: {final_state.angles[0]:.6f} rad")
        print("=" * 60)
        
        return final_state
    
    def solve(self) -> tuple[TimeSeriesData, Optional[float]]:
        """
        求解问题2 - 检测碰撞时间
        
        返回: (仿真数据, 碰撞时间)
        """
        print("\n" + "=" * 60)
        print("问题2: 碰撞检测")
        print("=" * 60)
        
        # 加载初始状态
        self.initial_state = self.load_initial_state_from_problem1()
        
        # 创建仿真控制器
        controller = SimulationController(
            self.dragon_config,
            self.path_handler,
            self.initial_state,
            self.collision_detector
        )
        
        print(f"\n继续仿真参数:")
        print(f"  起始时间: {self.initial_state.time:.2f}s")
        print(f"  时间步长: {self.time_step}s")
        print(f"  最大时长: {self.max_duration}s")
        print(f"  检测碰撞: 是")
        
        # 运行仿真直到碰撞
        print("\n开始仿真...")
        
        times = []
        states = []
        collision_time: Optional[float] = None
        
        t = 0.0
        step_count = 0
        
        while t < self.max_duration:
            # 单步仿真
            new_state = controller.step(self.time_step, self.head_speed)
            
            times.append(new_state.time)
            states.append(new_state)
            
            # 检测碰撞
            if self.collision_detector.check_state_collision(new_state):
                collision_time = new_state.time
                print(f"\n检测到碰撞!")
                print(f"  碰撞时间: {collision_time:.4f}s")
                print(f"  从问题1结束后运行时间: {t:.4f}s")
                break
            
            t += self.time_step
            step_count += 1
            
            # 每10秒报告一次进度
            if step_count % 1000 == 0:
                print(f"  进度: {t:.1f}s / {self.max_duration}s, 当前时间: {new_state.time:.1f}s")
        
        if collision_time is None:
            print(f"\n警告: 在{self.max_duration}s内未检测到碰撞")
        
        # 创建时间序列数据
        data = TimeSeriesData(times=np.array(times), states=states)
        
        print("\n" + "=" * 60)
        print(f"仿真完成")
        print(f"  总时间点数: {len(data.times)}")
        print(f"  时间范围: {data.times[0]:.2f}s - {data.times[-1]:.2f}s")
        if collision_time is not None:
            print(f"  碰撞时间: {collision_time:.4f}s")
        print("=" * 60)
        
        return data, collision_time
    
    def export_results(self, data: TimeSeriesData, collision_time: Optional[float]):
        """
        导出结果到Excel
        
        data: 仿真数据
        collision_time: 碰撞时间
        """
        print("\n" + "=" * 60)
        print("导出结果")
        print("=" * 60)
        
        # 导出主要结果文件
        result_file = self.config_dict['output'].get('result_file', 'result2.xlsx')
        self.exporter.export_problem1_result(data, self.dragon_config, result_file)
        
        # 额外保存碰撞时间信息
        info_file = self.results_dir / "collision_info.txt"
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("问题2结果 - 碰撞检测\n")
            f.write("=" * 60 + "\n\n")
            if collision_time is not None:
                f.write(f"碰撞时间: {collision_time:.4f} s\n")
                f.write(f"从问题1结束后运行时间: {collision_time - data.times[0]:.4f} s\n")
            else:
                f.write(f"在{self.max_duration}s内未检测到碰撞\n")
            f.write(f"\n仿真参数:\n")
            f.write(f"  螺距: {self.spiral_pitch} m\n")
            f.write(f"  龙头速度: {self.head_speed} m/s\n")
            f.write(f"  时间步长: {self.time_step} s\n")
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"结果已导出到: {self.results_dir}")
        print("=" * 60)
    
    def generate_visualization(self, data: TimeSeriesData, collision_time: Optional[float]):
        """
        生成可视化动画(并行渲染)
        
        data: 仿真数据
        collision_time: 碰撞时间
        """
        from ..visualization.animator import ParallelAnimationGenerator
        
        print("\n" + "=" * 60)
        print("生成可视化动画(并行渲染)")
        print("=" * 60)
        
        # 创建并行动画生成器
        animator = ParallelAnimationGenerator(
            config=self.dragon_config,
            fps=self.config_dict['visualization'].get('animation_fps', 10),
            dpi=self.config_dict['visualization'].get('figure_dpi', 100)
        )
        
        # 高亮节点
        highlight_indices = self.config_dict['visualization'].get('highlight_segments', [0, 1, 51, 101, 151, 201, 223])
        
        # 动画输出目录
        animations_dir = Path(self.config_dict['output'].get('animations_dir', 'outputs/animations'))
        
        # 生成MP4
        mp4_file = animations_dir / 'problem2.mp4'
        animator.generate_mp4(
            data=data,
            output_path=mp4_file,
            highlight_indices=highlight_indices,
            frame_interval=10,
            figsize=(12, 12)
        )
        
        # 生成GIF
        gif_file = animations_dir / 'problem2.gif'
        animator.generate_gif(
            data=data,
            output_path=gif_file,
            highlight_indices=highlight_indices,
            frame_interval=20,
            figsize=(10, 10)
        )
        
        print("=" * 60)
    
    def run(self) -> tuple[TimeSeriesData, Optional[float]]:
        """运行完整流程"""
        # 求解
        data, collision_time = self.solve()
        
        # 导出结果
        self.export_results(data, collision_time)
        
        print("\n问题2求解完成!")
        
        return data, collision_time


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='求解问题2: 碰撞检测')
    parser.add_argument('--no-viz', action='store_true',
                       help='不生成可视化')
    args = parser.parse_args()
    
    solver = Problem2Solver()
    data, collision_time = solver.run()
    
    if collision_time:
        print(f"\n最终结果: 碰撞发生在 {collision_time:.4f}s")
    else:
        print(f"\n最终结果: 未检测到碰撞")
    
    # 生成可视化
    if not args.no_viz:
        solver.generate_visualization(data, collision_time)

