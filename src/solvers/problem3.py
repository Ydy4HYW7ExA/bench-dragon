"""
问题3求解器 - 螺距优化

通过二分搜索找到使龙头恰好到达边界的最小螺距
"""

from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

from ..core.models import DragonConfig, TimeSeriesData, SimulationState
from ..control.simulation import SimulationController, create_initial_state
from ..core.paths import SpiralInHandler
from ..core.geometry import CollisionDetector
from ..io.config import ConfigLoader


class Problem3Solver:
    """
    问题3求解器
    
    使用二分搜索法找到最小螺距,使得龙头恰好到达边界(距离原点0.55m)
    """
    
    def __init__(self, config_path: str = "configs/problem3.yaml"):
        """
        初始化问题3求解器
        
        config_path: 配置文件路径
        """
        # 加载配置
        self.config_dict = ConfigLoader.load_with_inheritance(Path(config_path))
        
        # 提取龙的配置
        dragon_cfg = self.config_dict['dragon']
        self.dragon_config = DragonConfig.from_dict(dragon_cfg)
        
        # 搜索参数
        search_config = self.config_dict.get('optimization', {})
        self.min_pitch = search_config.get('min_pitch', 0.40)
        self.max_pitch = search_config.get('max_pitch', 1.00)
        self.tolerance = search_config.get('tolerance', 0.001)
        self.target_radius = search_config.get('target_radius', 0.55)
        
        # 仿真参数
        self.initial_angle = self.config_dict['path'].get('initial_angle', 16 * 2 * np.pi)
        self.head_speed = self.config_dict['motion']['head_speed']
        self.duration = self.config_dict['motion']['duration']
        self.time_step = self.config_dict['simulation']['time_step']
        
        # 输出设置
        output_cfg = self.config_dict['output']
        self.problem_dir = Path(output_cfg['problem_dir'])
        self.results_dir = self.problem_dir / output_cfg.get('results_subdir', 'results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 碰撞检测器
        self.collision_detector = CollisionDetector(
            self.dragon_config,
            discretization_n=self.config_dict['simulation'].get('collision_discretization_n', 10),
            discretization_m=self.config_dict['simulation'].get('collision_discretization_m', 20)
        )
    
    def simulate_with_pitch(self, pitch: float) -> tuple[bool, float]:
        """
        使用给定螺距运行仿真
        
        pitch: 螺距值
        
        返回: (是否到达目标, 最小半径)
        """
        # 获取求解器配置
        solver_config = self.config_dict['simulation'].get('solvers', {})
        
        # 创建路径处理器
        path_handler = SpiralInHandler(
            pitch=pitch,
            solver_config=solver_config
        )
        
        # 创建初始状态
        initial_state = create_initial_state(
            self.dragon_config,
            path_handler,
            self.initial_angle
        )
        
        # 创建控制器
        controller = SimulationController(
            self.dragon_config,
            path_handler,
            initial_state,
            self.collision_detector
        )
        
        # 记录最小半径
        min_radius = float('inf')
        reached_target = False
        
        # 运行仿真
        t = 0.0
        while t < self.duration:
            # 单步仿真
            new_state = controller.step(self.time_step, self.head_speed)
            
            # 计算龙头半径
            head_pos = new_state.positions[0]
            radius = np.sqrt(head_pos[0]**2 + head_pos[1]**2)
            
            min_radius = min(min_radius, radius)
            
            # 检查是否到达目标
            if radius <= self.target_radius:
                reached_target = True
                break
            
            # 检查碰撞
            if self.collision_detector.check_state_collision(new_state):
                break
            
            t += self.time_step
        
        return reached_target, min_radius
    
    def binary_search_pitch(self) -> float:
        """
        二分搜索最小螺距
        
        返回: 最优螺距
        """
        print("\n" + "=" * 60)
        print("问题3: 螺距优化(二分搜索)")
        print("=" * 60)
        
        print(f"\n搜索参数:")
        print(f"  螺距范围: [{self.min_pitch:.4f}, {self.max_pitch:.4f}]")
        print(f"  目标半径: {self.target_radius:.4f}m")
        print(f"  精度要求: {self.tolerance:.6f}m")
        
        left = self.min_pitch
        right = self.max_pitch
        best_pitch = left
        
        iteration = 0
        print("\n开始二分搜索...")
        
        while right - left > self.tolerance:
            iteration += 1
            mid = (left + right) / 2.0
            
            print(f"\n迭代 {iteration}:")
            print(f"  当前区间: [{left:.6f}, {right:.6f}]")
            print(f"  测试螺距: {mid:.6f}m")
            
            # 测试中间值
            reached, min_radius = self.simulate_with_pitch(mid)
            
            print(f"  最小半径: {min_radius:.6f}m")
            print(f"  到达目标: {'是' if reached else '否'}")
            
            if reached:
                # 螺距太大,减小
                right = mid
                best_pitch = mid
                print(f"  → 螺距过大,搜索左半区间")
            else:
                # 螺距太小,增大
                left = mid
                print(f"  → 螺距过小,搜索右半区间")
        
        print("\n" + "=" * 60)
        print(f"搜索完成!")
        print(f"  最优螺距: {best_pitch:.6f}m")
        print(f"  迭代次数: {iteration}")
        print("=" * 60)
        
        return best_pitch
    
    def export_results(self, optimal_pitch: float):
        """
        导出结果到txt文件
        
        optimal_pitch: 最优螺距
        """
        print("\n" + "=" * 60)
        print("导出结果")
        print("=" * 60)
        
        # 输出文件
        output_file = self.results_dir / self.config_dict['output'].get('result_file', 'result3.txt')
        
        # 使用最优螺距再次运行以获取详细信息
        reached, min_radius = self.simulate_with_pitch(optimal_pitch)
        
        # 写入txt文件（按题目要求的格式，同时包含详细信息）
        with open(output_file, 'w', encoding='utf-8') as f:
            # 题目要求的格式（第一行）
            f.write(f"最小螺距为：{optimal_pitch:.6f} m\n")
            f.write("\n" + "=" * 60 + "\n")
            
            # 详细信息
            f.write("问题3 - 最小螺距优化详细结果\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("优化结果:\n")
            f.write(f"  最优螺距: {optimal_pitch:.6f} m\n")
            f.write(f"  调头空间半径: {self.target_radius:.6f} m\n")
            f.write(f"  实际到达半径: {min_radius:.6f} m\n")
            f.write(f"  误差: {abs(min_radius - self.target_radius):.6f} m\n")
            f.write(f"  是否成功到达: {'是' if reached else '否'}\n\n")
            
            f.write("搜索参数:\n")
            f.write(f"  搜索区间: [{self.min_pitch:.6f}, {self.max_pitch:.6f}] m\n")
            f.write(f"  精度要求: {self.tolerance:.6f} m\n")
            f.write(f"  仿真时长: {self.duration:.1f} s\n")
            f.write(f"  时间步长: {self.time_step:.4f} s\n\n")
            
            f.write("龙的配置:\n")
            f.write(f"  板凳总数: {self.dragon_config.num_segments}\n")
            f.write(f"  龙头长度: {self.dragon_config.head_length:.3f} m\n")
            f.write(f"  龙身长度: {self.dragon_config.body_length:.3f} m\n")
            f.write(f"  板凳宽度: {self.dragon_config.bench_width:.3f} m\n")
            f.write(f"  把手间距(龙头): {self.dragon_config.head_handle_distance:.3f} m\n")
            f.write(f"  把手间距(龙身): {self.dragon_config.body_handle_distance:.3f} m\n")
            f.write(f"  龙头速度: {self.head_speed:.3f} m/s\n\n")
            
            f.write("物理意义:\n")
            f.write(f"  最小螺距{optimal_pitch:.6f}m确保龙头前把手能够沿螺线\n")
            f.write(f"  盘入到调头空间边界(半径{self.target_radius:.3f}m)，\n")
            f.write(f"  为从顺时针盘入切换到逆时针盘出提供必要的调头空间。\n")
            f.write(f"  如果螺距小于此值，龙头无法到达调头空间；\n")
            f.write(f"  如果螺距大于此值，则会浪费空间。\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"结果已导出到: {output_file}")
        print("=" * 60)
    
    def run(self) -> float:
        """运行完整流程"""
        # 搜索最优螺距
        optimal_pitch = self.binary_search_pitch()
        
        # 导出结果
        self.export_results(optimal_pitch)
        
        print("\n问题3求解完成!")
        
        return optimal_pitch


if __name__ == "__main__":
    solver = Problem3Solver()
    optimal_pitch = solver.run()
    print(f"\n最终结果: 最优螺距 = {optimal_pitch:.6f}m")
