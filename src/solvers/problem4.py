"""
问题4求解器 - 双圆弧S形调头路径

完整实现:
1. 螺线盘入段(螺距1.7m, 顺时针)
2. 双圆弧S形调头段(R1=2*R2, 相切连接)
3. 螺线盘出段(逆时针, 与盘入中心对称)

优化圆弧参数,使调头曲线尽可能短,同时满足:
- 调头空间直径9m(半径4.5m)
- 与螺线相切

输出-100s到100s的位置和速度数据
"""

from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, List

from ..core.models import DragonConfig, TimeSeriesData, SimulationState
from ..control.simulation import SimulationController, create_initial_state
from ..core.uturn_geometry import compute_uturn_geometry, optimize_uturn_radius
from ..core.uturn_path import DoubleArcUTurnHandler
from ..core.geometry import CollisionDetector
from ..io.export import DataExporter
from ..io.config import ConfigLoader
from ..visualization.renderer import DragonRenderer
from ..visualization.animator import AnimationGenerator
from ..visualization.plotter import Plotter


class Problem4Solver:
    """
    问题4求解器 - 双圆弧S形调头路径设计与仿真
    """
    
    def __init__(self, config_path: str = "configs/problem4.yaml"):
        """
        初始化问题4求解器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config_dict = ConfigLoader.load_with_inheritance(Path(config_path))
        
        # 龙的配置
        dragon_cfg = self.config_dict['dragon']
        self.dragon_config = DragonConfig.from_dict(dragon_cfg)
        
        # 路径参数
        path_cfg = self.config_dict['path']
        self.pitch = path_cfg.get('pitch', 1.7)  # 螺距1.7m
        self.turnaround_diameter = path_cfg.get('turnaround_diameter', 9.0)  # 调头空间直径9m
        self.turnaround_radius = self.turnaround_diameter / 2  # 半径4.5m
        
        # 运动参数
        motion_cfg = self.config_dict['motion']
        self.head_speed = motion_cfg.get('head_speed', 1.0)  # 龙头速度1m/s
        self.time_range = motion_cfg.get('time_range', [-100, 100])  # 时间范围
        
        # 仿真参数
        sim_cfg = self.config_dict['simulation']
        self.time_step = sim_cfg.get('time_step', 0.1)
        
        # 输出设置
        output_cfg = self.config_dict['output']
        self.problem_dir = Path(output_cfg['problem_dir'])
        self.results_dir = self.problem_dir / output_cfg.get('results_subdir', 'results')
        self.data_dir = self.problem_dir / output_cfg.get('data_subdir', 'data')
        self.animations_dir = self.problem_dir / output_cfg.get('animations_subdir', 'animations')
        self.figures_dir = self.problem_dir / output_cfg.get('figures_subdir', 'figures')
        
        for dir_path in [self.results_dir, self.data_dir, self.animations_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.decimal_places = output_cfg.get('decimal_places', 6)
        
        # 数据导出器
        self.exporter = DataExporter(self.results_dir, self.decimal_places)
        
        # 特殊时刻
        self.special_times = motion_cfg.get('special_times', [-100, -50, 0, 50, 100])
        
        # 特殊节点
        self.special_indices = motion_cfg.get('special_indices', [0, 50, 100, 150, 200])
        
        print("\n" + "="*60)
        print("问题4求解器初始化")
        print("="*60)
        print(f"螺距: {self.pitch}m")
        print(f"调头空间直径: {self.turnaround_diameter}m")
        print(f"龙头速度: {self.head_speed}m/s")
        print(f"时间范围: {self.time_range}s")
    
    def design_uturn_path(self) -> DoubleArcUTurnHandler:
        """
        设计调头路径
        
        返回:
            DoubleArcUTurnHandler: 调头路径处理器
        """
        print("\n设计双圆弧S形调头路径...")
        
        # 优化圆弧半径r2,使调头曲线最短
        best_r2, geometry = optimize_uturn_radius(
            pitch=self.pitch,
            turnaround_radius=self.turnaround_radius,
            min_r2=0.5,
            max_r2=3.0,
            num_samples=30
        )
        
        print(f"\n调头曲线几何参数:")
        print(f"  第二段圆弧半径 r2: {best_r2:.4f}m")
        print(f"  第一段圆弧半径 r1: {geometry.r1:.4f}m (= 2*r2)")
        print(f"  第一段圆弧长度: {geometry.arc1_length:.4f}m")
        print(f"  第二段圆弧长度: {geometry.arc2_length:.4f}m")
        print(f"  调头曲线总长度: {geometry.total_length:.4f}m")
        print(f"  盘入螺线结束点: ({geometry.spiral_in_end_point.x:.4f}, {geometry.spiral_in_end_point.y:.4f})")
        print(f"  盘出螺线起始点: ({geometry.spiral_out_start_point.x:.4f}, {geometry.spiral_out_start_point.y:.4f})")
        
        # 计算-100s时龙头位置(需要确定螺线盘入段长度)
        # 在t=0时,龙头应该在调头曲线中点附近
        # 设t=0时龙头刚好在第一段圆弧结束点
        
        uturn_mid_arc_length = geometry.arc1_length  # 调头中点对应的弧长(在调头段内)
        
        # t=0时的总弧长
        arc_length_at_t0 = 0.0  # 待定
        
        # 从t=-100到t=0, 龙头移动距离
        distance_before_t0 = self.head_speed * abs(self.time_range[0])  # 100m
        
        # 在t=0时,龙头应该位于调头曲线的某个位置
        # 为简化,设t=0时龙头刚好到达调头起点
        # 则 arc_length_at_t0 = spiral_in_length
        
        # 从t=-100开始,需要螺线盘入段长度 >= 100m
        spiral_in_length = distance_before_t0  # 100m
        
        # 创建调头路径处理器
        uturn_path = DoubleArcUTurnHandler(
            geometry=geometry,
            pitch=self.pitch,
            spiral_in_length=spiral_in_length,
            clockwise_in=True
        )
        
        # 保存几何参数到文件
        self._save_path_info(geometry, best_r2, spiral_in_length)
        
        return uturn_path
    
    def _save_path_info(self, geometry, r2: float, spiral_in_length: float):
        """保存路径信息到txt文件"""
        info_file = self.results_dir / "path_info.txt"
        
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("问题4 - 双圆弧S形调头路径参数\n")
            f.write("="*50 + "\n\n")
            
            f.write("路径设计:\n")
            f.write(f"  螺距: {self.pitch:.6f} m\n")
            f.write(f"  调头空间直径: {self.turnaround_diameter:.6f} m\n")
            f.write(f"  调头空间半径: {self.turnaround_radius:.6f} m\n\n")
            
            f.write("调头曲线优化结果:\n")
            f.write(f"  第二段圆弧半径 r2: {r2:.6f} m\n")
            f.write(f"  第一段圆弧半径 r1: {geometry.r1:.6f} m (= 2*r2)\n\n")
            
            f.write("调头曲线几何参数:\n")
            f.write(f"  第一段圆弧:\n")
            f.write(f"    圆心: ({geometry.arc1_center.x:.6f}, {geometry.arc1_center.y:.6f})\n")
            f.write(f"    起始角: {np.degrees(geometry.arc1_start_angle):.6f}°\n")
            f.write(f"    结束角: {np.degrees(geometry.arc1_end_angle):.6f}°\n")
            f.write(f"    弧长: {geometry.arc1_length:.6f} m\n\n")
            
            f.write(f"  第二段圆弧:\n")
            f.write(f"    圆心: ({geometry.arc2_center.x:.6f}, {geometry.arc2_center.y:.6f})\n")
            f.write(f"    起始角: {np.degrees(geometry.arc2_start_angle):.6f}°\n")
            f.write(f"    结束角: {np.degrees(geometry.arc2_end_angle):.6f}°\n")
            f.write(f"    弧长: {geometry.arc2_length:.6f} m\n\n")
            
            f.write(f"  调头曲线总长度: {geometry.total_length:.6f} m\n\n")
            
            f.write("螺线衔接参数:\n")
            f.write(f"  盘入螺线段长度: {spiral_in_length:.6f} m\n")
            f.write(f"  盘入螺线结束点: ({geometry.spiral_in_end_point.x:.6f}, {geometry.spiral_in_end_point.y:.6f})\n")
            f.write(f"  盘入螺线结束极角: {np.degrees(geometry.spiral_in_end_angle):.6f}°\n")
            f.write(f"  盘出螺线起始点: ({geometry.spiral_out_start_point.x:.6f}, {geometry.spiral_out_start_point.y:.6f})\n")
            f.write(f"  盘出螺线起始极角: {np.degrees(geometry.spiral_out_start_angle):.6f}°\n\n")
            
            f.write("结论:\n")
            f.write(f"通过优化圆弧半径,调头曲线长度为 {geometry.total_length:.6f} m\n")
            f.write(f"这是在保持相切约束下的最短路径。\n")
        
        print(f"\n路径信息已保存到: {info_file}")
    
    def solve(self) -> Dict[str, Any]:
        """
        求解问题4
        
        返回:
            Dict: 求解结果
        """
        print("\n" + "="*60)
        print("开始求解问题4")
        print("="*60)
        
        # 1. 设计调头路径
        uturn_path = self.design_uturn_path()
        
        # 2. 创建初始状态
        print("\n创建初始状态...")
        
        # 在t=-100时的弧长,这里使用0作为路径的极角参数
        # 注意: create_initial_state使用initial_angle作为路径参数
        initial_param = 0.0  # 路径起点
        
        initial_state = create_initial_state(
            self.dragon_config,
            uturn_path,
            initial_angle=initial_param  # 使用initial_angle参数
        )
        
        # 3. 创建碰撞检测器
        collision_detector = CollisionDetector(
            self.dragon_config,
            discretization_n=self.config_dict['simulation'].get('collision_discretization_n', 10),
            discretization_m=self.config_dict['simulation'].get('collision_discretization_m', 20)
        )
        
        # 4. 创建仿真控制器
        print("\n创建仿真控制器...")
        controller = SimulationController(
            self.dragon_config,
            uturn_path,
            initial_state,
            collision_detector
        )
        
        # 5. 运行仿真
        print(f"\n运行仿真: {self.time_range[0]}s 到 {self.time_range[1]}s...")
        
        # 累积时间序列数据
        times_list = []
        states_list = []
        
        t = self.time_range[0]  # -100s
        current_state = initial_state.clone()
        
        step_count = 0
        total_steps = int((self.time_range[1] - self.time_range[0]) / self.time_step)
        
        while t <= self.time_range[1]:
            # 记录数据
            times_list.append(t)
            states_list.append(current_state.clone())
            
            # 单步仿真
            current_state = controller.step(self.time_step, self.head_speed)
            
            t += self.time_step
            step_count += 1
            
            # 进度提示
            if step_count % 100 == 0:
                progress = step_count / total_steps * 100
                print(f"  进度: {progress:.1f}% (t={t:.1f}s)")
        
        print(f"\n仿真完成! 总步数: {step_count}")
        
        # 创建TimeSeriesData对象
        time_series = TimeSeriesData(
            times=np.array(times_list),
            states=states_list
        )
        
        # 6. 导出数据
        print("\n导出数据...")
        self.exporter.export_problem1_result(
            time_series,
            self.dragon_config,
            filename="result4.xlsx"
        )
        
        # 7. 生成可视化
        if self.config_dict['output'].get('enable_visualization', True):
            self._generate_visualizations(time_series, uturn_path)
        
        # 8. 打印特殊时刻数据
        self._print_special_times(time_series)
        
        return {
            'time_series': time_series,
            'uturn_path': uturn_path,
            'success': True
        }
    
    def _generate_visualizations(self, time_series: TimeSeriesData, path_handler):
        """生成可视化"""
        print("\n" + "="*60)
        print("生成可视化")
        print("="*60)
        
        from ..visualization.animator import ParallelAnimationGenerator
        
        # 创建动画生成器
        animator = ParallelAnimationGenerator(
            config=self.dragon_config,
            fps=self.config_dict['visualization'].get('fps', 30),
            dpi=100
        )
        
        # 高亮节点索引
        highlight_indices = self.special_indices
        
        # 生成MP4
        mp4_file = self.animations_dir / 'animation.mp4'
        print(f"\n生成MP4动画: {mp4_file}")
        animator.generate_mp4(
            data=time_series,
            output_path=mp4_file,
            highlight_indices=highlight_indices,
            frame_interval=5,
            figsize=(14, 14)
        )
        
        # 生成GIF
        gif_file = self.animations_dir / 'animation.gif'
        print(f"生成GIF动画: {gif_file}")
        animator.generate_gif(
            data=time_series,
            output_path=gif_file,
            highlight_indices=highlight_indices,
            frame_interval=10,
            figsize=(12, 12)
        )
        
        print("\n可视化生成完成!")
        print("="*60)
    
    def _print_special_times(self, time_series: TimeSeriesData):
        """打印特殊时刻的数据"""
        print("\n" + "="*60)
        print("特殊时刻数据")
        print("="*60)
        
        for t in self.special_times:
            state = time_series.get_state_at_time(t)
            if not state:
                continue
            
            print(f"\nt = {t}s:")
            print("-" * 40)
            
            # 龙头前把手
            print(f"龙头前把手: ({state.positions[0, 0]:.6f}, {state.positions[0, 1]:.6f}), v={state.velocities[0]:.6f}m/s")
            
            # 特殊节点（第1、51、101、151、201节龙身前把手）
            for idx in [1, 51, 101, 151, 201]:
                if idx < len(state.positions):
                    pos_x = state.positions[idx, 0]
                    pos_y = state.positions[idx, 1]
                    vel = state.velocities[idx]
                    print(f"第{idx}节龙身前把手: ({pos_x:.6f}, {pos_y:.6f}), v={vel:.6f}m/s")
            
            # 龙尾后把手
            pos_x = state.positions[-1, 0]
            pos_y = state.positions[-1, 1]
            vel = state.velocities[-1]
            print(f"龙尾后把手: ({pos_x:.6f}, {pos_y:.6f}), v={vel:.6f}m/s")
    
    def generate_visualization(self, time_series: TimeSeriesData):
        """
        生成可视化动画(并行渲染)
        
        time_series: 时间序列数据
        """
        from ..visualization.animator import ParallelAnimationGenerator
        
        print("\n" + "=" * 60)
        print("生成可视化动画(并行渲染)")
        print("=" * 60)
        
        # 创建并行动画生成器
        dragon_config = DragonConfig(
            head_length=3.41,
            body_length=2.20,
            bench_width=0.30,
            handle_offset=0.275,
            handle_diameter=0.055,
            num_segments=223
        )
        
        animator = ParallelAnimationGenerator(
            config=dragon_config,
            fps=10,
            dpi=100
        )
        
        # 高亮节点
        highlight_indices = [0, 1, 51, 101, 151, 201, 223]
        
        # 生成MP4
        mp4_file = self.animations_dir / 'animation.mp4'
        animator.generate_mp4(
            data=time_series,
            output_path=mp4_file,
            highlight_indices=highlight_indices,
            frame_interval=5,
            figsize=(14, 14)
        )
        
        # 生成GIF
        gif_file = self.animations_dir / 'animation.gif'
        animator.generate_gif(
            data=time_series,
            output_path=gif_file,
            highlight_indices=highlight_indices,
            frame_interval=10,
            figsize=(12, 12)
        )
        
        print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='求解问题4: S形调头路径设计')
    parser.add_argument('--no-viz', action='store_true',
                       help='不生成可视化')
    args = parser.parse_args()
    
    solver = Problem4Solver()
    result = solver.solve()
    
    if result['success']:
        print("\n" + "="*60)
        print("问题4求解成功!")
        print("="*60)
        
        # 生成可视化
        if not args.no_viz and 'time_series' in result:
            solver.generate_visualization(result['time_series'])
    else:
        print("\n问题4求解失败!")


if __name__ == "__main__":
    main()
