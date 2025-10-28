"""
问题1求解器
螺线盘入问题
"""

import numpy as np
from pathlib import Path

from ..core.models import DragonConfig, TimeSeriesData
from ..core.paths import SpiralInHandler
from ..core.geometry import CollisionDetector
from ..control.simulation import SimulationController, create_initial_state
from ..io.config import ConfigLoader
from ..io.export import DataExporter
from ..visualization.animator import AnimationGenerator
from ..visualization.plotter import Plotter


class Problem1Solver:
    """
    问题1: 螺线盘入
    
    舞龙队沿螺距为55cm的等距螺线顺时针盘入,
    龙头前把手的行进速度始终保持1m/s,
    初始时龙头位于螺线第16圈
    """
    
    def __init__(self, config_path: str = "configs/problem1.yaml"):
        """
        初始化求解器
        
        config_path: 配置文件路径
        """
        # 加载配置
        self.config_dict = ConfigLoader.load_with_inheritance(Path(config_path))
        
        # 提取龙的配置
        dragon_cfg = self.config_dict['dragon']
        self.dragon_config = DragonConfig.from_dict(dragon_cfg)
        
        # 提取问题参数
        self.spiral_pitch = self.config_dict['path']['spiral_pitch']
        self.initial_angle = self.config_dict['path']['initial_angle']
        self.head_speed = self.config_dict['motion']['head_speed']
        self.duration = self.config_dict['motion']['duration']
        self.time_step = self.config_dict['simulation']['time_step']
        
        # 输出设置 - 构建完整路径
        output_cfg = self.config_dict['output']
        self.problem_dir = Path(output_cfg['problem_dir'])
        
        # 创建各个子目录
        self.results_dir = self.problem_dir / output_cfg.get('results_subdir', 'results')
        self.data_dir = self.problem_dir / output_cfg.get('data_subdir', 'data')
        self.animations_dir = self.problem_dir / output_cfg.get('animations_subdir', 'animations')
        self.figures_dir = self.problem_dir / output_cfg.get('figures_subdir', 'figures')
        
        # 创建所有必要的目录
        for dir_path in [self.results_dir, self.data_dir, self.animations_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 获取求解器配置
        solver_config = self.config_dict['simulation'].get('solvers', {})
        
        # 创建路径处理器
        self.path_handler = SpiralInHandler(
            pitch=self.spiral_pitch,
            solver_config=solver_config
        )
        
        # 创建碰撞检测器(可选)
        if self.config_dict['simulation'].get('collision_check_enabled', False):
            self.collision_detector = CollisionDetector(
                self.dragon_config,
                discretization_n=self.config_dict['simulation'].get('collision_discretization_n', 10),
                discretization_m=self.config_dict['simulation'].get('collision_discretization_m', 20)
            )
        else:
            self.collision_detector = None
        
        # 创建初始状态
        self.initial_state = create_initial_state(
            self.dragon_config,
            self.path_handler,
            self.initial_angle
        )
        
        # 创建仿真控制器
        self.controller = SimulationController(
            self.dragon_config,
            self.path_handler,
            self.initial_state,
            self.collision_detector
        )
        
        # 创建数据导出器
        decimal_places = self.config_dict['output'].get('decimal_places', 6)
        self.exporter = DataExporter(self.results_dir, decimal_places)
        
        # 创建可视化工具
        viz_config = self.config_dict.get('visualization', {})
        if viz_config.get('enabled', True):
            self.animator = AnimationGenerator(
                self.dragon_config,
                fps=viz_config.get('animation_fps', 10),
                dpi=viz_config.get('animation_dpi', 150)
            )
            self.plotter = Plotter(self.dragon_config, self.figures_dir)
        else:
            self.animator = None
            self.plotter = None
    
    def solve(self) -> TimeSeriesData:
        """
        求解问题1
        
        返回: 时间序列数据
        """
        print("=" * 60)
        print("问题1: 螺线盘入仿真")
        print("=" * 60)
        print(f"螺距: {self.spiral_pitch} m")
        print(f"初始角度: {self.initial_angle:.6f} rad ({self.initial_angle/(2*np.pi):.2f} 圈)")
        print(f"龙头速度: {self.head_speed} m/s")
        print(f"仿真时长: {self.duration} s")
        print(f"时间步长: {self.time_step} s")
        print("=" * 60)
        
        # 运行仿真
        data = self.controller.run(
            duration=self.duration,
            dt=self.time_step,
            speed=self.head_speed,
            show_progress=True
        )
        
        print(f"\n仿真完成! 总步数: {len(data)}")
        
        return data
    
    def export_results(self, data: TimeSeriesData):
        """
        导出结果
        
        data: 仿真数据
        """
        print("\n" + "=" * 60)
        print("导出结果")
        print("=" * 60)
        
        # 导出主要结果文件
        result_file = self.config_dict['output'].get('result_file', 'result1.xlsx')
        self.exporter.export_problem1_result(data, self.dragon_config, result_file)
        
        # 导出论文数据
        paper_times = self.config_dict['output'].get('paper_times', [0, 60, 120, 180, 240, 300])
        paper_indices = self.config_dict['visualization'].get('highlight_segments', [0, 1, 51, 101, 151, 201, 223])
        paper_file = self.config_dict['output'].get('paper_data_file', 'paper_data.xlsx')
        self.exporter.export_paper_data(data, self.dragon_config, paper_times, paper_indices, paper_file)
        
        # 保存仿真数据
        if self.config_dict['output'].get('save_simulation_data', True):
            sim_file = self.config_dict['output'].get('simulation_data_file', 'simulation.pkl')
            data_file = self.data_dir / sim_file
            data.save(data_file)
            print(f"仿真数据已保存到: {data_file}")
        
        print("=" * 60)
    
    def generate_visualization(self, data: TimeSeriesData):
        """
        生成可视化动画
        
        data: 仿真数据
        """
        from ..visualization.animator import ParallelAnimationGenerator
        
        print("\n" + "=" * 60)
        print("生成可视化动画")
        print("=" * 60)
        
        # 创建动画生成器
        animator = ParallelAnimationGenerator(
            config=self.dragon_config,
            fps=self.config_dict['visualization'].get('animation_fps', 10),
            dpi=self.config_dict['visualization'].get('figure_dpi', 100)
        )
        
        # 高亮节点
        highlight_indices = self.config_dict['visualization'].get('highlight_segments', [0, 1, 51, 101, 151, 201, 223])
        
        # 生成MP4
        mp4_file = self.config_dict['visualization'].get('animation_dir', 'outputs/animations') + '/problem1.mp4'
        mp4_path = Path(mp4_file)
        animator.generate_mp4(
            data=data,
            output_path=mp4_path,
            highlight_indices=highlight_indices,
            frame_interval=10,  # 每10帧取1帧
            figsize=(12, 12)
        )
        
        # 生成GIF（更小的尺寸）
        gif_file = self.config_dict['visualization'].get('animation_dir', 'outputs/animations') + '/problem1.gif'
        gif_path = Path(gif_file)
        animator.generate_gif(
            data=data,
            output_path=gif_path,
            highlight_indices=highlight_indices,
            frame_interval=20,  # 每20帧取1帧
            figsize=(10, 10)
        )
        
        print("=" * 60)
    
    def create_visualizations(self, data: TimeSeriesData):
        """
        创建可视化(使用并行渲染)
        
        data: 仿真数据
        """
        from ..visualization.animator import ParallelAnimationGenerator
        
        print("\n" + "=" * 60)
        print("生成可视化(并行渲染)")
        print("=" * 60)
        
        viz_config = self.config_dict.get('visualization', {})
        highlight_indices = viz_config.get('highlight_segments', [0, 1, 51, 101, 151, 201, 223])
        
        # 创建并行动画生成器
        animator = ParallelAnimationGenerator(
            config=self.dragon_config,
            fps=viz_config.get('animation_fps', 10),
            dpi=viz_config.get('figure_dpi', 100)
        )
        
        # 计算路径范围（从最小到最大角度）
        all_angles = np.concatenate([s.angles for s in data.states])
        path_range = (float(all_angles.min() - 2*np.pi), float(all_angles.max() + 2*np.pi))
        
        # 生成动画
        anim_config = viz_config.get('animation', {})
        if anim_config.get('enabled', True):
            output_file = self.animations_dir / anim_config.get('output_file', 'animation.mp4')
            gif_file = self.animations_dir / anim_config.get('gif_output_file', 'animation.gif')
            frame_interval = anim_config.get('frame_interval', 10)
            gif_frame_interval = anim_config.get('gif_frame_interval', 20)
            
            animator.generate_mp4(
                data, output_file,
                highlight_indices=highlight_indices,
                frame_interval=frame_interval,
                path_handler=self.path_handler,
                path_range=path_range
            )
            
            # 同时生成GIF
            animator.generate_gif(
                data, gif_file,
                highlight_indices=highlight_indices,
                frame_interval=gif_frame_interval,
                path_handler=self.path_handler,
                path_range=path_range
            )
        
        # 生成快照
        snapshot_config = viz_config.get('snapshots', {})
        if snapshot_config.get('enabled', True):
            if self.plotter is not None:
                snapshot_times = snapshot_config.get('times', [0, 60, 120, 180, 240, 300])
                pattern = snapshot_config.get('output_pattern', 'problem1_snapshot_{time}s.png')
            
                for time in snapshot_times:
                    if time <= data.times[-1]:
                        state = data.get_state_at_time(time)
                        if state:
                            filename = pattern.format(time=int(time))
                            self.plotter.plot_snapshot(state, time, highlight_indices, filename)
        
        # 生成分析图表
        plot_configs = viz_config.get('plots', [])
        if self.plotter is not None:
            for plot_config in plot_configs:
                plot_type = plot_config.get('type')
                output_file = plot_config.get('output_file', f'{plot_type}.png')
            
                if plot_type == 'trajectory':
                    self.plotter.plot_trajectory(data, highlight_indices, output_file,
                                                path_handler=self.path_handler,
                                                path_range=path_range)
                elif plot_type == 'position_vs_time':
                    self.plotter.plot_position_vs_time(data, highlight_indices, output_file)
                elif plot_type == 'velocity_vs_time':
                    self.plotter.plot_velocity_vs_time(data, highlight_indices, output_file)
        
        print("=" * 60)
    
    def run(self):
        """运行完整流程"""
        # 求解
        data = self.solve()
        
        # 导出结果
        self.export_results(data)
        
        # 可视化
        self.create_visualizations(data)
        
        print("\n问题1求解完成!")
        
        return data


# 独立运行接口
def main():
    """独立运行问题1"""
    solver = Problem1Solver()
    solver.run()


if __name__ == "__main__":
    main()
