"""
问题5求解器 - 速度优化

使用问题4的完整路径(螺线盘入 + 双圆弧调头 + 螺线盘出),
确定龙头的最大速度,使得所有把手的速度均不超过2m/s。

关键点:
1. 使用问题4设计的调头路径
2. 计算所有把手(包括板凳中心把手和板凳左右两侧点)的速度
3. 约束: max(所有点的速度) ≤ 2.0 m/s
4. 二分搜索最优龙头速度
"""

from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple, List

from ..core.models import DragonConfig, SimulationState
from ..control.simulation import SimulationController, create_initial_state
from ..core.uturn_geometry import optimize_uturn_radius
from ..core.uturn_path import DoubleArcUTurnHandler
from ..core.geometry import CollisionDetector
from ..io.config import ConfigLoader


class Problem5Solver:
    """
    问题5求解器 - 速度优化
    
    目标: 最大化龙头速度
    约束: 所有把手及板凳边缘点速度 ≤ 2.0 m/s
    """
    
    def __init__(self, config_path: str = "configs/problem5.yaml"):
        """
        初始化问题5求解器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config_dict = ConfigLoader.load_with_inheritance(Path(config_path))
        
        # 龙的配置
        dragon_cfg = self.config_dict['dragon']
        self.dragon_config = DragonConfig.from_dict(dragon_cfg)
        
        # 路径参数(与问题4相同)
        path_cfg = self.config_dict['path']
        self.pitch = path_cfg.get('pitch', 1.7)
        self.turnaround_diameter = path_cfg.get('turnaround_diameter', 9.0)
        self.turnaround_radius = self.turnaround_diameter / 2
        
        # 优化参数
        opt_cfg = self.config_dict.get('optimization', {})
        self.min_velocity = opt_cfg.get('min_velocity', 0.5)
        self.max_velocity = opt_cfg.get('max_velocity', 3.0)
        self.tolerance = opt_cfg.get('tolerance', 0.001)
        self.max_handle_velocity = opt_cfg.get('max_handle_velocity', 2.0)
        
        # 测试参数
        self.test_duration = opt_cfg.get('test_duration', 150.0)
        self.time_step = self.config_dict['simulation'].get('time_step', 0.1)
        
        # 输出设置
        output_cfg = self.config_dict['output']
        self.problem_dir = Path(output_cfg['problem_dir'])
        self.results_dir = self.problem_dir / output_cfg.get('results_subdir', 'results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("问题5求解器初始化")
        print("="*60)
        print(f"路径: 问题4调头路径(螺距{self.pitch}m, 调头空间直径{self.turnaround_diameter}m)")
        print(f"速度约束: ≤ {self.max_handle_velocity}m/s")
        print(f"优化范围: [{self.min_velocity}, {self.max_velocity}]m/s")
        print(f"精度: {self.tolerance}m/s")
    
    def create_uturn_path(self) -> DoubleArcUTurnHandler:
        """
        创建调头路径(与问题4相同)
        
        返回:
            DoubleArcUTurnHandler: 调头路径处理器
        """
        # 优化圆弧半径
        best_r2, geometry = optimize_uturn_radius(
            pitch=self.pitch,
            turnaround_radius=self.turnaround_radius,
            min_r2=0.5,
            max_r2=3.0,
            num_samples=30
        )
        
        # 螺线盘入段长度(足够长以覆盖测试区间)
        spiral_in_length = self.test_duration * self.max_velocity  # 保守估计
        
        # 创建路径处理器
        uturn_path = DoubleArcUTurnHandler(
            geometry=geometry,
            pitch=self.pitch,
            spiral_in_length=spiral_in_length,
            clockwise_in=True
        )
        
        return uturn_path
    
    def compute_all_velocities(self, state: SimulationState, dt: float,
                               next_state: SimulationState) -> List[float]:
        """
        计算所有相关点的速度(包括把手和板凳边缘)
        
        参数:
            state: 当前状态
            dt: 时间步长
            next_state: 下一时刻状态
        
        返回:
            List[float]: 所有点的速度列表
        """
        velocities = []
        
        # 对每个板凳
        for i in range(self.dragon_config.num_segments):
            segment = state.get_segment(i, self.dragon_config)
            next_segment = next_state.get_segment(i, self.dragon_config)
            
            # 1. 前把手速度(已在state.velocities中)
            velocities.append(float(state.velocities[i]))
            
            # 2. 板凳中心速度
            center = segment.get_center()
            next_center = next_segment.get_center()
            
            vx_center = (next_center.x - center.x) / dt
            vy_center = (next_center.y - center.y) / dt
            v_center = np.sqrt(vx_center**2 + vy_center**2)
            velocities.append(float(v_center))
            
            # 3. 板凳角速度
            angle = segment.get_direction()
            next_angle = next_segment.get_direction()
            
            # 处理角度跳变
            angle_diff = next_angle - angle
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            omega = angle_diff / dt
            
            # 4. 板凳四个角点的速度
            # 板凳宽度和长度
            width = self.dragon_config.bench_width
            
            # 根据节点类型获取板凳长度
            if i == 0:
                # 龙头
                length = self.dragon_config.head_length
            else:
                # 龙身和龙尾
                length = self.dragon_config.body_length
            
            # 板凳方向和垂直方向
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            # 四个角点相对于中心的偏移
            # 前左, 前右, 后左, 后右
            half_length = length / 2
            half_width = width / 2
            
            corners = [
                (half_length, half_width),   # 前左
                (half_length, -half_width),  # 前右
                (-half_length, half_width),  # 后左
                (-half_length, -half_width), # 后右
            ]
            
            for dx_local, dy_local in corners:
                # 转到全局坐标系
                dx = dx_local * cos_a - dy_local * sin_a
                dy = dx_local * sin_a + dy_local * cos_a
                
                # 角点速度 = 中心速度 + 旋转速度
                # v = v_center + omega × r
                vx = vx_center - omega * dy
                vy = vy_center + omega * dx
                
                v_corner = np.sqrt(vx**2 + vy**2)
                velocities.append(float(v_corner))
        
        # 龙尾后把手
        velocities.append(float(state.velocities[-1]))
        
        return velocities
    
    def check_velocity_constraint(self, v_head: float) -> Tuple[bool, float, Dict[str, Any]]:
        """
        检查给定龙头速度是否满足约束
        
        参数:
            v_head: 龙头速度(m/s)
        
        返回:
            (是否满足, 最大速度, 详细信息)
        """
        # 创建路径
        uturn_path = self.create_uturn_path()
        
        # 创建初始状态
        initial_state = create_initial_state(
            self.dragon_config,
            uturn_path,
            initial_angle=0.0  # 路径起点
        )
        
        # 碰撞检测器
        collision_detector = CollisionDetector(
            self.dragon_config,
            discretization_n=10,
            discretization_m=20
        )
        
        # 仿真控制器
        controller = SimulationController(
            self.dragon_config,
            uturn_path,
            initial_state,
            collision_detector
        )
        
        # 运行仿真并监控速度
        max_velocity = 0.0
        max_time = 0.0
        max_segment_id = 0
        max_point_type = ""
        
        t = 0.0
        prev_state = None
        current_state = initial_state.clone()
        
        check_interval = 5  # 每5步检查一次
        step_count = 0
        
        while t < self.test_duration:
            # 单步仿真
            current_state = controller.step(self.time_step, v_head)
            
            # 定期检查速度
            if step_count % check_interval == 0 and prev_state is not None:
                # 计算所有点的速度
                dt_check = self.time_step * check_interval
                all_velocities = self.compute_all_velocities(
                    prev_state, dt_check, current_state
                )
                
                # 找最大值
                for vel in all_velocities:
                    if vel > max_velocity:
                        max_velocity = vel
                        max_time = t
                
                # 早停: 如果已超出限制10%
                if max_velocity > self.max_handle_velocity * 1.1:
                    return False, max_velocity, {
                        'max_velocity': max_velocity,
                        'max_time': max_time,
                        'early_stop': True
                    }
            
            if step_count % check_interval == 0:
                prev_state = current_state.clone()
            
            t += self.time_step
            step_count += 1
        
        # 判断是否满足约束
        satisfied = max_velocity <= self.max_handle_velocity
        
        details = {
            'max_velocity': max_velocity,
            'max_time': max_time,
            'early_stop': False
        }
        
        return satisfied, max_velocity, details
    
    def binary_search(self) -> Tuple[float, Dict[str, Any]]:
        """
        二分搜索最大龙头速度
        
        返回:
            (最优速度, 详细信息)
        """
        print("\n开始二分搜索最优龙头速度...")
        
        left = self.min_velocity
        right = self.max_velocity
        best_velocity = left
        
        iteration = 0
        max_iterations = 30
        
        while right - left > self.tolerance and iteration < max_iterations:
            mid = (left + right) / 2
            iteration += 1
            
            print(f"\n迭代 {iteration}: 测试 v_head = {mid:.4f} m/s...")
            
            satisfied, max_vel, details = self.check_velocity_constraint(mid)
            
            print(f"  最大速度: {max_vel:.4f} m/s")
            print(f"  是否满足约束: {satisfied}")
            
            if satisfied:
                # 满足约束,尝试更大的速度
                best_velocity = mid
                left = mid
                print(f"  ✓ 满足约束,尝试更大速度")
            else:
                # 不满足约束,降低速度
                right = mid
                print(f"  ✗ 不满足约束,降低速度")
        
        print(f"\n二分搜索完成! 最优龙头速度: {best_velocity:.6f} m/s")
        
        # 最后验证一次
        print(f"\n最终验证...")
        satisfied, max_vel, details = self.check_velocity_constraint(best_velocity)
        
        return best_velocity, {
            'optimal_velocity': best_velocity,
            'max_handle_velocity': max_vel,
            'constraint': self.max_handle_velocity,
            'satisfied': satisfied,
            'iterations': iteration,
            **details
        }
    
    def solve(self) -> Dict[str, Any]:
        """
        求解问题5
        
        返回:
            Dict: 求解结果
        """
        print("\n" + "="*60)
        print("开始求解问题5")
        print("="*60)
        
        # 二分搜索
        optimal_velocity, details = self.binary_search()
        
        # 保存结果
        self._save_result(optimal_velocity, details)
        
        # 打印结果
        self._print_result(optimal_velocity, details)
        
        return {
            'optimal_velocity': optimal_velocity,
            'details': details,
            'success': True
        }
    
    def _save_result(self, optimal_velocity: float, details: Dict[str, Any]):
        """保存结果到txt文件"""
        result_file = self.results_dir / "result5.txt"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            # 题目要求的关键信息（第一行）
            f.write(f"最大龙头速度为：{optimal_velocity:.6f} m/s\n")
            f.write("\n" + "=" * 60 + "\n")
            
            # 详细信息
            f.write("问题5 - 速度优化详细结果\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("优化结果:\n")
            f.write(f"  最优龙头速度: {optimal_velocity:.6f} m/s\n")
            f.write(f"  速度约束: 所有把手速度 ≤ {self.max_handle_velocity:.6f} m/s\n")
            f.write(f"  实际最大把手速度: {details['max_handle_velocity']:.6f} m/s\n")
            f.write(f"  约束满足情况: {'满足' if details['satisfied'] else '不满足'}\n")
            f.write(f"  余量: {self.max_handle_velocity - details['max_handle_velocity']:.6f} m/s\n\n")
            
            f.write("优化参数:\n")
            f.write(f"  搜索区间: [{self.min_velocity:.6f}, {self.max_velocity:.6f}] m/s\n")
            f.write(f"  精度要求: {self.tolerance:.6f} m/s\n")
            f.write(f"  搜索迭代次数: {details['iterations']}\n")
            f.write(f"  测试时长: {self.test_duration:.1f} s\n")
            f.write(f"  时间步长: {self.time_step:.4f} s\n\n")
            
            f.write("路径参数（与问题4相同）:\n")
            f.write(f"  螺线螺距: {self.pitch:.6f} m\n")
            f.write(f"  调头空间直径: {self.turnaround_diameter:.6f} m\n")
            f.write(f"  调头空间半径: {self.turnaround_radius:.6f} m\n\n")
            
            f.write("龙的配置:\n")
            f.write(f"  板凳总数: {self.dragon_config.num_segments}\n")
            f.write(f"  龙头长度: {self.dragon_config.head_length:.3f} m\n")
            f.write(f"  龙身长度: {self.dragon_config.body_length:.3f} m\n")
            f.write(f"  板凳宽度: {self.dragon_config.bench_width:.3f} m\n")
            f.write(f"  把手间距(龙头): {self.dragon_config.head_handle_distance:.3f} m\n")
            f.write(f"  把手间距(龙身): {self.dragon_config.body_handle_distance:.3f} m\n\n")
            
            f.write("速度计算说明:\n")
            f.write("  本优化考虑了以下所有点的速度:\n")
            f.write("  1. 所有把手中心点的速度\n")
            f.write("  2. 所有板凳中心点的速度\n")
            f.write("  3. 所有板凳四个角点的速度（考虑旋转效应）\n")
            f.write("  角点速度计算公式: v = v_center + ω × r\n")
            f.write("  其中ω为板凳角速度，r为角点相对于中心的位置向量\n\n")
            
            f.write("物理意义:\n")
            f.write(f"  在最大龙头速度{optimal_velocity:.6f}m/s下，舞龙队沿问题4设定的\n")
            f.write(f"  调头路径行进时，所有把手及板凳边缘点的速度均不超过\n")
            f.write(f"  {self.max_handle_velocity:.6f}m/s，确保舞龙队的安全性和可控性。\n")
            f.write(f"  这是在满足速度约束前提下的最大可行速度。\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"\n结果已保存到: {result_file}")
    
    def _print_result(self, optimal_velocity: float, details: Dict[str, Any]):
        """打印结果"""
        print("\n" + "="*60)
        print("问题5求解结果")
        print("="*60)
        print(f"最大龙头速度: {optimal_velocity:.6f} m/s")
        print(f"此时最大把手速度: {details['max_handle_velocity']:.6f} m/s")
        print(f"速度约束: ≤ {self.max_handle_velocity} m/s")
        print(f"是否满足约束: {details['satisfied']}")
        print("="*60)


def main():
    """主函数"""
    solver = Problem5Solver()
    result = solver.solve()
    
    if result['success']:
        print("\n问题5求解成功!")
    else:
        print("\n问题5求解失败!")


if __name__ == "__main__":
    main()
