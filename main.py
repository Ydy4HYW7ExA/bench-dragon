"""
板凳龙仿真系统 - 主程序入口
"""

import argparse
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.solvers.problem1 import Problem1Solver
from src.solvers.problem2 import Problem2Solver
from src.solvers.problem3 import Problem3Solver
from src.solvers.problem4 import Problem4Solver
from src.solvers.problem5 import Problem5Solver


def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='板凳龙仿真系统')
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # solve命令
    solve_parser = subparsers.add_parser('solve', help='求解问题')
    solve_parser.add_argument('--problem', type=int, choices=[1, 2, 3, 4, 5],
                              required=True, help='问题编号')
    solve_parser.add_argument('--config', type=str, help='配置文件路径(可选)')
    
    # all命令 - 求解所有问题
    all_parser = subparsers.add_parser('all', help='求解所有问题')
    all_parser.add_argument('--skip', type=int, nargs='*', help='跳过的问题编号')
    
    args = parser.parse_args()
    
    if args.command == 'solve':
        solve_problem(args)
    elif args.command == 'all':
        solve_all_problems(args)
    else:
        parser.print_help()


def solve_problem(args):
    """求解单个问题"""
    print(f"\n开始求解问题{args.problem}...")
    
    if args.problem == 1:
        config_path = args.config or "configs/problem1.yaml"
        solver = Problem1Solver(config_path)
        solver.run()
    
    elif args.problem == 2:
        config_path = args.config or "configs/problem2.yaml"
        solver = Problem2Solver(config_path)
        solver.run()
    
    elif args.problem == 3:
        config_path = args.config or "configs/problem3.yaml"
        solver = Problem3Solver(config_path)
        solver.run()
    
    elif args.problem == 4:
        config_path = args.config or "configs/problem4.yaml"
        solver = Problem4Solver(config_path)
        solver.solve()
    
    elif args.problem == 5:
        config_path = args.config or "configs/problem5.yaml"
        solver = Problem5Solver(config_path)
        solver.solve()


def solve_all_problems(args):
    """求解所有问题"""
    skip = set(args.skip) if args.skip else set()
    
    print("\n" + "=" * 80)
    print("板凳龙仿真系统 - 求解所有问题")
    print("=" * 80)
    
    problems = [
        (1, "螺线盘入", Problem1Solver, "configs/problem1.yaml"),
        (2, "碰撞检测", Problem2Solver, "configs/problem2.yaml"),
        (3, "螺距优化", Problem3Solver, "configs/problem3.yaml"),
        (4, "调头路径", Problem4Solver, "configs/problem4.yaml"),
        (5, "速度优化", Problem5Solver, "configs/problem5.yaml"),
    ]
    
    for problem_num, name, solver_class, config_path in problems:
        if problem_num in skip:
            print(f"\n跳过问题{problem_num}: {name}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"问题{problem_num}: {name}")
        print(f"{'=' * 80}")
        
        try:
            solver = solver_class(config_path)
            # 问题4和问题5使用solve()方法，其他使用run()方法
            if problem_num in [4, 5]:
                solver.solve()
            else:
                solver.run()
            print(f"\n✓ 问题{problem_num}求解完成!")
        except Exception as e:
            print(f"\n✗ 问题{problem_num}求解失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("所有问题求解完成!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

