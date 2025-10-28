# 板凳龙仿真系统

浙闽地区传统民俗文化"板凳龙"（又称"盘龙"）的数学建模与仿真系统。

## 项目简介

本项目实现了板凳龙舞龙队的运动仿真，包括：
- 等距螺线盘入运动
- 碰撞检测与终止时刻计算
- 最小螺距优化
- 双圆弧S形调头路径设计
- 最大龙头速度优化

## 系统要求

- Python 3.10+
- 依赖包：numpy, scipy, matplotlib, pandas, openpyxl, pyyaml

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. 运行仿真

```bash
# 运行主程序（查看参数）
python main.py

# 或直接运行单个问题求解器
python -m src.solvers.problem1  # 问题1：螺线盘入
python -m src.solvers.problem2  # 问题2：碰撞终止
python -m src.solvers.problem3  # 问题3：最小螺距
python -m src.solvers.problem4  # 问题4：调头路径
python -m src.solvers.problem5  # 问题5：最大速度
```

## 项目结构

```
bench_dragon/
├── main.py                 # 主程序入口
├── README.md              # 本文档
├── requirements.txt       # 依赖列表
├── configs/               # 配置文件
│   ├── base.yaml         # 基础配置
│   ├── problem1.yaml     # 问题1配置
│   ├── problem2.yaml     # 问题2配置
│   ├── problem3.yaml     # 问题3配置
│   ├── problem4.yaml     # 问题4配置
│   └── problem5.yaml     # 问题5配置
├── src/                   # 源代码
│   ├── core/             # 核心模块
│   │   ├── models.py     # 数据模型
│   │   ├── paths.py      # 路径处理
│   │   └── collision.py  # 碰撞检测
│   ├── control/          # 控制模块
│   │   └── simulation.py # 运动仿真
│   ├── solvers/          # 问题求解器
│   │   ├── problem1.py   # 螺线盘入
│   │   ├── problem2.py   # 碰撞终止
│   │   ├── problem3.py   # 最小螺距
│   │   ├── problem4.py   # 调头路径
│   │   └── problem5.py   # 最大速度
│   ├── visualization/    # 可视化模块
│   │   ├── animator.py   # 动画生成
│   │   ├── plotter.py    # 图表绘制
│   │   └── renderer.py   # 渲染器
│   └── utils/            # 工具模块
│       ├── config.py     # 配置管理
│       └── export.py     # 数据导出
└── outputs/              # 输出目录
    ├── problem1/         # 问题1输出
    │   ├── results/      # Excel结果文件
    │   ├── data/         # 仿真数据
    │   ├── animations/   # 动画文件
    │   └── figures/      # 静态图表
    ├── problem2/         # 问题2输出
    ├── problem3/         # 问题3输出
    ├── problem4/         # 问题4输出
    └── problem5/         # 问题5输出
```

## 问题说明

### 问题1：螺线盘入仿真
- 舞龙队沿螺距55cm的等距螺线顺时针盘入
- 龙头速度1m/s，从第16圈开始
- 仿真300秒，每秒输出位置和速度
- 输出：`result1.xlsx`（完整数据）、`paper_data.xlsx`（论文数据）

### 问题2：盘入终止时刻
- 继续问题1的运动，启用碰撞检测
- 确定板凳之间不发生碰撞的最晚时刻
- 输出：`result2.xlsx`（终止时刻的位置和速度）

### 问题3：最小螺距搜索
- 调头空间：直径9m的圆形区域
- 搜索最小螺距，使龙头能盘入到调头空间边界
- 输出：`result3.txt`（最小螺距值）

### 问题4：双圆弧调头路径
- 螺距1.7m，螺线盘入+双圆弧S形调头+螺线盘出
- 前段圆弧半径是后段的2倍，各部分相切
- 时间范围：-100s到100s（0时刻为调头开始）
- 输出：`result4.xlsx`（完整数据）、动画、图表

### 问题5：最大龙头速度
- 使用问题4的路径
- 优化龙头速度，使所有把手速度≤2m/s
- 输出：`result5.txt`（最大速度值）

## 配置说明

配置文件采用YAML格式，支持继承。修改 `configs/` 目录下的文件可调整：
- 仿真参数（时间步长、精度等）
- 物理参数（龙的尺寸、速度等）
- 输出设置（文件路径、格式等）
- 可视化选项（动画、图表等）

## 输出文件

### Excel文件
- `result1.xlsx`：问题1完整结果（位置和速度工作表）
- `result2.xlsx`：问题2终止时刻结果
- `result4.xlsx`：问题4调头过程结果
- `paper_data.xlsx`：论文所需的特殊时刻数据

### 文本文件
- `result3.txt`：问题3最小螺距
- `result5.txt`：问题5最大速度

### 可视化文件
- `animation.mp4`：完整动画（MP4格式）
- `animation.gif`：动画（GIF格式）
- `trajectory.png`：轨迹图
- `position.png`：位置随时间变化图
- `velocity.png`：速度随时间变化图

## 技术特点

### 高精度数值计算
- ODE求解器：相对误差1e-12，绝对误差1e-15
- 使用Brent方法和二分法确保解的唯一性和稳定性
- 智能搜索方向和范围估计

### 完整的物理约束
- 刚性铰链连接：保持相邻把手间距恒定
- 路径约束：所有把手中心位于指定路径上
- 碰撞检测：板凳离散化检测（长度方向10点，宽度方向20点）

### 高效并行渲染
- 使用多进程池并行生成动画帧
- 自适应帧间隔减少渲染时间
- 支持路径可视化（螺线、圆弧等）

## 开发说明

- 核心算法：基于路径参数的运动学建模
- 路径求解：前向计算（龙头→龙尾），避免累积误差
- 优化方法：二分搜索、黄金分割搜索
- 数据结构：使用 `@dataclass` 确保类型安全

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请提交 Issue。
