"""
数据导出模块
支持导出到Excel等格式
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

from ..core.models import TimeSeriesData, DragonConfig


class DataExporter:
    """数据导出器"""
    
    def __init__(self, output_dir: Path, decimal_places: int = 6):
        """
        初始化导出器
        
        output_dir: 输出目录
        decimal_places: 小数位数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.decimal_places = decimal_places
    
    def export_problem1_result(self, data: TimeSeriesData, config: DragonConfig,
                               filename: str = "result1.xlsx"):
        """
        导出问题1的结果
        
        格式:
        - 位置工作表: 每个把手的x和y坐标随时间变化
        - 速度工作表: 每个把手的速度随时间变化
        """
        filepath = self.output_dir / filename
        
        # 准备数据
        times = data.times
        num_times = len(times)
        num_handles = config.num_handles
        
        # 创建位置数据框
        position_data = {}
        position_data['时间(s)'] = times
        
        # 龙头
        position_data['龙头x (m)'] = [round(data.states[i].positions[0, 0], self.decimal_places) 
                                      for i in range(num_times)]
        position_data['龙头y (m)'] = [round(data.states[i].positions[0, 1], self.decimal_places) 
                                      for i in range(num_times)]
        
        # 龙身 (1-221节)
        for j in range(1, config.num_segments):
            position_data[f'第{j}节龙身x (m)'] = [round(data.states[i].positions[j, 0], self.decimal_places) 
                                                 for i in range(num_times)]
            position_data[f'第{j}节龙身y (m)'] = [round(data.states[i].positions[j, 1], self.decimal_places) 
                                                 for i in range(num_times)]
        
        # 龙尾
        position_data['龙尾x (m)'] = [round(data.states[i].positions[config.num_segments-1, 0], self.decimal_places) 
                                     for i in range(num_times)]
        position_data['龙尾y (m)'] = [round(data.states[i].positions[config.num_segments-1, 1], self.decimal_places) 
                                     for i in range(num_times)]
        
        # 龙尾后把手
        position_data['龙尾(后)x (m)'] = [round(data.states[i].positions[config.num_segments, 0], self.decimal_places) 
                                        for i in range(num_times)]
        position_data['龙尾(后)y (m)'] = [round(data.states[i].positions[config.num_segments, 1], self.decimal_places) 
                                        for i in range(num_times)]
        
        # 创建速度数据框
        velocity_data = {}
        velocity_data['时间(s)'] = times
        
        velocity_data['龙头 (m/s)'] = [round(data.states[i].velocities[0], self.decimal_places) 
                                      for i in range(num_times)]
        
        for j in range(1, config.num_segments):
            velocity_data[f'第{j}节龙身 (m/s)'] = [round(data.states[i].velocities[j], self.decimal_places) 
                                                  for i in range(num_times)]
        
        velocity_data['龙尾 (m/s)'] = [round(data.states[i].velocities[config.num_segments-1], self.decimal_places) 
                                      for i in range(num_times)]
        velocity_data['龙尾(后) (m/s)'] = [round(data.states[i].velocities[config.num_segments], self.decimal_places) 
                                         for i in range(num_times)]
        
        # 转置数据(时间作为列)
        position_df = pd.DataFrame(position_data).set_index('时间(s)').T
        velocity_df = pd.DataFrame(velocity_data).set_index('时间(s)').T
        
        # 写入Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            position_df.to_excel(writer, sheet_name='位置')
            velocity_df.to_excel(writer, sheet_name='速度')
        
        print(f"结果已导出到: {filepath}")
    
    def export_problem2_result(self, final_state, config: DragonConfig,
                               filename: str = "result2.xlsx"):
        """
        导出问题2的结果
        
        格式: 单个工作表,包含最终时刻所有把手的位置和速度
        """
        filepath = self.output_dir / filename
        
        # 准备数据
        rows = []
        
        # 龙头
        rows.append({
            '节点': '龙头',
            '横坐标x (m)': round(final_state.positions[0, 0], self.decimal_places),
            '纵坐标y (m)': round(final_state.positions[0, 1], self.decimal_places),
            '速度 (m/s)': round(final_state.velocities[0], self.decimal_places)
        })
        
        # 龙身
        for i in range(1, config.num_segments):
            rows.append({
                '节点': f'第{i}节龙身',
                '横坐标x (m)': round(final_state.positions[i, 0], self.decimal_places),
                '纵坐标y (m)': round(final_state.positions[i, 1], self.decimal_places),
                '速度 (m/s)': round(final_state.velocities[i], self.decimal_places)
            })
        
        # 龙尾
        rows.append({
            '节点': '龙尾',
            '横坐标x (m)': round(final_state.positions[config.num_segments-1, 0], self.decimal_places),
            '纵坐标y (m)': round(final_state.positions[config.num_segments-1, 1], self.decimal_places),
            '速度 (m/s)': round(final_state.velocities[config.num_segments-1], self.decimal_places)
        })
        
        # 龙尾后把手
        rows.append({
            '节点': '龙尾(后)',
            '横坐标x (m)': round(final_state.positions[config.num_segments, 0], self.decimal_places),
            '纵坐标y (m)': round(final_state.positions[config.num_segments, 1], self.decimal_places),
            '速度 (m/s)': round(final_state.velocities[config.num_segments], self.decimal_places)
        })
        
        df = pd.DataFrame(rows)
        df.to_excel(filepath, index=False)
        
        print(f"结果已导出到: {filepath}")
    
    def export_paper_data(self, data: TimeSeriesData, config: DragonConfig,
                          times: List[float], indices: List[int],
                          filename: str = "paper_data.xlsx"):
        """
        导出论文中需要的特定时间点和节点的数据
        
        times: 需要的时间点
        indices: 需要的节点索引
        """
        filepath = self.output_dir / filename
        
        rows = []
        for time in times:
            # 找到最接近的时间点
            state = data.get_state_at_time(time)
            
            if state is None:
                continue
            
            for idx in indices:
                if idx == 0:
                    node_name = "龙头前把手"
                elif idx == config.num_segments:
                    node_name = "龙尾后把手"
                else:
                    node_name = f"第{idx}节龙身前把手"
                
                rows.append({
                    '时间(s)': time,
                    '节点': node_name,
                    'x (m)': round(state.positions[idx, 0], self.decimal_places),
                    'y (m)': round(state.positions[idx, 1], self.decimal_places),
                    '速度 (m/s)': round(state.velocities[idx], self.decimal_places)
                })
        
        df = pd.DataFrame(rows)
        df.to_excel(filepath, index=False)
        
        print(f"论文数据已导出到: {filepath}")
