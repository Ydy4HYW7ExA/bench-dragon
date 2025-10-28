"""
配置加载器
支持YAML配置文件的加载、合并和验证
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy


class ConfigLoader:
    """配置加载器"""
    
    @staticmethod
    def load_yaml(filepath: Path) -> Dict[str, Any]:
        """加载YAML配置文件"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config or {}
    
    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个配置字典
        override中的值会覆盖base中的值
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    @staticmethod
    def load_with_inheritance(filepath: Path, config_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        加载配置文件,支持继承
        如果配置文件中有 'inherit_from' 字段,会先加载父配置,然后合并
        """
        filepath = Path(filepath)
        if config_dir is None:
            config_dir = filepath.parent
        
        config = ConfigLoader.load_yaml(filepath)
        
        # 检查是否需要继承
        if 'inherit_from' in config:
            parent_path = config_dir / config['inherit_from']
            parent_config = ConfigLoader.load_with_inheritance(parent_path, config_dir)
            
            # 移除inherit_from字段
            config.pop('inherit_from')
            
            # 合并配置
            config = ConfigLoader.merge_configs(parent_config, config)
        
        return config
    
    @staticmethod
    def get_nested(config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        获取嵌套配置值
        path: 用点分隔的路径, 如 "dragon.head_length"
        """
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @staticmethod
    def set_nested(config: Dict[str, Any], path: str, value: Any):
        """
        设置嵌套配置值
        path: 用点分隔的路径, 如 "dragon.head_length"
        """
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
