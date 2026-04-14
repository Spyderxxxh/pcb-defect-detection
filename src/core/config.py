"""
配置管理系统
支持YAML配置驱动，无需修改代码即可调整参数
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    配置类
    支持字典式访问和属性访问
    
    Example:
        >>> cfg = Config({'model': {'type': 'YOLOv8', 'backbone': 'resnet50'}})
        >>> cfg.model.type
        'YOLOv8'
        >>> cfg['model']['backbone']
        'resnet50'
    """
    
    def __init__(self, cfg_dict: Dict[str, Any] = None, filename: Optional[str] = None):
        if cfg_dict is None:
            cfg_dict = {}
        self._cfg_dict = cfg_dict
        self._filename = filename
        
        # 递归转换为Config对象
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                cfg_dict[key] = Config(value)
    
    def __getitem__(self, key: str) -> Any:
        return self._cfg_dict[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._cfg_dict[key] = value
    
    def __getattr__(self, key: str) -> Any:
        try:
            return self._cfg_dict[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._cfg_dict[key] = value
    
    def __contains__(self, key: str) -> bool:
        return key in self._cfg_dict
    
    def __repr__(self) -> str:
        return f"Config({self._cfg_dict})"
    
    def get(self, key: str, default: Any = None) -> Any:
        """安全获取配置项"""
        return self._cfg_dict.get(key, default)
    
    def update(self, other: Dict[str, Any]) -> None:
        """更新配置"""
        self._cfg_dict.update(other)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典"""
        result = {}
        for key, value in self._cfg_dict.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def dump(self, filepath: Optional[str] = None) -> None:
        """保存配置到YAML文件"""
        filepath = filepath or self._filename
        if filepath is None:
            raise ValueError("No filename specified")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def copy(self) -> 'Config':
        """深拷贝配置"""
        import copy
        return Config(copy.deepcopy(self.to_dict()))
    
    @property
    def filename(self) -> Optional[str]:
        return self._filename


def load_config(filepath: str) -> Config:
    """
    从YAML文件加载配置
    
    Args:
        filepath: YAML文件路径
        
    Returns:
        Config对象
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    
    return Config(cfg_dict, str(filepath))


def merge_configs(base_cfg: Config, override_cfg: Dict[str, Any]) -> Config:
    """
    合并配置，override_cfg会覆盖base_cfg中的同名配置
    
    Args:
        base_cfg: 基础配置
        override_cfg: 覆盖配置
        
    Returns:
        合并后的新配置
    """
    merged = base_cfg.copy()
    
    def recursive_merge(base: Dict, override: Dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                recursive_merge(base[key], value)
            else:
                base[key] = value
    
    recursive_merge(merged._cfg_dict, override_cfg)
    return merged


# 配置模板
DEFAULT_CONFIG = {
    'project': {
        'name': 'mvp_pipeline',
        'version': '1.0.0'
    },
    'model': {
        'type': 'YOLOv8NonLocal',
        'backbone': 'resnet50',
        'num_classes': 5,
        'pretrained': True
    },
    'dataset': {
        'type': 'PCBDefectDataset',
        'data_root': './data/pcb_defect',
        'img_size': 640,
        'batch_size': 16
    },
    'training': {
        'epochs': 100,
        'lr': 0.001,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'device': 'cuda'
    },
    'agent': {
        'enabled': True,
        'mode': 'balanced',  # high_precision, balanced, high_speed
        'ab_test': True
    },
    'deployment': {
        'tensorrt': False,
        'onnx': False,
        'quantization': False
    }
}
