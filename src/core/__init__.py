"""
Core module for MVP Pipeline
包含Registry模式和配置管理系统
"""

from .registry import Registry, MODEL_REGISTRY, DATASET_REGISTRY
from .config import Config, load_config

__all__ = [
    'Registry',
    'MODEL_REGISTRY', 
    'DATASET_REGISTRY',
    'Config',
    'load_config'
]
