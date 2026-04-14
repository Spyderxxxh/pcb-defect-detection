"""
Registry Pattern Implementation
实现模型的统一注册与管理，支持30分钟快速接入新模型
"""

import inspect
from typing import Dict, Any, Callable, Optional
from functools import wraps


class Registry:
    """
    通用注册器类
    支持模型、数据集、优化器等各种组件的注册
    
    Example:
        >>> MODEL_REGISTRY = Registry('model')
        >>> @MODEL_REGISTRY.register()
        ... class YOLOv8NonLocal:
        ...     pass
        >>> model_class = MODEL_REGISTRY.get('YOLOv8NonLocal')
    """
    
    def __init__(self, name: str):
        self.name = name
        self._module_dict: Dict[str, Any] = {}
    
    def __len__(self) -> int:
        return len(self._module_dict)
    
    def __contains__(self, key: str) -> bool:
        return key in self._module_dict
    
    def __repr__(self) -> str:
        return f"Registry(name={self.name}, items={list(self._module_dict.keys())})"
    
    @property
    def module_dict(self) -> Dict[str, Any]:
        """获取所有注册的模块"""
        return self._module_dict
    
    def get(self, key: str) -> Any:
        """
        根据key获取注册的模块
        
        Args:
            key: 模块名称
            
        Returns:
            注册的模块类或函数
            
        Raises:
            KeyError: 如果key不存在
        """
        if key not in self._module_dict:
            raise KeyError(f"{key} not found in {self.name} registry. "
                          f"Available: {list(self._module_dict.keys())}")
        return self._module_dict[key]
    
    def register(self, name: Optional[str] = None, force: bool = False) -> Callable:
        """
        注册装饰器
        
        Args:
            name: 注册名称，默认为类/函数名
            force: 是否强制覆盖已存在的注册
            
        Returns:
            装饰器函数
        """
        def decorator(obj: Any) -> Any:
            module_name = name if name is not None else obj.__name__
            
            if module_name in self._module_dict and not force:
                raise KeyError(f"{module_name} already registered in {self.name}. "
                              f"Use force=True to override.")
            
            self._module_dict[module_name] = obj
            return obj
        
        return decorator
    
    def unregister(self, name: str) -> None:
        """注销模块"""
        if name in self._module_dict:
            del self._module_dict[name]
    
    def list_modules(self) -> list:
        """列出所有已注册的模块名称"""
        return list(self._module_dict.keys())


# 全局注册器实例
MODEL_REGISTRY = Registry('model')
DATASET_REGISTRY = Registry('dataset')
OPTIMIZER_REGISTRY = Registry('optimizer')
SCHEDULER_REGISTRY = Registry('scheduler')
TRANSFORM_REGISTRY = Registry('transform')
AGENT_REGISTRY = Registry('agent')


def build_from_config(cfg: Dict[str, Any], registry: Registry, 
                      default_args: Optional[Dict] = None) -> Any:
    """
    从配置字典构建对象
    
    Args:
        cfg: 配置字典，必须包含 'type' 字段
        registry: 注册器实例
        default_args: 默认参数
        
    Returns:
        构建的对象实例
        
    Example:
        >>> cfg = dict(type='YOLOv8NonLocal', num_classes=5, backbone='resnet50')
        >>> model = build_from_config(cfg, MODEL_REGISTRY)
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be dict, got {type(cfg)}")
    
    if 'type' not in cfg:
        raise KeyError(f"cfg must contain 'type' field, got {cfg.keys()}")
    
    cfg = cfg.copy()
    obj_type = cfg.pop('type')
    
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    else:
        obj_cls = obj_type
    
    args = default_args or {}
    args.update(cfg)
    
    return obj_cls(**args)


class ModelBuilder:
    """
    模型构建器
    简化从配置构建模型的过程
    """
    
    @staticmethod
    def build(cfg: Dict[str, Any], **kwargs) -> Any:
        """构建模型"""
        return build_from_config(cfg, MODEL_REGISTRY, kwargs)
    
    @staticmethod
    def list_models() -> list:
        """列出所有可用模型"""
        return MODEL_REGISTRY.list_modules()


class DatasetBuilder:
    """数据集构建器"""
    
    @staticmethod
    def build(cfg: Dict[str, Any], **kwargs) -> Any:
        """构建数据集"""
        return build_from_config(cfg, DATASET_REGISTRY, kwargs)
    
    @staticmethod
    def list_datasets() -> list:
        """列出所有可用数据集"""
        return DATASET_REGISTRY.list_modules()
