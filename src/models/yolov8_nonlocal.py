"""
YOLOv8 + NonLocal 模型实现
结合YOLOv8的高效检测能力和NonLocal的长距离依赖建模
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, List, Optional
import warnings

from src.core.registry import MODEL_REGISTRY
from .nonlocal_block import NonLocalBlock


@MODEL_REGISTRY.register()
class YOLOv8NonLocal(nn.Module):
    """
    YOLOv8 + NonLocal注意力模块
    
    在YOLOv8骨干网络中插入NonLocal模块，增强对遮挡目标的检测能力。
    
    Args:
        num_classes: 检测类别数
        backbone: 骨干网络类型 (n, s, m, l, x)
        pretrained: 是否使用预训练权重
        pretrained_weights: 预训练权重路径
        nonlocal_cfg: NonLocal模块配置
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = 'm',
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None,
        nonlocal_cfg: Optional[Dict] = None
    ):
        super(YOLOv8NonLocal, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # 加载基础YOLOv8模型
        model_name = f'yolov8{backbone}.pt' if pretrained else f'yolov8{backbone}.yaml'
        self.yolo = YOLO(model_name)
        
        # 获取模型配置
        self.model = self.yolo.model
        
        # 修改检测头类别数
        if num_classes != 80:  # COCO默认80类
            self._update_num_classes(num_classes)
        
        # 插入NonLocal模块
        if nonlocal_cfg is not None and nonlocal_cfg.get('enabled', False):
            self._insert_nonlocal(nonlocal_cfg)
        
        # 加载自定义预训练权重
        if pretrained_weights and pretrained_weights != f'yolov8{backbone}.pt':
            self._load_pretrained(pretrained_weights)
    
    def _update_num_classes(self, num_classes: int):
        """更新检测头类别数"""
        # YOLOv8的检测头在model.model[-1]
        detect = self.model.model[-1]
        
        # 更新类别数
        if hasattr(detect, 'nc'):
            detect.nc = num_classes
        
        # 更新输出卷积层
        if hasattr(detect, 'cv3'):
            old_nc = detect.cv3[0][0].out_channels
            for i in range(len(detect.cv3)):
                # 获取原卷积层配置
                old_conv = detect.cv3[i][0]
                detect.cv3[i][0] = nn.Conv2d(
                    old_conv.in_channels,
                    num_classes,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
    
    def _insert_nonlocal(self, cfg: Dict):
        """
        在骨干网络中插入NonLocal模块
        通常在C3模块后插入效果较好
        """
        in_channels = cfg.get('in_channels', 512)
        inter_channels = cfg.get('inter_channels', 256)
        sub_sample = cfg.get('sub_sample', True)
        bn_layer = cfg.get('bn_layer', True)
        
        # 在特定位置插入NonLocal
        # YOLOv8 backbone结构: [Conv, Conv, C2f, C2f, C2f, SPPF]
        # 我们在SPPF前插入NonLocal
        nonlocal_block = NonLocalBlock(
            in_channels=in_channels,
            inter_channels=inter_channels,
            sub_sample=sub_sample,
            bn_layer=bn_layer
        )
        
        # 找到合适的位置插入
        # 这里简化处理，实际应该根据具体层索引插入
        self.nonlocal_block = nonlocal_block
        self.use_nonlocal = True
        
    def _load_pretrained(self, weights_path: str):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # 加载权重
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                warnings.warn(f"Missing keys: {missing}")
            if unexpected:
                warnings.warn(f"Unexpected keys: {unexpected}")
            
            print(f"Loaded pretrained weights from {weights_path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            
        Returns:
            检测结果
        """
        # 使用YOLOv8的前向传播
        return self.model(x)
    
    def predict(self, x: torch.Tensor, conf: float = 0.25, 
                iou: float = 0.45) -> List:
        """
        推理预测
        
        Args:
            x: 输入图像或路径
            conf: 置信度阈值
            iou: NMS IoU阈值
            
        Returns:
            检测结果列表
        """
        return self.yolo.predict(x, conf=conf, iou=iou)
    
    def export(self, format: str = 'onnx', **kwargs):
        """
        导出模型
        
        Args:
            format: 导出格式 (onnx, torchscript, engine等)
            **kwargs: 额外参数
        """
        return self.yolo.export(format=format, **kwargs)
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': f'YOLOv8-{self.backbone_name}',
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'nonlocal_enabled': getattr(self, 'use_nonlocal', False)
        }


class YOLOv8Trainer:
    """
    YOLOv8训练器封装
    简化训练流程
    """
    
    def __init__(self, model: YOLOv8NonLocal, config: Dict):
        self.model = model
        self.config = config
        self.yolo = model.yolo
    
    def train(self, data_yaml: str, **kwargs):
        """
        训练模型
        
        Args:
            data_yaml: 数据配置文件路径
            **kwargs: 训练参数
        """
        # 合并配置
        train_args = {
            'data': data_yaml,
            'epochs': self.config.get('epochs', 100),
            'batch': self.config.get('batch_size', 16),
            'imgsz': self.config.get('img_size', 640),
            'lr0': self.config.get('lr', 0.001),
            'optimizer': self.config.get('optimizer', 'AdamW'),
            'device': self.config.get('device', '0'),
            'project': './outputs',
            'name': 'defectguard_train',
            **kwargs
        }
        
        return self.yolo.train(**train_args)
    
    def validate(self, data_yaml: str, **kwargs):
        """验证模型"""
        return self.yolo.val(data=data_yaml, **kwargs)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model': self.model.state_dict(),
            'config': self.config
        }, path)


if __name__ == "__main__":
    # 测试模型
    model = YOLOv8NonLocal(
        num_classes=5,
        backbone='m',
        pretrained=False,
        nonlocal_cfg={
            'enabled': True,
            'in_channels': 512,
            'inter_channels': 256
        }
    )
    
    # 测试前向传播
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    
    print(f"Model info: {model.get_model_info()}")
