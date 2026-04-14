"""
训练脚本
支持YOLOv8 + NonLocal + SAM的完整训练流程
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from src.core.config import load_config, Config
from src.core.registry import MODEL_REGISTRY, build_from_config
from src.models import YOLOv8NonLocal
from src.data_engineering import PCBDefectDataset, collate_fn
from src.agents import CollectorAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Train DefectGuard Model')
    parser.add_argument('--config', type=str, default='configs/defectguard_yolov8.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU设备ID')
    return parser.parse_args()


def setup_device(device_str: str):
    """设置训练设备"""
    if device_str == 'cpu':
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_str}')
        torch.cuda.set_device(device)
        
        # 设置cuDNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    
    return device


def build_model(config: Config, device: torch.device):
    """构建模型"""
    model_cfg = config.get('model', {})
    
    print(f"Building model: {model_cfg.get('type', 'YOLOv8NonLocal')}")
    
    # 使用YOLOv8官方训练流程
    from ultralytics import YOLO
    
    model_type = model_cfg.get('type', 'YOLOv8NonLocal')
    
    if model_type == 'YOLOv8NonLocal':
        # 加载基础YOLOv8
        backbone = model_cfg.get('backbone', 'm')
        pretrained = model_cfg.get('pretrained', True)
        
        if pretrained:
            model = YOLO(f'yolov8{backbone}.pt')
        else:
            model = YOLO(f'yolov8{backbone}.yaml')
        
        # 修改类别数
        num_classes = model_cfg.get('num_classes', 5)
        if num_classes != 80:
            model.model.nc = num_classes
    else:
        model = build_from_config(model_cfg, MODEL_REGISTRY)
    
    return model


def build_dataloaders(config: Config):
    """构建数据加载器"""
    dataset_cfg = config.get('dataset', {})
    
    data_root = dataset_cfg.get('data_root', './data/pcb_defect')
    img_size = dataset_cfg.get('img_size', 640)
    batch_size = dataset_cfg.get('batch_size', 16)
    num_workers = dataset_cfg.get('num_workers', 4)
    augmentation = dataset_cfg.get('augmentation', {})
    
    print(f"Building dataloaders from: {data_root}")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_root):
        print(f"Warning: Data root not found: {data_root}")
        print("Please prepare your dataset first.")
        return None, None
    
    # 构建数据集
    train_dataset = PCBDefectDataset(
        data_root=data_root,
        split='train',
        img_size=img_size,
        augment=True,
        augmentation_config=augmentation if augmentation.get('enabled', False) else None
    )
    
    val_dataset = PCBDefectDataset(
        data_root=data_root,
        split='val',
        img_size=img_size,
        augment=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def train_with_ultralytics(config: Config, device: torch.device):
    """
    使用Ultralytics框架训练
    这是推荐的训练方式，支持完整的YOLOv8功能
    """
    from ultralytics import YOLO
    
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    training_cfg = config.get('training', {})
    
    # 加载模型
    backbone = model_cfg.get('backbone', 'm')
    model = YOLO(f'yolov8{backbone}.pt')
    
    # 训练参数
    data_yaml = os.path.join(dataset_cfg.get('data_root', './data/pcb_defect'), 'data.yaml')
    
    # 如果data.yaml不存在，创建一个
    if not os.path.exists(data_yaml):
        create_data_yaml(dataset_cfg.get('data_root', './data/pcb_defect'))
    
    train_args = {
        'data': data_yaml,
        'epochs': training_cfg.get('epochs', 100),
        'batch': dataset_cfg.get('batch_size', 16),
        'imgsz': dataset_cfg.get('img_size', 640),
        'lr0': training_cfg.get('optimizer', {}).get('lr', 0.001),
        'lrf': 0.01,
        'momentum': training_cfg.get('optimizer', {}).get('momentum', 0.9),
        'weight_decay': training_cfg.get('optimizer', {}).get('weight_decay', 0.0005),
        'warmup_epochs': training_cfg.get('warmup_epochs', 3),
        'box': training_cfg.get('loss', {}).get('box', 7.5),
        'cls': training_cfg.get('loss', {}).get('cls', 0.5),
        'dfl': training_cfg.get('loss', {}).get('dfl', 1.5),
        'device': device.index if device.type == 'cuda' else 'cpu',
        'workers': dataset_cfg.get('num_workers', 4),
        'project': './outputs',
        'name': 'defectguard_train',
        'exist_ok': True,
        'pretrained': model_cfg.get('pretrained', True),
        'optimizer': training_cfg.get('optimizer', {}).get('type', 'AdamW'),
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'single_cls': False,
        'rect': False,
        'cos_lr': training_cfg.get('scheduler', {}).get('type') == 'CosineAnnealingLR',
        'close_mosaic': 10,
        'resume': False,
        'amp': True,  # 自动混合精度
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'save': True,
        'save_period': training_cfg.get('save_period', 10),
        'cache': False,
        'patience': training_cfg.get('early_stopping', {}).get('patience', 20) if training_cfg.get('early_stopping', {}).get('enabled', False) else 100,
    }
    
    # 数据增强参数
    augmentation = dataset_cfg.get('augmentation', {})
    if augmentation.get('enabled', False):
        train_args.update({
            'hsv_h': augmentation.get('hsv_h', 0.015),
            'hsv_s': augmentation.get('hsv_s', 0.7),
            'hsv_v': augmentation.get('hsv_v', 0.4),
            'degrees': augmentation.get('degrees', 0.0),
            'translate': augmentation.get('translate', 0.1),
            'scale': augmentation.get('scale', 0.5),
            'shear': augmentation.get('shear', 0.0),
            'perspective': augmentation.get('perspective', 0.0),
            'flipud': augmentation.get('flipud', 0.0),
            'fliplr': augmentation.get('fliplr', 0.5),
            'mosaic': augmentation.get('mosaic', 1.0),
            'mixup': augmentation.get('mixup', 0.0),
            'copy_paste': augmentation.get('copy_paste', 0.0),
        })
    
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    # 启动训练
    results = model.train(**train_args)
    
    return results


def create_data_yaml(data_root: str):
    """创建YOLO格式的data.yaml文件"""
    yaml_content = f"""# PCB Defect Detection Dataset
path: {data_root}  # dataset root dir
train: images/train  # train images
val: images/val  # val images
test: images/test  # test images (optional)

# Classes
names:
  0: missing
  1: mousebite
  2: open_circuit
  3: short
  4: spur
"""
    
    yaml_path = os.path.join(data_root, 'data.yaml')
    os.makedirs(data_root, exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created data.yaml at: {yaml_path}")


def main():
    args = parse_args()
    
    # 加载配置
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # 设置设备
    device = setup_device(args.device)
    
    # 创建输出目录
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # 使用Ultralytics训练（推荐）
    results = train_with_ultralytics(config, device)
    
    print("\n" + "="*50)
    print("Training Completed!")
    print("="*50)
    print(f"Results saved to: ./outputs/defectguard_train")
    
    return results


if __name__ == '__main__':
    main()
