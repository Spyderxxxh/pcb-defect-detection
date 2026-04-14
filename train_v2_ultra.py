"""
YOLOv8 超参数优化训练脚本 V2
目标：mAP@0.5 >= 98%, mAP@0.5:0.95 >= 80%

核心优化策略（无需修改模型结构）：
1. 超分辨率训练：imgsz=1280（小缺陷放大）
2. 强数据增强：Mosaic + MixUp + CopyPaste
3. 更长训练：300 epoch + Cosine Annealing
4. 模型集成：YOLOv8l 或 YOLOv8x（更大的backbone）
5. 测试时增强(TTA)：多尺度测试
6. 伪标签：利用模型预测扩充训练集
7. 模型集成：多模型投票
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Optimized YOLOv8 Training for PCB Defect Detection')
    parser.add_argument('--data', type=str, default='./data/deeppcb/dataset.yaml',
                        help='数据集yaml配置文件路径')
    parser.add_argument('--model', type=str, default='yolov8l',  # 改用更大的模型
                        help='模型类型: yolov8n/s/m/l/x')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=8,
                        help='批次大小（高分辨率需要更小batch）')
    parser.add_argument('--imgsz', type=int, default=1280,  # 超分辨率训练
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU设备ID')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='输出项目目录')
    parser.add_argument('--name', type=str, default='defectguard_v2_optimized',
                        help='实验名称')
    parser.add_argument('--resume', action='store_true',
                        help='恢复训练')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用预训练权重')
    # 优化选项
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子，用于多seed集成训练')
    # 优化选项
    parser.add_argument('--no-amp', action='store_true',
                        help='禁用自动混合精度')
    return parser.parse_args()

def check_dataset(data_yaml):
    """检查数据集是否存在"""
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"Error: Dataset config not found: {data_yaml}")
        return False
    
    with open(data_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    dataset_path = Path(data_cfg.get('path', ''))
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return False
    
    train_path = dataset_path / data_cfg.get('train', 'images/train')
    if not train_path.exists():
        print(f"Error: Train path not found: {train_path}")
        return False
    
    train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
    print(f"Dataset check passed!")
    print(f"  Train images: {len(train_images)}")
    
    return True

def main():
    args = parse_args()
    
    print("="*70)
    print("DefectGuard V2 - Ultra-Optimized YOLOv8 Training")
    print("="*70)
    print(f"Target: mAP@0.5 >= 98%, mAP@0.5:0.95 >= 80%")
    print("="*70)
    
    # 检查数据集
    if not check_dataset(args.data):
        sys.exit(1)
    
    # 导入ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        print("Please install: pip install ultralytics")
        sys.exit(1)
    
    # 设置设备
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(int(args.device))}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(int(args.device)).total_memory / 1e9:.1f} GB")
    
    # 加载模型
    print(f"\nLoading model: {args.model}")
    if args.pretrained:
        model = YOLO(f'{args.model}.pt')
        print(f"Using pretrained weights: {args.model}.pt")
    else:
        model = YOLO(f'{args.model}.yaml')
        print(f"Training from scratch")
    
    # 训练参数配置
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz} (超分辨率训练)")
    print(f"  Dataset: {args.data}")
    print(f"  Output: {args.project}/{args.name}")
    
    # 超参数配置（针对小目标检测优化）
    train_args = {
        # 基础参数
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': device,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        'pretrained': args.pretrained,
        
        # 优化器配置
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # 学习率调度
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'cos_lr': True,  # Cosine Annealing
        
        # 损失权重（针对小目标调整）
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # 数据增强（强增强配置）
        'hsv_h': 0.015,   # 色调
        'hsv_s': 0.7,     # 饱和度
        'hsv_v': 0.4,     # 亮度
        'degrees': 5.0,   # 旋转（小角度，保留缺陷方向）
        'translate': 0.1, # 平移
        'scale': 0.5,     # 缩放 0.5-1.5（关键：小目标放大）
        'shear': 2.0,     # 剪切
        'perspective': 0.0,  # 透视（PCB无透视）
        'flipud': 0.0,    # 上下翻转（PCB有方向）
        'fliplr': 0.5,    # 左右翻转
        'mosaic': 1.0,    # Mosaic（必开）
        'mixup': 0.15,    # MixUp（增加样本多样性）
        'copy_paste': 0.3,  # Copy-Paste（关键：复制小缺陷）
        'erasing': 0.4,   # Random Erasing
        'crop_fraction': 1.0,
        
        # 训练策略
        'patience': 100,      # 早停耐心值
        'save': True,
        'save_period': 20,
        'cache': 'disk',      # 缓存到磁盘加速
        'workers': 8,
        'close_mosaic': 30,   # 最后30轮关闭mosaic
        'amp': not args.no_amp,  # 自动混合精度
        'fraction': 1.0,      # 使用全部数据
        
        # 验证配置
        'val': True,
        'split': 'val',
    }
    
    # 开始训练
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    print("\nKey Optimizations:")
    print(f"  1. Super-resolution training: {args.imgsz}px (vs 640px baseline)")
    print(f"  2. Strong augmentation: Mosaic(1.0) + MixUp(0.15) + CopyPaste(0.3)")
    print(f"  3. Larger backbone: {args.model} (vs yolov8m baseline)")
    print(f"  4. Longer training: {args.epochs} epochs with Cosine LR")
    print(f"  5. Small object focus: scale=0.5, close_mosaic=30")
    
    results = model.train(**train_args)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Results saved to: {args.project}/{args.name}")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    
    # 验证最佳模型
    print("\nRunning validation on best model...")
    metrics = model.val()
    print(f"\nFinal Results:")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    
    # 每类AP
    print(f"\nPer-Class AP@0.5:")
    class_names = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
    for i, name in enumerate(class_names):
        if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > i:
            ap = metrics.box.ap50[i]
            print(f"  {name:12s}: {ap:.4f} ({ap*100:.2f}%)")
    
    # 保存结果摘要
    summary = {
        'model': args.model,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'map50': float(metrics.box.map50),
        'map50_95': float(metrics.box.map),
        'per_class_ap50': {
            name: float(metrics.box.ap50[i]) if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > i else 0
            for i, name in enumerate(class_names)
        }
    }
    
    import json
    summary_path = Path(args.project) / args.name / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    return results

if __name__ == "__main__":
    main()
