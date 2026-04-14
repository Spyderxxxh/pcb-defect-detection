"""
YOLOv8 训练脚本 - 简化版
直接使用Ultralytics框架训练，支持DeepPCB等公开数据集
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on PCB Defect Dataset')
    parser.add_argument('--data', type=str, default='./data/deeppcb/dataset.yaml',
                        help='数据集yaml配置文件路径')
    parser.add_argument('--model', type=str, default='yolov8m',
                        help='模型类型: yolov8n/s/m/l/x')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU设备ID')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='输出项目目录')
    parser.add_argument('--name', type=str, default='deeppcb_train',
                        help='实验名称')
    parser.add_argument('--resume', action='store_true',
                        help='恢复训练')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用预训练权重')
    return parser.parse_args()

def check_dataset(data_yaml):
    """检查数据集是否存在"""
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"Error: Dataset config not found: {data_yaml}")
        print("\nPlease prepare dataset first:")
        print("  python scripts/prepare_deeppcb.py")
        return False
    
    # 读取yaml检查路径
    with open(data_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    dataset_path = Path(data_cfg.get('path', ''))
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return False
    
    # 检查训练集
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
    
    print("="*60)
    print("DefectGuard - YOLOv8 Training")
    print("="*60)
    
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
    
    # 加载模型
    print(f"\nLoading model: {args.model}")
    if args.pretrained:
        model = YOLO(f'{args.model}.pt')
        print(f"  Using pretrained weights: {args.model}.pt")
    else:
        model = YOLO(f'{args.model}.yaml')
        print(f"  Training from scratch")
    
    # 训练参数
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Dataset: {args.data}")
    print(f"  Output: {args.project}/{args.name}")
    
    # 开始训练
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=args.pretrained,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=50,  # 早停耐心值
        save=True,
        save_period=10,
        cache=False,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Results saved to: {args.project}/{args.name}")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    
    # 验证
    print("\nRunning validation...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return results

if __name__ == "__main__":
    main()
