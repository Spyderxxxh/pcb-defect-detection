"""
数据集实现
支持YOLO格式和自定义PCB缺陷检测数据集
"""

import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class YOLODataset(Dataset):
    """
    YOLO格式数据集
    
    目录结构：
    data/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: int = 640,
        augment: bool = True,
        augmentation_config: Optional[Dict] = None
    ):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        self.img_dir = os.path.join(data_root, 'images', split)
        self.label_dir = os.path.join(data_root, 'labels', split)
        
        # 加载图像列表
        self.img_files = self._load_image_list()
        
        # 数据增强
        self.transform = self._build_transform(augmentation_config)
    
    def _load_image_list(self) -> List[str]:
        """加载图像列表"""
        img_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            img_files.extend([
                f for f in os.listdir(self.img_dir)
                if f.lower().endswith(ext)
            ])
        return sorted(img_files)
    
    def _build_transform(self, config: Optional[Dict]) -> A.Compose:
        """构建数据增强管道"""
        if not self.augment or config is None:
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        transforms = [
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size),
        ]
        
        # 颜色增强
        if config.get('hsv_h', 0) > 0:
            transforms.append(A.HueSaturationValue(
                hue_limit=int(config['hsv_h'] * 180),
                sat_limit=int(config['hsv_s'] * 100),
                val_limit=int(config['hsv_v'] * 100)
            ))
        
        # 几何变换
        if config.get('degrees', 0) > 0:
            transforms.append(A.Rotate(limit=int(config['degrees'])))
        
        if config.get('translate', 0) > 0:
            transforms.append(A.ShiftScaleRotate(
                shift_limit=config['translate'],
                scale_limit=config.get('scale', 0.5) - 1,
                rotate_limit=0
            ))
        
        if config.get('fliplr', 0) > 0:
            transforms.append(A.HorizontalFlip(p=config['fliplr']))
        
        # Mosaic增强
        if config.get('mosaic', 0) > 0 and np.random.random() < config['mosaic']:
            # Mosaic需要特殊处理，这里简化
            pass
        
        transforms.extend([
            A.Normalize(),
            ToTensorV2()
        ])
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        
        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)
        
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels
            )
            img = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        
        # 构建目标张量
        targets = []
        for bbox, cls in zip(bboxes, class_labels):
            targets.append([cls] + list(bbox))
        
        if not targets:
            targets = [[0, 0, 0, 0, 0]]  # 填充空目标
        
        targets = torch.tensor(targets, dtype=torch.float32)
        
        return img, targets


@MODEL_REGISTRY.register()
class PCBDefectDataset(YOLODataset):
    """
    PCB缺陷检测数据集
    
    缺陷类别：
    0: missing (缺件)
    1: mousebite (鼠咬)
    2: open_circuit (开路)
    3: short (短路)
    4: spur (毛刺)
    """
    
    CLASSES = ['missing', 'mousebite', 'open_circuit', 'short', 'spur']
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: int = 640,
        augment: bool = True,
        augmentation_config: Optional[Dict] = None
    ):
        super().__init__(data_root, split, img_size, augment, augmentation_config)
        self.num_classes = len(self.CLASSES)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布统计"""
        distribution = {cls: 0 for cls in self.CLASSES}
        
        for img_file in self.img_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            if class_id < len(self.CLASSES):
                                distribution[self.CLASSES[class_id]] += 1
        
        return distribution


def collate_fn(batch):
    """自定义collate函数处理不同大小的目标"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    
    # 为每个样本的目标添加batch索引
    targets_with_batch = []
    for i, target in enumerate(targets):
        batch_idx = torch.full((len(target), 1), i)
        targets_with_batch.append(torch.cat([batch_idx, target], dim=1))
    
    targets = torch.cat(targets_with_batch, 0)
    
    return images, targets


if __name__ == "__main__":
    # 测试数据集
    # dataset = PCBDefectDataset(
    #     data_root='./data/pcb_defect',
    #     split='train',
    #     img_size=640
    # )
    # 
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Class distribution: {dataset.get_class_distribution()}")
    # 
    # img, target = dataset[0]
    # print(f"Image shape: {img.shape}")
    # print(f"Target shape: {target.shape}")
    pass
