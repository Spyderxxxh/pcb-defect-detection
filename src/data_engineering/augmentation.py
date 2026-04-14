"""
数据增强管道
支持 Mosaic、MixUp、HSV、几何变换等
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Callable
import random


class AugmentationPipeline:
    """
    数据增强管道
    
    支持：
    - Mosaic增强：4图拼接提升小目标检测
    - MixUp：图像混合增强
    - HSV颜色空间增强
    - 几何变换：旋转、平移、缩放、翻转
    """
    
    def __init__(
        self,
        img_size: int = 640,
        hsv_h: float = 0.015,  # HSV色调
        hsv_s: float = 0.7,    # HSV饱和度
        hsv_v: float = 0.4,    # HSV亮度
        degrees: float = 0.0,  # 旋转角度
        translate: float = 0.1,  # 平移
        scale: float = 0.5,    # 缩放
        shear: float = 0.0,    # 剪切
        perspective: float = 0.0,  # 透视
        flipud: float = 0.0,   # 上下翻转
        fliplr: float = 0.5,   # 左右翻转
        mosaic: float = 1.0,   # Mosaic概率
        mixup: float = 0.0,    # MixUp概率
        copy_paste: float = 0.0,  # Copy-Paste概率
        auto_augment: Optional[str] = None,  # AutoAugment策略
    ):
        self.img_size = img_size
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flipud = flipud
        self.fliplr = fliplr
        self.mosaic_prob = mosaic
        self.mixup_prob = mixup
        self.copy_paste_prob = copy_paste
        self.auto_augment = auto_augment
        
        # 存储Mosaic需要的其他样本
        self.mosaic_buffer = []
        self.mosaic_size = 4
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
        mosaic_samples: Optional[List[Tuple]] = None
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        应用数据增强
        
        Args:
            image: 输入图像 (H, W, C)
            bboxes: YOLO格式边界框 [x_center, y_center, width, height]
            class_labels: 类别标签
            mosaic_samples: Mosaic增强需要的其他样本 [(img, bboxes, labels), ...]
        
        Returns:
            增强后的图像、边界框、类别标签
        """
        # Mosaic增强
        if mosaic_samples and random.random() < self.mosaic_prob:
            image, bboxes, class_labels = self._apply_mosaic(
                image, bboxes, class_labels, mosaic_samples
            )
        
        # 标准Albumentations增强
        transform = self._build_transform()
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        image = transformed['image']
        bboxes = transformed['bboxes']
        class_labels = transformed['class_labels']
        
        # MixUp增强
        if mosaic_samples and random.random() < self.mixup_prob:
            mix_img, mix_bboxes, mix_labels = random.choice(mosaic_samples)
            image, bboxes, class_labels = self._apply_mixup(
                image, bboxes, class_labels,
                mix_img, mix_bboxes, mix_labels
            )
        
        return image, list(bboxes), list(class_labels)
    
    def _build_transform(self) -> A.Compose:
        """构建基础增强管道"""
        transforms = []
        
        # 颜色增强
        if self.hsv_h > 0 or self.hsv_s > 0 or self.hsv_v > 0:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=int(self.hsv_h * 180),
                sat_shift_limit=int(self.hsv_s * 255),
                val_shift_limit=int(self.hsv_v * 255),
                p=0.5
            ))
        
        # 几何变换
        if self.degrees > 0 or self.translate > 0 or self.scale > 0 or self.shear > 0:
            transforms.append(A.ShiftScaleRotate(
                shift_limit=self.translate,
                scale_limit=self.scale - 1,
                rotate_limit=int(self.degrees),
                p=0.5
            ))
        
        # 翻转
        if self.flipud > 0:
            transforms.append(A.VerticalFlip(p=self.flipud))
        if self.fliplr > 0:
            transforms.append(A.HorizontalFlip(p=self.fliplr))
        
        # 透视变换
        if self.perspective > 0:
            transforms.append(A.Perspective(
                scale=(0.05, self.perspective),
                p=0.5
            ))
        
        # 标准化
        transforms.extend([
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=114  # YOLO灰色填充
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3  # 过滤过小/超出边界的框
            )
        )
    
    def _apply_mosaic(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
        samples: List[Tuple]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Mosaic增强：4张图拼接成1张
        
        布局：
        | 1 | 2 |
        | 3 | 4 |
        """
        if len(samples) < 3:
            return image, bboxes, class_labels
        
        # 随机选择中心点
        xc = int(random.uniform(self.img_size * 0.5, self.img_size * 1.5))
        yc = int(random.uniform(self.img_size * 0.5, self.img_size * 1.5))
        
        # 创建大画布
        mosaic_img = np.full(
            (self.img_size * 2, self.img_size * 2, 3),
            114, dtype=np.uint8
        )
        
        all_bboxes = []
        all_labels = []
        
        # 4个位置的偏移
        positions = [
            (0, 0, xc, yc),           # 左上
            (xc, 0, self.img_size * 2, yc),  # 右上
            (0, yc, xc, self.img_size * 2),  # 左下
            (xc, yc, self.img_size * 2, self.img_size * 2)  # 右下
        ]
        
        samples = [(image, bboxes, class_labels)] + list(samples[:3])
        
        for i, (img, boxes, labels) in enumerate(samples):
            if img is None:
                continue
            
            h, w = img.shape[:2]
            x1a, y1a, x2a, y2a = positions[i]
            
            # 计算源图像裁剪区域
            if i == 0:  # 左上
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # 右上
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # 左下
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            else:  # 右下
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)
            
            # 粘贴到画布
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # 调整边界框坐标
            dx = x1a - x1b
            dy = y1a - y1b
            
            for box, label in zip(boxes, labels):
                x_c, y_c, bw, bh = box
                
                # 转换到绝对坐标
                x1 = (x_c - bw / 2) * w + dx
                y1 = (y_c - bh / 2) * h + dy
                x2 = (x_c + bw / 2) * w + dx
                y2 = (y_c + bh / 2) * h + dy
                
                # 裁剪到当前区域
                x1 = max(x1a, min(x1, x2a))
                y1 = max(y1a, min(y1, y2a))
                x2 = max(x1a, min(x2, x2a))
                y2 = max(y1a, min(y2, y2a))
                
                # 过滤无效框
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                
                # 转换回YOLO格式（相对于mosaic图）
                new_x_c = ((x1 + x2) / 2) / (self.img_size * 2)
                new_y_c = ((y1 + y2) / 2) / (self.img_size * 2)
                new_bw = (x2 - x1) / (self.img_size * 2)
                new_bh = (y2 - y1) / (self.img_size * 2)
                
                all_bboxes.append([new_x_c, new_y_c, new_bw, new_bh])
                all_labels.append(label)
        
        # 缩放到目标尺寸
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))
        
        # 调整bbox坐标
        scale = self.img_size / (self.img_size * 2)
        all_bboxes = [[b[0], b[1], b[2] * 2, b[3] * 2] for b in all_bboxes]
        
        return mosaic_img, all_bboxes, all_labels
    
    def _apply_mixup(
        self,
        img1: np.ndarray,
        bboxes1: List[List[float]],
        labels1: List[int],
        img2: np.ndarray,
        bboxes2: List[List[float]],
        labels2: List[int],
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        MixUp增强：两张图像按权重混合
        """
        # 确保尺寸一致
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h))
        
        # 随机混合系数
        lam = np.random.beta(alpha, alpha)
        
        # 混合图像
        mixed_img = lam * img1 + (1 - lam) * img2
        mixed_img = mixed_img.astype(np.uint8)
        
        # 合并边界框（简单合并，实际应用可更复杂）
        mixed_bboxes = bboxes1 + bboxes2
        mixed_labels = labels1 + labels2
        
        return mixed_img, mixed_bboxes, mixed_labels


class CopyPaste:
    """
    Copy-Paste增强：复制目标并粘贴到其他位置
    用于增加小目标样本
    """
    
    def __init__(self, p: float = 0.5, max_objects: int = 3):
        self.p = p
        self.max_objects = max_objects
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        if random.random() > self.p or len(bboxes) == 0:
            return image, bboxes, class_labels
        
        h, w = image.shape[:2]
        new_bboxes = list(bboxes)
        new_labels = list(class_labels)
        
        # 随机选择要复制的目标
        num_copy = min(self.max_objects, len(bboxes))
        indices = random.sample(range(len(bboxes)), num_copy)
        
        for idx in indices:
            x_c, y_c, bw, bh = bboxes[idx]
            
            # 转换为像素坐标
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            
            # 提取目标区域
            obj_img = image[y1:y2, x1:x2].copy()
            obj_h, obj_w = obj_img.shape[:2]
            
            # 随机选择粘贴位置
            new_x1 = random.randint(0, max(1, w - obj_w))
            new_y1 = random.randint(0, max(1, h - obj_h))
            new_x2 = new_x1 + obj_w
            new_y2 = new_y1 + obj_h
            
            # 粘贴（简单覆盖，可用泊松融合优化）
            image[new_y1:new_y2, new_x1:new_x2] = obj_img
            
            # 添加新边界框
            new_x_c = (new_x1 + new_x2) / 2 / w
            new_y_c = (new_y1 + new_y2) / 2 / h
            new_bw = obj_w / w
            new_bh = obj_h / h
            
            new_bboxes.append([new_x_c, new_y_c, new_bw, new_bh])
            new_labels.append(class_labels[idx])
        
        return image, new_bboxes, new_labels
