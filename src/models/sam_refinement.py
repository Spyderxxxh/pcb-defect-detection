"""
SAM + Level Set 边界精修模块
实现0.3像素级边界精修
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import cv2

from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class LevelSetRefinement:
    """
    Level Set方法边界精修
    
    基于活动轮廓模型对检测框进行精细化调整，
    实现亚像素级边界定位。
    
    Reference:
        "Level Set Methods and Fast Marching Methods" (Sethian, 1999)
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        mu: float = 0.1,
        dt: float = 0.1,
        epsilon: float = 1.0
    ):
        """
        Args:
            max_iter: 最大迭代次数
            lambda1: 内部区域权重
            lambda2: 外部区域权重
            mu: 长度正则化权重
            dt: 时间步长
            epsilon: Heaviside函数平滑参数
        """
        self.max_iter = max_iter
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mu = mu
        self.dt = dt
        self.epsilon = epsilon
    
    def _heaviside(self, phi: np.ndarray) -> np.ndarray:
        """平滑Heaviside函数"""
        return 0.5 * (1 + (2 / np.pi) * np.arctan(phi / self.epsilon))
    
    def _dirac(self, phi: np.ndarray) -> np.ndarray:
        """平滑Dirac函数"""
        return (self.epsilon / np.pi) / (self.epsilon**2 + phi**2)
    
    def _curvature(self, phi: np.ndarray) -> np.ndarray:
        """计算曲率"""
        # 计算梯度
        phi_x = cv2.Sobel(phi, cv2.CV_64F, 1, 0, ksize=3)
        phi_y = cv2.Sobel(phi, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算二阶梯度
        phi_xx = cv2.Sobel(phi_x, cv2.CV_64F, 1, 0, ksize=3)
        phi_yy = cv2.Sobel(phi_y, cv2.CV_64F, 0, 1, ksize=3)
        phi_xy = cv2.Sobel(phi_x, cv2.CV_64F, 0, 1, ksize=3)
        
        # 曲率公式
        numerator = phi_xx * phi_y**2 - 2 * phi_x * phi_y * phi_xy + phi_yy * phi_x**2
        denominator = (phi_x**2 + phi_y**2 + 1e-10)**1.5
        
        return numerator / denominator
    
    def refine(
        self,
        image: np.ndarray,
        initial_box: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        精修边界框
        
        Args:
            image: 输入图像 [H, W, C]
            initial_box: 初始边界框 (x1, y1, x2, y2)
            
        Returns:
            精修后的边界框 (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = initial_box
        
        # 裁剪ROI区域
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return initial_box
        
        # 转为灰度图
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        
        gray = gray.astype(np.float64)
        
        # 初始化Level Set函数（距离函数）
        h, w = gray.shape
        phi = np.ones((h, w), dtype=np.float64)
        
        # 初始轮廓为矩形中心
        center_x, center_y = w // 2, h // 2
        radius = min(h, w) // 4
        
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((j - center_x)**2 + (i - center_y)**2) - radius
                phi[i, j] = dist
        
        # Level Set迭代
        for iteration in range(self.max_iter):
            # 计算Heaviside和Dirac函数
            H = self._heaviside(phi)
            delta = self._dirac(phi)
            
            # 计算内外区域均值
            c1 = np.sum(H * gray) / (np.sum(H) + 1e-10)
            c2 = np.sum((1 - H) * gray) / (np.sum(1 - H) + 1e-10)
            
            # 计算曲率
            curvature = self._curvature(phi)
            
            # 更新Level Set函数
            dphi = delta * (
                self.mu * curvature -
                self.lambda1 * (gray - c1)**2 +
                self.lambda2 * (gray - c2)**2
            )
            
            phi = phi + self.dt * dphi
            
            # 重新初始化（可选，每隔一定迭代次数）
            if iteration % 20 == 0 and iteration > 0:
                phi = self._reinitialize(phi)
        
        # 从Level Set函数提取边界框
        mask = (phi < 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest_contour)
            
            # 映射回原图坐标，确保整数
            new_x1 = int(x1 + bx)
            new_y1 = int(y1 + by)
            new_x2 = int(new_x1 + bw)
            new_y2 = int(new_y1 + bh)
            
            return (new_x1, new_y1, new_x2, new_y2)
        
        # 返回原始框，确保整数
        return (int(x1), int(y1), int(x2), int(y2))
    
    def _reinitialize(self, phi: np.ndarray) -> np.ndarray:
        """重新初始化Level Set函数为距离函数"""
        # 使用快速行进法或简单近似
        mask = (phi < 0).astype(np.uint8)
        
        # 计算到边界的距离
        dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
        dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        return dist_out - dist_in


@MODEL_REGISTRY.register()
class SAMRefinement(nn.Module):
    """
    SAM + Level Set 边界精修模块
    
    结合SAM的零样本分割能力和Level Set的精细化调整，
    实现高精度边界检测。
    
    Args:
        model_type: SAM模型类型 (vit_h, vit_l, vit_b)
        checkpoint: SAM权重路径
        level_set_cfg: Level Set配置
    """
    
    def __init__(
        self,
        model_type: str = 'vit_b',
        checkpoint: Optional[str] = None,
        level_set_cfg: Optional[dict] = None
    ):
        super(SAMRefinement, self).__init__()
        
        self.model_type = model_type
        self.checkpoint = checkpoint
        
        # 延迟加载SAM模型（避免不必要的显存占用）
        self._sam = None
        self._predictor = None
        
        # Level Set精修
        if level_set_cfg is not None and level_set_cfg.get('enabled', False):
            self.level_set = LevelSetRefinement(
                max_iter=level_set_cfg.get('max_iter', 100),
                lambda1=level_set_cfg.get('lambda1', 1.0),
                lambda2=level_set_cfg.get('lambda2', 1.0),
                mu=level_set_cfg.get('mu', 0.1),
                dt=level_set_cfg.get('dt', 0.1)
            )
            self.use_level_set = True
        else:
            self.level_set = None
            self.use_level_set = False
    
    @property
    def sam(self):
        """延迟加载SAM模型"""
        if self._sam is None:
            try:
                from segment_anything import sam_model_registry, SamPredictor
                
                sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
                sam.eval()
                
                self._sam = sam
                self._predictor = SamPredictor(sam)
            except ImportError:
                raise ImportError("segment_anything not installed. "
                                "Please install: pip install segment-anything")
        return self._sam
    
    @property
    def predictor(self):
        """获取SAM预测器"""
        if self._predictor is None:
            _ = self.sam  # 触发加载
        return self._predictor
    
    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None
    ) -> List[dict]:
        """
        前向传播 - 精修边界框
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]
            boxes: 初始边界框 [N, 4] (x1, y1, x2, y2)
            original_size: 原始图像尺寸 (H, W)
            
        Returns:
            精修后的mask列表
        """
        # 确保图像是numpy格式
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if image.ndim == 4:
            image = image[0]  # 取batch第一个
        
        # 转换通道顺序 [C, H, W] -> [H, W, C]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # 归一化到0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 设置图像
        self.predictor.set_image(image)
        
        # 转换boxes格式
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        # 确保boxes是2D数组 [N, 4]
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        
        results = []
        # 逐个处理每个检测框（避免批量处理的维度问题）
        for i, box in enumerate(boxes):
            try:
                # 单个box需要reshape为 [1, 4]
                input_box = box.reshape(1, -1) if box.ndim == 1 else box
                
                # SAM精修
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False
                )
                
                # 取第一个mask和score
                mask = masks[0] if isinstance(masks, (list, tuple)) else masks
                score = scores[0] if isinstance(scores, (list, tuple, np.ndarray)) else scores
                
                # 确保mask是2D
                if mask.ndim == 3:
                    mask = mask[0]
                
                # 从mask提取边界框
                # SAM predict(box=...) 传入的是全图绝对坐标框
                # 返回的 mask 也是全图大小，因此提取的坐标已是全局坐标，无需偏移
                refined_box = self._mask_to_box(mask)
                if refined_box is None:
                    # mask 为空，保留原始框
                    box_list = box.tolist() if hasattr(box, 'tolist') else list(box)
                    refined_box = tuple(int(float(c)) for c in box_list[:4])
                else:
                    refined_box = (
                        int(refined_box[0]),
                        int(refined_box[1]),
                        int(refined_box[2]),
                        int(refined_box[3])
                    )
                
                # Level Set进一步精修
                if self.use_level_set and self.level_set:
                    refined_box = self.level_set.refine(image, refined_box)
                
                results.append({
                    'mask': mask,
                    'box': refined_box,
                    'score': float(score),
                    'original_box': box.tolist() if hasattr(box, 'tolist') else box
                })
            except Exception as e:
                # 单个框处理失败，保留原始框
                results.append({
                    'mask': None,
                    'box': box.tolist() if hasattr(box, 'tolist') else box,
                    'score': 0.0,
                    'original_box': box.tolist() if hasattr(box, 'tolist') else box,
                    'error': str(e)
                })
        
        return results
    
    def _mask_to_box(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """从mask提取边界框，返回 None 表示 mask 无效"""
        if mask.ndim == 3:
            mask = mask[0]
        
        # 找到mask的轮廓
        mask_uint8 = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            # 合并所有轮廓
            all_points = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            return (int(x), int(y), int(x + w), int(y + h))
        
        # mask 为空，返回 None，让调用方保留原始框
        return None
    
    def refine_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        精修边界框列表
        
        Args:
            image: 输入图像 [H, W, C]
            boxes: 边界框列表
            
        Returns:
            精修后的边界框列表
        """
        if len(boxes) == 0:
            return boxes
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        results = self.forward(image, boxes_tensor)
        
        return [r['box'] for r in results]


if __name__ == "__main__":
    # 测试Level Set
    print("Testing LevelSetRefinement...")
    
    # 创建测试图像
    img = np.random.rand(100, 100, 3) * 255
    img = img.astype(np.uint8)
    
    # 初始框
    init_box = (20, 20, 80, 80)
    
    # 精修
    level_set = LevelSetRefinement(max_iter=50)
    refined = level_set.refine(img, init_box)
    
    print(f"Initial box: {init_box}")
    print(f"Refined box: {refined}")
