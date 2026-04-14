"""
测试时增强 (TTA) 推理脚本
通过多尺度测试和水平翻转提升检测精度

使用方法:
    python inference_tta.py --weights runs/detect/defectguard_v2_optimized/weights/best.pt \
                            --source data/deeppcb/yolo_format/images/test \
                            --output outputs/tta_results
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='TTA Inference for YOLOv8')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--source', type=str, required=True,
                        help='测试图像目录')
    parser.add_argument('--output', type=str, default='outputs/tta_results',
                        help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU设备ID')
    # TTA配置
    parser.add_argument('--tta', action='store_true', default=True,
                        help='启用测试时增强')
    parser.add_argument('--scales', type=int, nargs='+', default=[960, 1280, 1600],
                        help='多尺度测试尺寸')
    parser.add_argument('--flip', action='store_true', default=True,
                        help='启用水平翻转TTA')
    parser.add_argument('--merge-iou', type=float, default=0.5,
                        help='TTA结果合并的IoU阈值')
    return parser.parse_args()

def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, method='gaussian'):
    """
    Soft-NMS: 降低重叠框的分数而不是直接删除
    
    Args:
        boxes: [N, 4] 边界框 (xyxy)
        scores: [N] 置信度分数
        iou_threshold: IoU阈值
        sigma: 高斯函数的sigma参数
        method: 'linear', 'gaussian', 或 'hard'
    
    Returns:
        保留的索引列表
    """
    N = len(boxes)
    if N == 0:
        return []
    
    indexes = np.arange(N)
    keep = []
    
    while len(indexes) > 0:
        # 选择分数最高的框
        max_idx = np.argmax(scores[indexes])
        max_box_idx = indexes[max_idx]
        keep.append(max_box_idx)
        
        if len(indexes) == 1:
            break
        
        # 计算当前框与其他框的IoU
        max_box = boxes[max_box_idx]
        other_indexes = np.delete(indexes, max_idx)
        other_boxes = boxes[other_indexes]
        
        ious = compute_iou_batch(max_box.reshape(1, 4), other_boxes).flatten()
        
        # 根据IoU降低分数
        for i, iou in enumerate(ious):
            if iou > iou_threshold:
                if method == 'linear':
                    scores[other_indexes[i]] *= (1 - iou)
                elif method == 'gaussian':
                    scores[other_indexes[i]] *= np.exp(-(iou * iou) / sigma)
                elif method == 'hard':
                    scores[other_indexes[i]] = 0
        
        # 移除当前最高分的框
        indexes = np.delete(indexes, max_idx)
        
        # 移除分数低于阈值的框
        indexes = indexes[scores[indexes] > 0.01]
    
    return keep

def compute_iou_batch(boxes1, boxes2):
    """
    计算两组框之间的IoU矩阵
    
    Args:
        boxes1: [N, 4] (xyxy)
        boxes2: [M, 4] (xyxy)
    
    Returns:
        iou: [N, M] IoU矩阵
    """
    # 计算交集
    x1 = np.maximum(boxes1[:, 0].reshape(-1, 1), boxes2[:, 0].reshape(1, -1))
    y1 = np.maximum(boxes1[:, 1].reshape(-1, 1), boxes2[:, 1].reshape(1, -1))
    x2 = np.minimum(boxes1[:, 2].reshape(-1, 1), boxes2[:, 2].reshape(1, -1))
    y2 = np.minimum(boxes1[:, 3].reshape(-1, 1), boxes2[:, 3].reshape(1, -1))
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # 计算并集
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1.reshape(-1, 1) + area2.reshape(1, -1) - intersection
    
    return intersection / (union + 1e-10)

def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.5):
    """
    Weighted Boxes Fusion (WBF): 多模型/多尺度结果融合算法
    
    参考: https://arxiv.org/abs/1910.13302
    
    Args:
        boxes_list: 多个来源的框列表，每个是 [N, 4] 数组
        scores_list: 对应的置信度列表
        labels_list: 对应的类别列表
        weights: 每个来源的权重
        iou_thr: 匹配IoU阈值
    
    Returns:
        fused_boxes, fused_scores, fused_labels
    """
    if weights is None:
        weights = np.ones(len(boxes_list))
    
    # 收集所有框
    all_boxes = []
    all_scores = []
    all_labels = []
    all_sources = []
    
    for i, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for j in range(len(boxes)):
            all_boxes.append(boxes[j])
            all_scores.append(scores[j] * weights[i])
            all_labels.append(labels[j])
            all_sources.append(i)
    
    if len(all_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 按类别分组融合
    fused_boxes = []
    fused_scores = []
    fused_labels = []
    
    unique_labels = np.unique(all_labels)
    
    for label in unique_labels:
        mask = all_labels == label
        label_boxes = all_boxes[mask]
        label_scores = all_scores[mask]
        
        # 使用聚类方法融合框
        if len(label_boxes) == 0:
            continue
        
        # 简单的加权平均（按置信度加权）
        weights_norm = label_scores / label_scores.sum()
        fused_box = np.average(label_boxes, axis=0, weights=weights_norm)
        fused_score = label_scores.max()  # 使用最大置信度
        
        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)
        fused_labels.append(label)
    
    return np.array(fused_boxes), np.array(fused_scores), np.array(fused_labels)

def run_tta_inference(model, image_path, scales=[960, 1280, 1600], 
                      conf=0.25, iou=0.45, use_flip=True):
    """
    执行TTA推理
    
    Args:
        model: YOLO模型
        image_path: 图像路径
        scales: 多尺度尺寸列表
        conf: 置信度阈值
        iou: NMS IoU阈值
        use_flip: 是否使用水平翻转
    
    Returns:
        融合后的检测结果
    """
    from ultralytics import YOLO
    
    # 读取原始图像
    img = cv2.imread(str(image_path))
    if img is None:
        return [], [], []
    
    h, w = img.shape[:2]
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    # 多尺度推理
    for scale in scales:
        # 原始尺度推理
        results = model(img, conf=conf, iou=iou, imgsz=scale, verbose=False)
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                labels = r.boxes.cls.cpu().numpy().astype(int)
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)
        
        # 水平翻转推理
        if use_flip:
            img_flipped = cv2.flip(img, 1)
            results_flip = model(img_flipped, conf=conf, iou=iou, imgsz=scale, verbose=False)
            for r in results_flip:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    # 翻转框坐标回原始图像
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    scores = r.boxes.conf.cpu().numpy()
                    labels = r.boxes.cls.cpu().numpy().astype(int)
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_labels.append(labels)
    
    # 合并所有结果
    if len(all_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    all_boxes = np.vstack(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    # 按类别进行NMS
    final_boxes = []
    final_scores = []
    final_labels = []
    
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        mask = all_labels == label
        boxes = all_boxes[mask]
        scores = all_scores[mask]
        
        # Soft-NMS
        keep = soft_nms(boxes, scores, iou_threshold=iou, method='gaussian')
        
        final_boxes.extend(boxes[keep])
        final_scores.extend(scores[keep])
        final_labels.extend([label] * len(keep))
    
    return np.array(final_boxes), np.array(final_scores), np.array(final_labels)

def main():
    args = parse_args()
    
    print("="*70)
    print("TTA Inference for PCB Defect Detection")
    print("="*70)
    
    # 导入ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        sys.exit(1)
    
    # 设置设备
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 加载模型
    print(f"\nLoading model: {args.weights}")
    model = YOLO(args.weights)
    model.to(device)
    
    # 准备输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取测试图像
    source_dir = Path(args.source)
    image_files = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
    print(f"\nFound {len(image_files)} images")
    
    # TTA配置
    if args.tta:
        print(f"\nTTA Configuration:")
        print(f"  Scales: {args.scales}")
        print(f"  Horizontal flip: {args.flip}")
        print(f"  Merge IoU threshold: {args.merge_iou}")
    
    # 推理
    print("\nRunning inference...")
    all_results = []
    
    for img_path in tqdm(image_files, desc="Processing"):
        if args.tta:
            # TTA推理
            boxes, scores, labels = run_tta_inference(
                model, img_path, 
                scales=args.scales,
                conf=args.conf,
                iou=args.iou,
                use_flip=args.flip
            )
        else:
            # 普通推理
            results = model(img_path, conf=args.conf, iou=args.iou, verbose=False)
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    labels = r.boxes.cls.cpu().numpy().astype(int)
                else:
                    boxes, scores, labels = np.array([]), np.array([]), np.array([])
        
        # 保存结果
        result = {
            'image': img_path.name,
            'boxes': boxes.tolist() if len(boxes) > 0 else [],
            'scores': scores.tolist() if len(scores) > 0 else [],
            'labels': labels.tolist() if len(labels) > 0 else [],
            'num_detections': len(boxes)
        }
        all_results.append(result)
    
    # 保存结果
    results_file = output_dir / 'tta_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Total detections: {sum(r['num_detections'] for r in all_results)}")

if __name__ == "__main__":
    main()
