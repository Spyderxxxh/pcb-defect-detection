"""
模型评估脚本
计算mAP、精度、召回率等指标
支持三种性能模式对比
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import torch
from ultralytics import YOLO

# 添加项目路径
sys.path.append('.')
from src.core.config import load_config
from src.models.sam_refinement import SAMRefinement


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DefectGuard Model')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data', type=str, required=True,
                        help='验证集目录或yaml配置文件')
    parser.add_argument('--mode', type=str, default='balanced',
                        choices=['high_precision', 'balanced', 'high_speed'],
                        help='性能模式')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU设备ID')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU阈值')
    parser.add_argument('--save-results', action='store_true',
                        help='保存详细结果')
    return parser.parse_args()


def compute_iou(box1, box2):
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def evaluate_detections(detections, ground_truths, iou_threshold=0.5):
    """
    评估检测结果
    
    Returns:
        precision, recall, f1, ap
    """
    if len(detections) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    if len(ground_truths) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    matched_gt = set()
    tp_list = []
    conf_list = []
    
    for det in detections:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truths):
            if i in matched_gt:
                continue
            
            iou = compute_iou(det['bbox'], gt['bbox'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = i
        
        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            tp_list.append(1)
        else:
            tp_list.append(0)
        
        conf_list.append(det['confidence'])
    
    # 计算Precision和Recall
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum([1 - tp for tp in tp_list])
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / len(ground_truths)
    
    # 计算AP（使用11点插值）
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    # 最终Precision和Recall
    precision = precisions[-1] if len(precisions) > 0 else 0
    recall = recalls[-1] if len(recalls) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return precision, recall, f1, ap


def load_ground_truths(label_path, img_width, img_height):
    """加载YOLO格式的标注"""
    gts = []
    
    if not os.path.exists(label_path):
        return gts
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height
                
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                
                gts.append({
                    'class_id': cls_id,
                    'bbox': [x1, y1, x2, y2]
                })
    
    return gts


def main():
    args = parse_args()
    
    print("="*60)
    print("DefectGuard Model Evaluation")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Weights: {args.weights}")
    print(f"Data: {args.data}")
    
    # 设置设备
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 加载模型
    print("\nLoading model...")
    model = YOLO(args.weights)
    model.to(device)
    
    # 根据模式设置参数
    if args.mode == 'high_precision':
        conf, iou = 0.5, 0.65  # 提高conf减少FP，降低iou保留更多TP
        imgsz = 640
        half = False
    elif args.mode == 'balanced':
        conf, iou = 0.5, 0.65
        imgsz = 640
        half = False
    else:  # high_speed
        conf, iou = 0.3, 0.45
        imgsz = 480  # 降低分辨率
        half = True  # FP16半精度
    
    print(f"Parameters: conf={conf}, iou={iou}, imgsz={imgsz}, half={half}")
    
    # 查找验证图像
    if os.path.isdir(args.data):
        # 假设是目录结构 data/images/val 和 data/labels/val
        img_dir = Path(args.data) / 'images' / 'val'
        label_dir = Path(args.data) / 'labels' / 'val'
    else:
        print(f"Error: Data path not found: {args.data}")
        return
    
    if not img_dir.exists():
        # 尝试直接作为图像目录
        img_dir = Path(args.data)
        label_dir = Path(args.data).parent / 'labels' / 'val'
    
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    print(f"\nFound {len(image_files)} validation images")
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    # GPU预热 - 运行几次推理稳定GPU
    print("\nWarming up GPU...")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(dummy_img, conf=conf, iou=iou, imgsz=imgsz, half=half, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup complete")
    
    # 评估统计
    all_results = []
    total_inference_time = 0
    
    # 按类别统计
    class_names = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
    class_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(len(class_names))}
    
    print("\nEvaluating...")
    for img_path in tqdm(image_files, desc="Processing"):
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # 推理 - 多次采样取平均
        inference_times = []
        for _ in range(3):  # 每张图跑3次取平均
            start = time.time()
            results = model(img, conf=conf, iou=iou, imgsz=imgsz, half=half, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.time() - start)
        inference_time = np.mean(inference_times)  # 取平均
        total_inference_time += inference_time
        
        # 解析检测结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf),
                    'class_id': int(box.cls)
                })
        
        # 加载Ground Truth
        label_path = label_dir / f"{img_path.stem}.txt"
        ground_truths = load_ground_truths(label_path, w, h)
        
        # 评估
        precision, recall, f1, ap = evaluate_detections(detections, ground_truths)
        
        all_results.append({
            'image': img_path.name,
            'detections': len(detections),
            'ground_truths': len(ground_truths),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap,
            'inference_time': inference_time
        })
        
        # 更新类别统计
        matched_gt = set()
        for det in detections:
            cls_id = det['class_id']
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(ground_truths):
                if i in matched_gt or gt['class_id'] != cls_id:
                    continue
                
                iou = compute_iou(det['bbox'], gt['bbox'])
                if iou > best_iou and iou >= 0.5:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_gt_idx >= 0:
                class_stats[cls_id]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                class_stats[cls_id]['fp'] += 1
        
        # 统计FN
        for i, gt in enumerate(ground_truths):
            if i not in matched_gt:
                class_stats[gt['class_id']]['fn'] += 1
    
    # 计算总体指标
    avg_precision = np.mean([r['precision'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_f1 = np.mean([r['f1'] for r in all_results])
    map50 = np.mean([r['ap'] for r in all_results])
    avg_inference_time = total_inference_time / len(all_results)
    fps = 1.0 / avg_inference_time
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Parameters: conf={conf}, iou={iou}, imgsz={imgsz}, half={half}")
    print(f"Images evaluated: {len(all_results)}")
    print()
    print("Overall Metrics:")
    print(f"  mAP@0.5:    {map50:.4f}")
    print(f"  Precision:  {avg_precision:.4f}")
    print(f"  Recall:     {avg_recall:.4f}")
    print(f"  F1-Score:   {avg_f1:.4f}")
    print()
    print("Performance:")
    print(f"  Avg inference time: {avg_inference_time*1000:.2f}ms")
    print(f"  FPS: {fps:.1f}")
    print()
    print("Per-Class Metrics:")
    for cls_id, name in enumerate(class_names):
        stats = class_stats[cls_id]
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        print(f"  {name:12s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, "
              f"TP={tp}, FP={fp}, FN={fn}")
    
    # 保存结果
    if args.save_results:
        output_file = f'evaluation_{args.mode}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'mode': args.mode,
                'metrics': {
                    'map50': map50,
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1': avg_f1,
                    'fps': fps
                },
                'per_class': {
                    name: {
                        'precision': class_stats[i]['tp'] / (class_stats[i]['tp'] + class_stats[i]['fp'] + 1e-10),
                        'recall': class_stats[i]['tp'] / (class_stats[i]['tp'] + class_stats[i]['fn'] + 1e-10),
                        'tp': class_stats[i]['tp'],
                        'fp': class_stats[i]['fp'],
                        'fn': class_stats[i]['fn']
                    }
                    for i, name in enumerate(class_names)
                },
                'details': all_results
            }, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
