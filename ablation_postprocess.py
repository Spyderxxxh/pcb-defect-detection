"""
后处理方法消融实验

对比：
1. 无后处理 (baseline)
2. 传统后处理 (NMS + 置信度阈值)
3. LevelSet 边界细化
4. SAM + LevelSet 细化

用法:
    python ablation_postprocess.py \
        --weights runs/detect/runs/detect/defectguard_v2_ultra/weights/best.pt \
        --data ./data/deeppcb/yolo_format \
        --imgsz 1280 \
        --device 0 \
        --save-json
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2

from src.models.sam_refinement import SAMRefinement, LevelSetRefinement


def parse_args():
    parser = argparse.ArgumentParser(description='后处理方法消融实验')
    parser.add_argument('--weights', type=str, required=True, help='YOLO 权重路径')
    parser.add_argument('--data', type=str, required=True, help='数据配置目录')
    parser.add_argument('--imgsz', type=int, default=1280, help='推理分辨率')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--conf', type=float, default=0.001, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--save-json', action='store_true', help='保存JSON结果')
    return parser.parse_args()


def calculate_iou(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def nms(boxes, scores, iou_threshold):
    """NMS 非极大值抑制"""
    if len(boxes) == 0:
        return []
    
    # 按置信度排序
    order = np.argsort(scores)[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # 计算与最高分框的 IoU
        ious = [calculate_iou(boxes[i], boxes[j]) for j in order[1:]]
        
        # 保留 IoU 小于阈值的框
        mask = np.array(ious) < iou_threshold
        order = order[1:][mask]
    
    return keep


def compute_ap(recalls, precisions):
    """计算 AP"""
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))
    
    # 插值
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def evaluate_detections(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """评估检测结果"""
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0
    
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0
    
    # 匹配
    matched_gt = set()
    tp = []
    fp = []
    
    for pred_box, pred_score, pred_label in zip(pred_boxes, pred_scores, pred_labels):
        max_iou = 0
        max_idx = -1
        
        for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if idx in matched_gt or pred_label != gt_label:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_idx = idx
        
        if max_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched_gt.add(max_idx)
        else:
            tp.append(0)
            fp.append(1)
    
    tp = np.array(tp)
    fp = np.array(fp)
    
    if len(tp) == 0:
        return 0.0, 0.0, 0.0
    
    # 计算 AP
    sorted_idx = np.argsort(pred_scores)[::-1]
    tp = tp[sorted_idx]
    fp = fp[sorted_idx]
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    ap = compute_ap(recalls, precisions)
    
    return ap, np.sum(tp), np.sum(fp)


def load_gt_boxes(label_dir, image_name):
    """加载 GT 框"""
    label_path = Path(label_dir) / f"{image_name}.txt"
    
    if not label_path.exists():
        return [], []
    
    boxes = []
    labels = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls = int(parts[0])
            # YOLO 格式转换为 xyxy
            cx, cy, w, h = map(float, parts[1:5])
            
            # 假设图像大小为 640x640（需要根据实际情况调整）
            img_w, img_h = 640, 640
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
    
    return boxes, labels


def main():
    args = parse_args()
    
    # 处理设备
    device = args.device if 'cuda' in args.device else f"cuda:{args.device}"
    
    print("=" * 60)
    print("后处理方法消融实验")
    print("=" * 60)
    print(f"权重: {args.weights}")
    print(f"数据: {args.data}")
    print(f"推理分辨率: {args.imgsz}")
    print(f"设备: {device}")
    print()
    
    # 加载模型
    print("加载 YOLO 模型...")
    yolo = YOLO(args.weights)
    
    # 初始化后处理器
    print("初始化后处理器...")
    sam_refine = SAMRefinement(model_type='vit_b')
    sam_refine.eval()
    sam_refine = sam_refine.to(device)
    
    level_set = LevelSetRefinement(max_iter=50)
    
    # 获取验证集
    val_images_dir = Path(args.data) / 'images' / 'val'
    val_labels_dir = Path(args.data) / 'labels' / 'val'
    
    image_files = sorted(list(val_images_dir.glob('*.jpg')))
    print(f"验证集图像数量: {len(image_files)}")
    
    # 存储各方法的结果
    results = {
        'baseline': {'aps': [], 'tps': [], 'fps': []},
        'nms': {'aps': [], 'tps': [], 'fps': []},
        'levelset': {'aps': [], 'tps': [], 'fps': []},
        'sam_levelset': {'aps': [], 'tps': [], 'fps': []},
    }
    
    for img_path in image_files[:100]:  # 快速测试 100 张
        img_name = img_path.stem
        
        # 加载 GT
        gt_boxes, gt_labels = load_gt_boxes(val_labels_dir, img_name)
        
        # YOLO 推理
        img = cv2.imread(str(img_path))
        results_yolo = yolo.predict(img, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        
        # 原始检测结果
        boxes = results_yolo.boxes.xyxy.cpu().numpy()
        scores = results_yolo.boxes.conf.cpu().numpy()
        labels = results_yolo.boxes.cls.cpu().numpy()
        
        # 方法1: Baseline (无后处理)
        ap, tp, fp = evaluate_detections(boxes, scores, labels, gt_boxes, gt_labels)
        results['baseline']['aps'].append(ap)
        results['baseline']['tps'].append(tp)
        results['baseline']['fps'].append(fp)
        
        # 方法2: NMS
        if len(boxes) > 0:
            keep = nms(boxes, scores, args.iou)
            boxes_nms = boxes[keep]
            scores_nms = scores[keep]
            labels_nms = labels[keep]
        else:
            boxes_nms, scores_nms, labels_nms = [], [], []
        
        ap, tp, fp = evaluate_detections(boxes_nms, scores_nms, labels_nms, gt_boxes, gt_labels)
        results['nms']['aps'].append(ap)
        results['nms']['tps'].append(tp)
        results['nms']['fps'].append(fp)
        
        # 方法3: LevelSet
        if len(boxes) > 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes_ls = []
            for box in boxes:
                refined = level_set.refine(img_rgb, tuple(map(int, box)))
                boxes_ls.append(refined)
            boxes_ls = np.array(boxes_ls)
        else:
            boxes_ls = boxes
        
        ap, tp, fp = evaluate_detections(boxes_ls, scores, labels, gt_boxes, gt_labels)
        results['levelset']['aps'].append(ap)
        results['levelset']['tps'].append(tp)
        results['levelset']['fps'].append(fp)
        
        # 方法4: SAM + LevelSet
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)
        
        if len(boxes) > 0:
            refined_results = sam_refine.forward(img_tensor, boxes_tensor)
            boxes_sam = np.array([r['box'] for r in refined_results])
        else:
            boxes_sam = boxes
        
        ap, tp, fp = evaluate_detections(boxes_sam, scores, labels, gt_boxes, gt_labels)
        results['sam_levelset']['aps'].append(ap)
        results['sam_levelset']['tps'].append(tp)
        results['sam_levelset']['fps'].append(fp)
    
    # 汇总结果
    print()
    print("=" * 60)
    print("消融实验结果汇总")
    print("=" * 60)
    
    print(f"\n{'方法':<20} {'mAP@0.5':>12} {'TP':>8} {'FP':>8}")
    print("-" * 60)
    
    for method, data in results.items():
        map50 = np.mean(data['aps']) * 100
        tp_sum = sum(data['tps'])
        fp_sum = sum(data['fps'])
        print(f"{method:<20} {map50:>11.2f}% {tp_sum:>8} {fp_sum:>8}")
    
    # 保存结果
    if args.save_json:
        output = {
            'args': vars(args),
            'results': {k: {kk: list(vv) for kk, vv in v.items()} for k, v in results.items()},
            'summary': {
                method: {
                    'map50': float(np.mean(data['aps']) * 100),
                    'total_tp': sum(data['tps']),
                    'total_fp': sum(data['fps'])
                }
                for method, data in results.items()
            }
        }
        
        output_path = 'postprocess_ablation_results.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n结果已保存至: {output_path}")


if __name__ == '__main__':
    main()