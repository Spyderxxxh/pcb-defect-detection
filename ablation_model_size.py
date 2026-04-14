"""
YOLOv8 模型尺寸消融实验
对比 n/s/m/l/x 的精度 vs 速度 Pareto 曲线
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


# DeepPCB 6类缺陷
CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']

# mAP@0.5:0.95 使用的 IoU 阈值序列
MAP_IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 模型尺寸消融实验')
    parser.add_argument('--model-size', type=str, nargs='+',
                        default=['n', 's', 'm', 'l', 'x'],
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='要测试的模型尺寸')
    parser.add_argument('--weights-dir', type=str, 
                        default='runs/detect/runs/detect',
                        help='训练权重目录（优先查找微调权重）')
    parser.add_argument('--data', type=str, 
                        default='./data/deeppcb/yolo_format',
                        help='数据集路径')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='推理分辨率')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU设备ID')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU阈值')
    parser.add_argument('--save-json', action='store_true',
                        help='保存JSON结果')
    return parser.parse_args()


def find_model_weights(size):
    """查找指定尺寸的模型权重 - 使用预训练模型"""
    # 始终使用对应尺寸的预训练模型（这是消融实验的正确方式）
    pretrained_model = f'yolov8{size}.pt'
    print(f"  使用预训练模型: {pretrained_model}")
    return pretrained_model


def compute_iou(box1, box2):
    """计算两个 xyxy 框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def voc_ap(rec, prec):
    """VOC 2010+ AP（面积法）"""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


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


def evaluate_model(model_path, img_dir, label_dir, imgsz, conf, iou, device):
    """评估单个模型"""
    
    # 加载模型
    model = YOLO(model_path)
    model.to(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    print(f"  评估 {len(image_files)} 张图像...")
    
    # GPU预热
    dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(3):
        _ = model(dummy_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 收集全局检测和GT
    global_dets = []
    global_gts = []
    total_time = 0
    
    for img_path in tqdm(image_files, desc="    Processing", leave=False):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        img_id = img_path.stem
        
        # 推理
        start = time.time()
        results = model(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time += time.time() - start
        
        # 解析检测结果
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                det = {
                    'img_id': img_id,
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf),
                    'class_id': int(box.cls)
                }
                global_dets.append(det)
        
        # 加载GT
        label_path = label_dir / f"{img_path.stem}.txt"
        ground_truths = load_ground_truths(label_path, w, h)
        for gt in ground_truths:
            global_gts.append({**gt, 'img_id': img_id})
    
    # 计算 mAP@0.5
    ap_list = []
    for i, name in enumerate(CLASS_NAMES):
        cls_dets = [d for d in global_dets if d['class_id'] == i]
        cls_gts = [g for g in global_gts if g['class_id'] == i]
        
        if len(cls_gts) == 0:
            continue
        
        gt_by_img = {}
        for g in cls_gts:
            gt_by_img.setdefault(g['img_id'], []).append(g)
        
        cls_dets_sorted = sorted(cls_dets, key=lambda x: x['confidence'], reverse=True)
        matched = {}
        tp_arr = np.zeros(len(cls_dets_sorted))
        fp_arr = np.zeros(len(cls_dets_sorted))
        
        for k, det in enumerate(cls_dets_sorted):
            gts_img = gt_by_img.get(det['img_id'], [])
            m_set = matched.setdefault(det['img_id'], set())
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts_img):
                if j in m_set:
                    continue
                iou_val = compute_iou(det['bbox'], g['bbox'])
                if iou_val > best_iou:
                    best_iou, best_j = iou_val, j
            if best_iou >= 0.5 and best_j >= 0:
                m_set.add(best_j)
                tp_arr[k] = 1
            else:
                fp_arr[k] = 1
        
        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)
        rec = tp_cum / (len(cls_gts) + 1e-10)
        prec = tp_cum / (tp_cum + fp_cum + 1e-10)
        ap_list.append(voc_ap(rec, prec))
    
    map50 = float(np.mean(ap_list)) if ap_list else 0.0
    
    # 计算 mAP@0.5:0.95
    per_class_ap = {}
    for i, name in enumerate(CLASS_NAMES):
        cls_dets = [d for d in global_dets if d['class_id'] == i]
        cls_gts = [g for g in global_gts if g['class_id'] == i]
        
        if len(cls_gts) == 0:
            per_class_ap[name] = 0.0
            continue
        
        ap_iou_list = []
        for thr in MAP_IOU_THRESHOLDS:
            gt_by_img = {}
            for g in cls_gts:
                gt_by_img.setdefault(g['img_id'], []).append(g)
            
            cls_dets_sorted = sorted(cls_dets, key=lambda x: x['confidence'], reverse=True)
            matched = {}
            tp_arr = np.zeros(len(cls_dets_sorted))
            fp_arr = np.zeros(len(cls_dets_sorted))
            
            for k, det in enumerate(cls_dets_sorted):
                gts_img = gt_by_img.get(det['img_id'], [])
                m_set = matched.setdefault(det['img_id'], set())
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gts_img):
                    if j in m_set:
                        continue
                    iou_val = compute_iou(det['bbox'], g['bbox'])
                    if iou_val > best_iou:
                        best_iou, best_j = iou_val, j
                if best_iou >= thr and best_j >= 0:
                    m_set.add(best_j)
                    tp_arr[k] = 1
                else:
                    fp_arr[k] = 1
            
            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(fp_arr)
            rec = tp_cum / (len(cls_gts) + 1e-10)
            prec = tp_cum / (tp_cum + fp_cum + 1e-10)
            ap_iou_list.append(voc_ap(rec, prec))
        
        per_class_ap[name] = float(np.mean(ap_iou_list)) if ap_iou_list else 0.0
    
    map50_95 = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    
    avg_inference_time = total_time / len(image_files) if image_files else 0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    return {
        'model_size': model_path.replace('.pt', ''),
        'weights': model_path,
        'map50': map50,
        'map50_95': map50_95,
        'per_class_ap50': {name: ap_list[i] if i < len(ap_list) else 0 for i, name in enumerate(CLASS_NAMES)},
        'per_class_ap50_95': per_class_ap,
        'fps': fps,
        'num_images': len(image_files)
    }


def main():
    args = parse_args()
    
    print("="*60)
    print("YOLOv8 模型尺寸消融实验")
    print("="*60)
    print(f"模型尺寸: {args.model_size}")
    print(f"推理分辨率: {args.imgsz}")
    print(f"权重目录: {args.weights_dir}")
    print(f"数据: {args.data}")
    print(f"设备: cuda:{args.device}")
    print("="*60)
    
    # 设置设备
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 验证数据路径
    img_dir = Path(args.data) / 'images' / 'val'
    label_dir = Path(args.data) / 'labels' / 'val'
    
    if not img_dir.exists():
        print(f"错误: 图像目录不存在: {img_dir}")
        sys.exit(1)
    
    if not label_dir.exists():
        print(f"错误: 标签目录不存在: {label_dir}")
        sys.exit(1)
    
    print(f"验证集: {img_dir}")
    
    # 运行消融实验
    results = []
    for size in args.model_size:
        print(f"\n{'='*40}")
        print(f"测试模型: YOLOv8{size}")
        print(f"{'='*40}")
        
        # 查找权重 - 使用对应尺寸的预训练模型
        model_path = find_model_weights(size)
        print(f"  使用权重: {model_path}")
        
        result = evaluate_model(
            model_path=model_path,
            img_dir=img_dir,
            label_dir=label_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device
        )
        
        result['model_size'] = f'yolov8{size}'
        results.append(result)
        
        print(f"\n  mAP@0.5:       {result['map50']*100:.2f}%")
        print(f"  mAP@0.5:0.95:  {result['map50_95']*100:.2f}%")
        print(f"  FPS:           {result['fps']:.1f}")
    
    # 打印汇总表格
    print("\n" + "="*70)
    print("模型尺寸消融实验结果汇总")
    print("="*70)
    print(f"{'模型':>10} | {'mAP@0.5':>10} | {'mAP@0.5:0.95':>12} | {'FPS':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['model_size']:>10} | {r['map50']*100:>9.2f}% | {r['map50_95']*100:>11.2f}% | {r['fps']:>8.1f}")
    print("="*70)
    
    # 打印各类别 AP@0.5
    print("\n各类别 AP@0.5:")
    header = f"{'模型':>10}"
    for name in CLASS_NAMES:
        header += f" | {name:>8}"
    print(header)
    print("-"*len(header))
    for r in results:
        row = f"{r['model_size']:>10}"
        for name in CLASS_NAMES:
            row += f" | {r['per_class_ap50'][name]*100:>7.2f}%"
        print(row)
    
    # 计算 Pareto 最优
    print("\n" + "="*70)
    print("Pareto 最优分析")
    print("="*70)
    
    # 按精度排序找 Pareto 最优
    pareto_optimal = []
    for i, r1 in enumerate(results):
        is_pareto = True
        for j, r2 in enumerate(results):
            if i != j:
                # r2 在两个指标上都不比 r1 差，且至少一个更好
                if (r2['map50'] >= r1['map50'] and r2['map50_95'] >= r1['map50_95'] and 
                    (r2['map50'] > r1['map50'] or r2['map50_95'] > r1['map50_95'])):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_optimal.append(r1['model_size'])
    
    print(f"Pareto 最优解: {', '.join(pareto_optimal)}")
    print("\n推荐选择:")
    for r in results:
        if r['model_size'] in pareto_optimal:
            print(f"  {r['model_size']}: mAP@0.5={r['map50']*100:.2f}%, mAP@0.5:0.95={r['map50_95']*100:.2f}%, FPS={r['fps']:.1f}")
    
    # 保存JSON
    if args.save_json:
        output_file = 'model_size_ablation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment': 'model size ablation',
                'imgsz': args.imgsz,
                'data': args.data,
                'results': results,
                'pareto_optimal': pareto_optimal
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    main()