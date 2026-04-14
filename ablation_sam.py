"""
SAM 分割细化消融实验
对比有/无 SAM 的 mAP 差异
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
    parser = argparse.ArgumentParser(description='SAM 分割细化消融实验')
    parser.add_argument('--weights', type=str, required=True,
                        help='YOLO 模型权重路径')
    parser.add_argument('--sam-type', type=str, default='vit_b',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM 模型类型')
    parser.add_argument('--sam-checkpoint', type=str, default=None,
                        help='SAM 权重路径（可选，默认下载 vit_b）')
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


def load_sam(sam_type, checkpoint_path):
    """加载 SAM 模型"""
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("Error: segment_anything not installed")
        print("Please install: pip install segment-anything")
        sys.exit(1)
    
    if checkpoint_path:
        sam = sam_model_registry[sam_type](checkpoint=checkpoint_path)
    else:
        # 自动下载默认权重
        print(f"  自动下载 SAM {sam_type} 权重...")
        sam = sam_model_registry[sam_type](checkpoint=None)
    
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor


def refine_with_sam(predictor, image, boxes):
    """使用 SAM 精修边界框"""
    if len(boxes) == 0:
        return []
    
    refined_boxes = []
    
    # 设置图像
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    predictor.set_image(image)
    
    for box in boxes:
        try:
            # SAM 预测
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box.reshape(1, -1),
                multimask_output=False
            )
            
            mask = masks[0] if isinstance(masks, (list, tuple)) else masks
            
            # 从 mask 提取边界框
            mask_uint8 = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                all_points = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                refined_boxes.append([x, y, x + w, y + h])
            else:
                refined_boxes.append(box.tolist())
        except Exception as e:
            # SAM 失败，保留原始框
            refined_boxes.append(box.tolist())
    
    return refined_boxes


def evaluate_with_sam(model, predictor, img_dir, label_dir, imgsz, conf, iou, device, use_sam):
    """评估模型（可选是否使用 SAM 细化）"""
    
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
    sam_time = 0
    
    for img_path in tqdm(image_files, desc="    Processing", leave=False):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        img_id = img_path.stem
        
        # YOLOv8 推理
        start = time.time()
        results = model(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        yolo_time = time.time() - start
        total_time += yolo_time
        
        # 解析检测结果
        detections = []
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
                detections.append(det)
        
        # SAM 细化（如果启用）
        if use_sam and predictor and len(detections) > 0:
            sam_start = time.time()
            
            # 按类别分别处理
            for cls_id in set(d['class_id'] for d in detections):
                cls_dets = [d for d in detections if d['class_id'] == cls_id]
                boxes = np.array([d['bbox'] for d in cls_dets])
                
                # 转换为列表格式
                box_list = [b.tolist() if hasattr(b, 'tolist') else b for b in boxes]
                
                refined = refine_with_sam(predictor, img, np.array(box_list))
                
                # 更新检测框
                for i, det in enumerate(cls_dets):
                    if i < len(refined):
                        det['bbox'] = refined[i]
            
            sam_time += time.time() - sam_start
        
        # 添加到全局
        for det in detections:
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
        'use_sam': use_sam,
        'map50': map50,
        'map50_95': map50_95,
        'per_class_ap50': {name: ap_list[i] if i < len(ap_list) else 0 for i, name in enumerate(CLASS_NAMES)},
        'per_class_ap50_95': per_class_ap,
        'fps': fps,
        'num_images': len(image_files),
        'total_time': total_time,
        'sam_time': sam_time if use_sam else 0
    }


def main():
    args = parse_args()
    
    print("="*60)
    print("SAM 分割细化消融实验")
    print("="*60)
    print(f"YOLO 权重: {args.weights}")
    print(f"SAM 类型: {args.sam_type}")
    print(f"数据: {args.data}")
    print(f"推理分辨率: {args.imgsz}")
    print(f"设备: cuda:{args.device}")
    print("="*60)
    
    # 设置设备
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载 YOLO 模型
    print("\n加载 YOLO 模型...")
    model = YOLO(args.weights)
    model.to(device)
    
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
    
    # ========== 实验 1: 无 SAM ==========
    print(f"\n{'='*40}")
    print("实验 1: 无 SAM 细化")
    print(f"{'='*40}")
    
    result_no_sam = evaluate_with_sam(
        model=model,
        predictor=None,
        img_dir=img_dir,
        label_dir=label_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        use_sam=False
    )
    
    print(f"\n  mAP@0.5:       {result_no_sam['map50']*100:.2f}%")
    print(f"  mAP@0.5:0.95:  {result_no_sam['map50_95']*100:.2f}%")
    print(f"  FPS:           {result_no_sam['fps']:.1f}")
    
    # ========== 实验 2: 有 SAM ==========
    print(f"\n{'='*40}")
    print("实验 2: 有 SAM 细化")
    print(f"{'='*40}")
    
    print("\n加载 SAM 模型...")
    predictor = load_sam(args.sam_type, args.sam_checkpoint)
    print("SAM 模型加载完成")
    
    result_with_sam = evaluate_with_sam(
        model=model,
        predictor=predictor,
        img_dir=img_dir,
        label_dir=label_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        use_sam=True
    )
    
    print(f"\n  mAP@0.5:       {result_with_sam['map50']*100:.2f}%")
    print(f"  mAP@0.5:0.95:  {result_with_sam['map50_95']*100:.2f}%")
    print(f"  FPS:           {result_with_sam['fps']:.1f}")
    print(f"  SAM 耗时:      {result_with_sam['sam_time']:.2f}s")
    
    # ========== 结果对比 ==========
    print("\n" + "="*70)
    print("SAM 消融实验结果汇总")
    print("="*70)
    
    print(f"\n{'模式':>12} | {'mAP@0.5':>10} | {'mAP@0.5:0.95':>12} | {'FPS':>8}")
    print("-"*70)
    print(f"{'无 SAM':>12} | {result_no_sam['map50']*100:>9.2f}% | {result_no_sam['map50_95']*100:>11.2f}% | {result_no_sam['fps']:>8.1f}")
    print(f"{'有 SAM':>12} | {result_with_sam['map50']*100:>9.2f}% | {result_with_sam['map50_95']*100:>11.2f}% | {result_with_sam['fps']:>8.1f}")
    print("-"*70)
    
    delta_map50 = (result_with_sam['map50'] - result_no_sam['map50']) * 100
    delta_map50_95 = (result_with_sam['map50_95'] - result_no_sam['map50_95']) * 100
    print(f"{'Δ (SAM)':>12} | {delta_map50:>+9.2f}pp | {delta_map50_95:>+11.2f}pp |")
    print("="*70)
    
    # 打印各类别对比
    print("\n各类别 AP@0.5 对比:")
    header = f"{'类别':>12} | {'无 SAM':>10} | {'有 SAM':>10} | {'Δ':>8}"
    print(header)
    print("-"*len(header))
    for name in CLASS_NAMES:
        no_sam = result_no_sam['per_class_ap50'][name] * 100
        with_sam = result_with_sam['per_class_ap50'][name] * 100
        delta = with_sam - no_sam
        print(f"{name:>12} | {no_sam:>9.2f}% | {with_sam:>9.2f}% | {delta:>+7.2f}pp")
    
    # 保存JSON
    if args.save_json:
        output_file = 'sam_ablation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment': 'SAM refinement ablation',
                'yolo_weights': args.weights,
                'sam_type': args.sam_type,
                'no_sam': result_no_sam,
                'with_sam': result_with_sam,
                'delta_map50': delta_map50,
                'delta_map50_95': delta_map50_95
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    main()