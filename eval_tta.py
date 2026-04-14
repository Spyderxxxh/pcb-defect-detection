"""
TTA 推理结果评估脚本
读取 inference_tta.py 输出的 JSON 结果，计算 mAP@0.5 和 mAP@0.5:0.95
并与基线数字进行对比

Usage:
    # 评估 TTA 结果
    python eval_tta.py \
        --pred outputs/tta_results/tta_results.json \
        --label-dir data/deeppcb/yolo_format/labels/test \
        --image-dir data/deeppcb/yolo_format/images/test

    # 评估 TTA 结果并与基线对比
    python eval_tta.py \
        --pred outputs/tta_results/tta_results.json \
        --label-dir data/deeppcb/yolo_format/labels/test \
        --image-dir data/deeppcb/yolo_format/images/test \
        --baseline-map50 0.9587 \
        --baseline-map50-95 0.7112

    # 同时跑一个新模型的普通推理做对比（可选）
    python eval_tta.py \
        --pred outputs/tta_results/tta_results.json \
        --label-dir data/deeppcb/yolo_format/labels/test \
        --image-dir data/deeppcb/yolo_format/images/test \
        --compare-weights runs/detect/runs/detect/defectguard_v2_ultra/weights/best.pt
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2


# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)   # 0.50, 0.55, ..., 0.95（共10个）

# 基线数字（直接硬编码，方便对比）
BASELINE = {
    'map50':    0.9587,   # test 集
    'map50_95': 0.7112,
    'fps':      187.9,
    'per_class_ap50': {
        'open':      0.9687,
        'short':     0.9084,
        'mousebite': 0.9718,
        'spur':      0.9533,
        'copper':    0.9798,
        'pinhole':   0.9700,
    },
    'per_class_ap50_95': {
        'open':      0.6286,
        'short':     0.6182,
        'mousebite': 0.7033,
        'spur':      0.6917,
        'copper':    0.8281,
        'pinhole':   0.7974,
    }
}


# ──────────────────────────────────────────────────────────────
# AP 计算（与 baseline_compare.py 保持一致）
# ──────────────────────────────────────────────────────────────
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


def compute_ap_at_iou(all_dets, all_gts, cls_id, iou_thr):
    cls_dets = [d for d in all_dets if d['class_id'] == cls_id]
    cls_gts  = [g for g in all_gts  if g['class_id'] == cls_id]

    if len(cls_gts) == 0:
        return 0.0

    gt_by_img = {}
    for g in cls_gts:
        gt_by_img.setdefault(g['img_id'], []).append(g)

    cls_dets.sort(key=lambda x: x['confidence'], reverse=True)

    matched = {}
    tp_arr = np.zeros(len(cls_dets))
    fp_arr = np.zeros(len(cls_dets))

    for i, det in enumerate(cls_dets):
        img_id = det['img_id']
        gts_img = gt_by_img.get(img_id, [])
        matched_set = matched.setdefault(img_id, set())

        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts_img):
            if j in matched_set:
                continue
            iou = compute_iou(det['bbox'], g['bbox'])
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thr and best_j >= 0:
            matched_set.add(best_j)
            tp_arr[i] = 1
        else:
            fp_arr[i] = 1

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    rec  = tp_cum / (len(cls_gts) + 1e-10)
    prec = tp_cum / (tp_cum + fp_cum + 1e-10)

    return voc_ap(rec, prec)


# ──────────────────────────────────────────────────────────────
# 加载 TTA JSON 结果
# ──────────────────────────────────────────────────────────────
def load_tta_results(json_path, image_dir):
    """
    从 inference_tta.py 生成的 JSON 加载预测结果
    
    JSON 格式：
    [
      {
        "image": "xxx.jpg",
        "boxes": [[x1,y1,x2,y2], ...],
        "scores": [0.9, ...],
        "labels": [0, ...],
        "num_detections": N
      },
      ...
    ]
    
    Returns:
        all_dets: list of {'img_id', 'bbox', 'confidence', 'class_id'}
        img_sizes: dict {img_id: (w, h)}  用于加载 GT 时坐标转换
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)

    all_dets = []
    img_sizes = {}

    # 先获取所有图像尺寸（用于 GT 坐标转换）
    image_dir = Path(image_dir)
    print(f"  加载图像尺寸（用于 GT 坐标转换）...")
    for item in tqdm(raw, desc="  读取尺寸", ncols=70):
        img_id = Path(item['image']).stem
        img_path = image_dir / item['image']
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                img_sizes[img_id] = (w, h)
        
        # 加载检测框
        boxes  = item.get('boxes', [])
        scores = item.get('scores', [])
        labels = item.get('labels', [])
        
        for box, score, label in zip(boxes, scores, labels):
            all_dets.append({
                'img_id':     img_id,
                'bbox':       box,           # 已经是 xyxy 绝对坐标
                'confidence': float(score),
                'class_id':   int(label)
            })

    print(f"  共加载 {len(raw)} 张图的预测，{len(all_dets)} 个检测框")
    return all_dets, img_sizes


# ──────────────────────────────────────────────────────────────
# 加载 GT 标注
# ──────────────────────────────────────────────────────────────
def load_gt(label_dir, img_sizes):
    """
    从 YOLO 格式的 label 文件加载 GT
    
    Returns:
        all_gts: list of {'img_id', 'bbox', 'class_id'}
    """
    label_dir = Path(label_dir)
    all_gts = []
    
    for img_id, (w, h) in img_sizes.items():
        label_path = label_dir / f"{img_id}.txt"
        if not label_path.exists():
            continue
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc = float(parts[1]) * w
                    yc = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h
                    all_gts.append({
                        'img_id':   img_id,
                        'class_id': cls_id,
                        'bbox':     [xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2]
                    })
    
    print(f"  共加载 {len(all_gts)} 个 GT 标注框")
    return all_gts


# ──────────────────────────────────────────────────────────────
# 计算完整 mAP
# ──────────────────────────────────────────────────────────────
def compute_map(all_dets, all_gts):
    """计算所有类别的 mAP@0.5 和 mAP@0.5:0.95"""
    per_class_ap50 = {}
    per_class_ap50_95 = {}
    
    print("  计算 AP...")
    for i, name in enumerate(CLASS_NAMES):
        ap50 = compute_ap_at_iou(all_dets, all_gts, i, 0.5)
        per_class_ap50[name] = ap50
        
        ap_list = [compute_ap_at_iou(all_dets, all_gts, i, t) for t in IOU_THRESHOLDS]
        per_class_ap50_95[name] = float(np.mean(ap_list))
        
        print(f"    {name:<12} AP@0.5={ap50*100:.2f}%  AP@0.5:0.95={per_class_ap50_95[name]*100:.2f}%")
    
    map50    = float(np.mean(list(per_class_ap50.values())))
    map50_95 = float(np.mean(list(per_class_ap50_95.values())))
    
    return map50, map50_95, per_class_ap50, per_class_ap50_95


# ──────────────────────────────────────────────────────────────
# 普通推理评估（可选对比）
# ──────────────────────────────────────────────────────────────
def evaluate_model_normal(weights, image_dir, label_dir, conf=0.25, iou=0.45, imgsz=1280):
    """用普通推理（无TTA）评估一个模型"""
    try:
        from ultralytics import YOLO
        import torch
        import time
    except ImportError:
        print("ultralytics 未安装")
        return None
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = YOLO(weights)
    model.to(device)
    
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    all_dets = []
    all_gts  = []
    times    = []
    
    # 预热
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model(dummy, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    
    for img_path in tqdm(image_files, desc="  普通推理", ncols=70):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_id = img_path.stem
        
        t0 = time.perf_counter()
        res = model(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        
        for r in res:
            if r.boxes is None:
                continue
            for box in r.boxes:
                all_dets.append({
                    'img_id':     img_id,
                    'bbox':       box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf),
                    'class_id':   int(box.cls)
                })
        
        label_path = label_dir / f"{img_id}.txt"
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc = float(parts[1]) * w
                        yc = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h
                        all_gts.append({
                            'img_id':   img_id,
                            'class_id': cls_id,
                            'bbox':     [xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2]
                        })
    
    map50, map50_95, per_class_ap50, per_class_ap50_95 = compute_map(all_dets, all_gts)
    avg_time = float(np.mean(times))
    
    return {
        'map50':             map50,
        'map50_95':          map50_95,
        'fps':               1.0 / avg_time,
        'avg_latency_ms':    avg_time * 1000,
        'per_class_ap50':    per_class_ap50,
        'per_class_ap50_95': per_class_ap50_95,
        'num_images':        len(times),
    }


# ──────────────────────────────────────────────────────────────
# 打印对比表
# ──────────────────────────────────────────────────────────────
def print_report(tta_map50, tta_map50_95, tta_per50, tta_per50_95,
                 normal_metrics=None,
                 baseline=BASELINE,
                 num_images=500):
    
    sep = "=" * 80
    
    print(f"\n{sep}")
    print("  TTA 评估报告 vs 基线对比")
    print(sep)
    
    has_normal = normal_metrics is not None
    
    # 表头
    if has_normal:
        print(f"\n{'指标':<20} {'基线 YOLOv8m':<20} {'v2_ultra (无TTA)':<22} {'v2_ultra + TTA':<20} {'TTA vs 基线'}")
        print("-" * 100)
    else:
        print(f"\n{'指标':<20} {'基线 YOLOv8m':<20} {'v2_ultra + TTA':<20} {'TTA vs 基线'}")
        print("-" * 70)
    
    def pp(v):
        return f"{v*100:.2f}%"
    
    def delta_str(a, b):
        d = (b - a) * 100
        return f"{'+'if d>=0 else ''}{d:.2f}pp"
    
    rows = [
        ("mAP@0.5",      baseline['map50'],    tta_map50),
        ("mAP@0.5:0.95", baseline['map50_95'], tta_map50_95),
    ]
    
    for label, base_val, tta_val in rows:
        d = delta_str(base_val, tta_val)
        flag = " ✓" if (tta_val - base_val) * 100 >= 0.5 else " ~"
        if has_normal:
            key_map = {"mAP@0.5": "map50", "mAP@0.5:0.95": "map50_95"}
            n_val = normal_metrics[key_map[label]]
            print(f"  {label:<18} {pp(base_val):<20} {pp(n_val):<22} {pp(tta_val):<20} {d}{flag}")
        else:
            print(f"  {label:<18} {pp(base_val):<20} {pp(tta_val):<20} {d}{flag}")
    
    # 每类对比
    print(f"\n  Per-Class AP@0.5:")
    if has_normal:
        print(f"  {'类别':<14} {'基线':>10}  {'无TTA':>10}  {'TTA':>10}  {'ΔvsTTA-基线':>12}")
        print("  " + "-" * 58)
    else:
        print(f"  {'类别':<14} {'基线':>10}  {'TTA':>10}  {'Δ':>10}")
        print("  " + "-" * 46)
    
    for name in CLASS_NAMES:
        b = baseline['per_class_ap50'].get(name, 0)
        t = tta_per50.get(name, 0)
        d = delta_str(b, t)
        if has_normal:
            n = normal_metrics['per_class_ap50'].get(name, 0)
            print(f"  {name:<14} {b*100:>9.2f}%  {n*100:>9.2f}%  {t*100:>9.2f}%  {d:>12}")
        else:
            print(f"  {name:<14} {b*100:>9.2f}%  {t*100:>9.2f}%  {d:>10}")
    
    print(f"\n  Per-Class AP@0.5:0.95:")
    if has_normal:
        print(f"  {'类别':<14} {'基线':>10}  {'无TTA':>10}  {'TTA':>10}  {'Δ':>10}")
        print("  " + "-" * 58)
    else:
        print(f"  {'类别':<14} {'基线':>10}  {'TTA':>10}  {'Δ':>10}")
        print("  " + "-" * 46)
    
    for name in CLASS_NAMES:
        b = baseline['per_class_ap50_95'].get(name, 0)
        t = tta_per50_95.get(name, 0)
        d = delta_str(b, t)
        if has_normal:
            n = normal_metrics['per_class_ap50_95'].get(name, 0)
            print(f"  {name:<14} {b*100:>9.2f}%  {n*100:>9.2f}%  {t*100:>9.2f}%  {d:>10}")
        else:
            print(f"  {name:<14} {b*100:>9.2f}%  {t*100:>9.2f}%  {d:>10}")
    
    print(f"\n{sep}")
    
    # 总结
    d50    = (tta_map50    - baseline['map50'])    * 100
    d50_95 = (tta_map50_95 - baseline['map50_95']) * 100
    
    print(f"\n  📊 TTA 效果总结（vs 基线）：")
    print(f"     mAP@0.5     : {baseline['map50']*100:.2f}% → {tta_map50*100:.2f}%   ({'+' if d50>=0 else ''}{d50:.2f}pp)")
    print(f"     mAP@0.5:0.95: {baseline['map50_95']*100:.2f}% → {tta_map50_95*100:.2f}%   ({'+' if d50_95>=0 else ''}{d50_95:.2f}pp)")
    
    goal_50    = d50    >= 5.0
    goal_50_95 = d50_95 >= 7.0
    
    print(f"\n  🎯 目标达成情况：")
    print(f"     mAP@0.5    +5pp目标：{'✅ 达成' if goal_50 else f'❌ 未达成（差 {5.0-d50:.2f}pp）'}")
    print(f"     mAP@0.5:0.95 +7pp目标：{'✅ 达成' if goal_50_95 else f'❌ 未达成（差 {7.0-d50_95:.2f}pp）'}")
    
    if not goal_50 or not goal_50_95:
        print(f"\n  💡 建议：")
        if not goal_50:
            print(f"     - mAP@0.5 还需提升 {5.0-d50:.1f}pp → 建议使用模型集成（ensemble）或更大模型")
        if not goal_50_95:
            print(f"     - mAP@0.5:0.95 还需提升 {7.0-d50_95:.1f}pp → 建议增大 imgsz 到 1600 或加入 SAM 精修")
    
    print(f"\n{sep}")


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='评估 TTA 推理结果，计算 mAP 并与基线对比')
    p.add_argument('--pred',       type=str, required=True,
                   help='inference_tta.py 输出的 JSON 文件路径')
    p.add_argument('--label-dir',  type=str, required=True,
                   help='GT 标注目录（YOLO 格式 .txt 文件）')
    p.add_argument('--image-dir',  type=str, required=True,
                   help='测试图像目录（用于获取图像尺寸）')
    p.add_argument('--compare-weights', type=str, default=None,
                   help='（可选）同时跑普通推理对比，填入权重路径')
    p.add_argument('--compare-imgsz', type=int, default=1280,
                   help='普通推理的图像尺寸，默认 1280')
    p.add_argument('--baseline-map50',    type=float, default=None,
                   help='覆盖默认基线 mAP@0.5（小数形式，如 0.9587）')
    p.add_argument('--baseline-map50-95', type=float, default=None,
                   help='覆盖默认基线 mAP@0.5:0.95')
    p.add_argument('--output', type=str, default='outputs/tta_eval',
                   help='结果输出目录')
    return p.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("  TTA 结果评估器")
    print("=" * 70)
    
    # 覆盖基线数字（如果提供）
    baseline = dict(BASELINE)
    if args.baseline_map50 is not None:
        baseline['map50'] = args.baseline_map50
    if args.baseline_map50_95 is not None:
        baseline['map50_95'] = args.baseline_map50_95
    
    print(f"\n[1/2] 加载 TTA 预测结果: {args.pred}")
    all_dets, img_sizes = load_tta_results(args.pred, args.image_dir)
    
    print(f"\n[2/2] 加载 GT 标注: {args.label_dir}")
    all_gts = load_gt(args.label_dir, img_sizes)
    
    print(f"\n计算 TTA mAP...")
    tta_map50, tta_map50_95, tta_per50, tta_per50_95 = compute_map(all_dets, all_gts)
    
    # 可选：普通推理对比
    normal_metrics = None
    if args.compare_weights:
        print(f"\n可选：普通推理评估 {args.compare_weights}")
        normal_metrics = evaluate_model_normal(
            args.compare_weights,
            args.image_dir,
            args.label_dir,
            imgsz=args.compare_imgsz
        )
    
    # 打印报告
    print_report(
        tta_map50, tta_map50_95, tta_per50, tta_per50_95,
        normal_metrics=normal_metrics,
        baseline=baseline,
        num_images=len(img_sizes)
    )
    
    # 保存结果
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{ts}.json"
    
    result = {
        'timestamp': ts,
        'pred_file': args.pred,
        'num_images': len(img_sizes),
        'tta': {
            'map50': tta_map50,
            'map50_95': tta_map50_95,
            'per_class_ap50': tta_per50,
            'per_class_ap50_95': tta_per50_95,
        },
        'baseline': baseline,
        'delta_map50':    tta_map50    - baseline['map50'],
        'delta_map50_95': tta_map50_95 - baseline['map50_95'],
    }
    if normal_metrics:
        result['normal'] = normal_metrics
    
    import json
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  结果已保存 → {out_file}")


if __name__ == '__main__':
    main()
