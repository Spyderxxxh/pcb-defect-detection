"""
基线对比评估脚本
对比 YOLOv8m 纯基线 vs DefectGuard（YOLOv8 + NonLocal）系统
自动输出格式化对比表格，用于面试数据支撑

Usage:
    # 先训练基线（如果还没有权重）
    python train_yolo.py --model yolov8m --name baseline --epochs 100

    # 评估对比（需要提供两个权重文件）
    python baseline_compare.py \
        --baseline runs/detect/baseline/weights/best.pt \
        --enhanced runs/detect/defectguard/weights/best.pt \
        --data data/deeppcb/dataset.yaml

    # 只评估单个模型（用于快速验证）
    python baseline_compare.py \
        --baseline runs/detect/baseline/weights/best.pt \
        --data data/deeppcb/dataset.yaml
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent))


# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)   # 0.50, 0.55, ..., 0.95（共10个）
IOU_THRESHOLD_50 = 0.5


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Baseline vs DefectGuard Comparison')
    p.add_argument('--baseline', type=str, required=True,
                   help='基线模型权重路径（纯 YOLOv8m）')
    p.add_argument('--enhanced', type=str, default=None,
                   help='增强模型权重路径（YOLOv8 + NonLocal），不填则只评估基线')
    p.add_argument('--data', type=str, default='data/deeppcb/dataset.yaml',
                   help='数据集 yaml 配置路径')
    p.add_argument('--split', type=str, default='test',
                   choices=['val', 'test'],
                   help='评估分割：val 或 test')
    p.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    p.add_argument('--iou',  type=float, default=0.45, help='NMS IoU 阈值')
    p.add_argument('--imgsz', type=int, default=640,   help='推理图像尺寸')
    p.add_argument('--device', type=str, default='0',  help='GPU 设备 ID')
    return p.parse_args()


def compute_iou(b1, b2):
    """计算两个 xyxy 框的 IoU"""
    x1 = max(b1[0], b2[0]);  y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]);  y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def voc_ap(rec, prec):
    """VOC 2010+ AP（面积法）"""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


def compute_ap_at_iou(all_dets, all_gts, cls_id, iou_thr):
    """
    计算单个类别在给定 IoU 阈值下的 AP

    all_dets: list of {'bbox', 'confidence', 'class_id', 'img_id'}
    all_gts:  list of {'bbox', 'class_id', 'img_id'}
    """
    # 筛选当前类别
    cls_dets = [d for d in all_dets if d['class_id'] == cls_id]
    cls_gts  = [g for g in all_gts  if g['class_id'] == cls_id]

    if len(cls_gts) == 0:
        return 0.0

    # 按 img_id 分组 GT
    gt_by_img = {}
    for g in cls_gts:
        gt_by_img.setdefault(g['img_id'], []).append(g)

    # 按置信度降序
    cls_dets.sort(key=lambda x: x['confidence'], reverse=True)

    matched = {}   # img_id -> set of matched gt indices
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
# 核心评估函数
# ──────────────────────────────────────────────────────────────
def evaluate_model(model_path, img_dir, label_dir, conf, iou_thr, imgsz, device_str):
    """
    评估一个模型，返回完整指标字典
    """
    from ultralytics import YOLO

    device = f'cuda:{device_str}' if torch.cuda.is_available() else 'cpu'
    model  = YOLO(model_path)
    model.to(device)

    image_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {img_dir}")

    print(f"  Found {len(image_files)} images, device={device}")

    # 预热
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(3):
        model(dummy, conf=conf, iou=iou_thr, imgsz=imgsz, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    all_dets = []
    all_gts  = []
    inference_times = []

    for img_path in tqdm(image_files, desc=f"  Evaluating {Path(model_path).stem}", ncols=80):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_id = img_path.stem

        # 推理（3次取平均，稳定计时）
        t_list = []
        last_results = None
        for _ in range(3):
            t0 = time.perf_counter()
            res = model(img, conf=conf, iou=iou_thr, imgsz=imgsz, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_list.append(time.perf_counter() - t0)
            last_results = res
        inference_times.append(np.mean(t_list))

        # 解析检测框
        for r in last_results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                all_dets.append({
                    'img_id':     img_id,
                    'bbox':       box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf),
                    'class_id':   int(box.cls)
                })

        # 加载 GT
        label_path = label_dir / f"{img_id}.txt"
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id   = int(parts[0])
                        xc, yc   = float(parts[1])*w, float(parts[2])*h
                        bw, bh   = float(parts[3])*w, float(parts[4])*h
                        all_gts.append({
                            'img_id':   img_id,
                            'class_id': cls_id,
                            'bbox':     [xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2]
                        })

    # ── 计算 mAP@0.5 和 mAP@0.5:0.95 ──
    per_class_ap50     = {}
    per_class_ap50_95  = {}

    for i, name in enumerate(CLASS_NAMES):
        ap50 = compute_ap_at_iou(all_dets, all_gts, i, 0.5)
        per_class_ap50[name] = ap50

        ap_list = [compute_ap_at_iou(all_dets, all_gts, i, t) for t in IOU_THRESHOLDS]
        per_class_ap50_95[name] = float(np.mean(ap_list))

    map50    = float(np.mean(list(per_class_ap50.values())))
    map50_95 = float(np.mean(list(per_class_ap50_95.values())))

    avg_time = float(np.mean(inference_times))
    fps      = 1.0 / avg_time

    return {
        'map50':            map50,
        'map50_95':         map50_95,
        'fps':              fps,
        'avg_latency_ms':   avg_time * 1000,
        'per_class_ap50':   per_class_ap50,
        'per_class_ap50_95': per_class_ap50_95,
        'num_images':       len(inference_times),
    }


# ──────────────────────────────────────────────────────────────
# 打印对比表
# ──────────────────────────────────────────────────────────────
def print_comparison(baseline_metrics, enhanced_metrics=None):
    sep = "=" * 72

    def fmt(v, is_pct=True):
        return f"{v*100:.2f}%" if is_pct else f"{v:.1f}"

    def delta(a, b, is_pct=True):
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"({sign}{d*100:.2f}pp)" if is_pct else f"({sign}{d:.1f})"

    print(f"\n{sep}")
    print("  DefectGuard vs Baseline — Evaluation Report")
    print(sep)

    # 总体指标
    print(f"\n{'Metric':<20} {'Baseline (YOLOv8m)':<24}", end="")
    if enhanced_metrics:
        print(f"{'DefectGuard':<24} {'Delta':<16}", end="")
    print()
    print("-" * (20 + 24 + (40 if enhanced_metrics else 0)))

    metrics_cfg = [
        ("mAP@0.5",        "map50",          True),
        ("mAP@0.5:0.95",   "map50_95",       True),
        ("FPS",            "fps",            False),
        ("Latency (ms)",   "avg_latency_ms", False),
    ]
    for label, key, is_pct in metrics_cfg:
        base_val = baseline_metrics[key]
        row = f"  {label:<18} {fmt(base_val, is_pct):<24}"
        if enhanced_metrics:
            enh_val = enhanced_metrics[key]
            row += f"{fmt(enh_val, is_pct):<24} {delta(base_val, enh_val, is_pct):<16}"
        print(row)

    # 每类 AP@0.5
    print(f"\n  Per-Class AP@0.5:")
    print(f"  {'Class':<14} {'Baseline':>10}", end="")
    if enhanced_metrics:
        print(f"  {'DefectGuard':>12}  {'Delta':>10}", end="")
    print()
    print("  " + "-" * (14 + 12 + (28 if enhanced_metrics else 0)))

    for name in CLASS_NAMES:
        b = baseline_metrics['per_class_ap50'].get(name, 0.0)
        row = f"  {name:<14} {b*100:>9.2f}%"
        if enhanced_metrics:
            e = enhanced_metrics['per_class_ap50'].get(name, 0.0)
            d = e - b
            sign = "+" if d >= 0 else ""
            row += f"  {e*100:>11.2f}%  {sign}{d*100:>8.2f}pp"
        print(row)

    # 每类 AP@0.5:0.95
    print(f"\n  Per-Class AP@0.5:0.95:")
    print(f"  {'Class':<14} {'Baseline':>10}", end="")
    if enhanced_metrics:
        print(f"  {'DefectGuard':>12}  {'Delta':>10}", end="")
    print()
    print("  " + "-" * (14 + 12 + (28 if enhanced_metrics else 0)))

    for name in CLASS_NAMES:
        b = baseline_metrics['per_class_ap50_95'].get(name, 0.0)
        row = f"  {name:<14} {b*100:>9.2f}%"
        if enhanced_metrics:
            e = enhanced_metrics['per_class_ap50_95'].get(name, 0.0)
            d = e - b
            sign = "+" if d >= 0 else ""
            row += f"  {e*100:>11.2f}%  {sign}{d*100:>8.2f}pp"
        print(row)

    print(f"\n{sep}")

    # 面试用总结句
    if enhanced_metrics:
        delta_map50    = (enhanced_metrics['map50']    - baseline_metrics['map50'])    * 100
        delta_map50_95 = (enhanced_metrics['map50_95'] - baseline_metrics['map50_95']) * 100
        print("\n  [面试话术参考]")
        print(f"  在 DeepPCB 数据集（{baseline_metrics['num_images']} 张验证图）上：")
        print(f"  - mAP@0.5  : 基线 {baseline_metrics['map50']*100:.1f}%"
              f" → DefectGuard {enhanced_metrics['map50']*100:.1f}%"
              f"  （提升 {delta_map50:+.1f}pp）")
        print(f"  - mAP@0.5:0.95: 基线 {baseline_metrics['map50_95']*100:.1f}%"
              f" → DefectGuard {enhanced_metrics['map50_95']*100:.1f}%"
              f"  （提升 {delta_map50_95:+.1f}pp）")
        print(f"  - 推理速度: 基线 {baseline_metrics['fps']:.0f} FPS"
              f" / DefectGuard {enhanced_metrics['fps']:.0f} FPS")
    print(sep)


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 解析数据路径
    import yaml
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[Error] Dataset yaml not found: {data_yaml}")
        sys.exit(1)

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    # 兼容绝对/相对路径
    root = Path(data_cfg.get('path', data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root

    split = args.split
    img_dir   = root / 'images' / split
    label_dir = root / 'labels' / split

    if not img_dir.exists():
        print(f"[Error] Image dir not found: {img_dir}")
        sys.exit(1)

    print("=" * 72)
    print("  DefectGuard Baseline Comparison")
    print("=" * 72)
    print(f"  Data split : {split}")
    print(f"  Image dir  : {img_dir}")
    print(f"  Label dir  : {label_dir}")
    print(f"  conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")

    # 评估基线
    print(f"\n[1/{'2' if args.enhanced else '1'}] Evaluating Baseline: {args.baseline}")
    baseline_metrics = evaluate_model(
        args.baseline, img_dir, label_dir,
        args.conf, args.iou, args.imgsz, args.device
    )

    # 评估增强模型（可选）
    enhanced_metrics = None
    if args.enhanced:
        print(f"\n[2/2] Evaluating DefectGuard: {args.enhanced}")
        enhanced_metrics = evaluate_model(
            args.enhanced, img_dir, label_dir,
            args.conf, args.iou, args.imgsz, args.device
        )

    # 打印对比表
    print_comparison(baseline_metrics, enhanced_metrics)

    # 保存结果
    import json, datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"outputs/comparison_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'baseline':  baseline_metrics,
            'enhanced':  enhanced_metrics,
            'args': vars(args)
        }, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
