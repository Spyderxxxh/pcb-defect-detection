"""
导出后精度验证脚本
在 test set 上对比 PyTorch / ONNX / TRT-FP16 / TRT-INT8 的 mAP，
确认导出后精度损失在可接受范围内（通常 mAP 损失 < 0.3pp 是合格线）。

用法：
  python eval_exported.py \
      --pt-weights runs/detect/runs/detect/defectguard_v2_ultra/weights/best.pt \
      --export-dir outputs/exported_models \
      --image-dir  data/deeppcb/yolo_format/images/test \
      --label-dir  data/deeppcb/yolo_format/labels/test \
      --formats pt onnx trt-fp16 trt-int8
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────
# CUDA 环境修复（必须在 import torch 之前）
# ─────────────────────────────────────────────────────────────
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if _cvd == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("  ⚠️  修复 CUDA_VISIBLE_DEVICES='' → '0'")

_lib_path = "/usr/lib/x86_64-linux-gnu"
if _lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"{_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print(f"  ⚠️  添加 LD_LIBRARY_PATH={_lib_path}")


# ─────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="验证各导出格式精度（mAP）")
    parser.add_argument("--pt-weights", type=str,
        default="runs/detect/runs/detect/defectguard_v2_ultra/weights/best.pt")
    parser.add_argument("--export-dir", type=str, default="outputs/exported_models")
    parser.add_argument("--image-dir",  type=str,
        default="data/deeppcb/yolo_format/images/test")
    parser.add_argument("--label-dir",  type=str,
        default="data/deeppcb/yolo_format/labels/test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf",  type=float, default=0.001,
        help="低阈值保留更多候选框，mAP 评估更准确")
    parser.add_argument("--iou-thr", type=float, default=0.5,
        help="AP@0.5 的 IoU 阈值")
    parser.add_argument("--formats", nargs="+",
        default=["pt", "onnx", "trt-fp16", "trt-int8"])
    parser.add_argument("--output-dir", type=str, default="outputs/eval_exported")
    parser.add_argument("--max-images", type=int, default=500,
        help="最多评估多少张（0=全部）")
    return parser.parse_args()


CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]


# ─────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────
def load_labels(label_dir: str) -> Dict[str, List]:
    """
    读取 YOLO 格式 label 文件
    返回 {stem: [[cls, cx, cy, w, h], ...]}
    """
    labels = {}
    for txt in Path(label_dir).glob("*.txt"):
        boxes = []
        with open(txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
        labels[txt.stem] = boxes
    return labels


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


# ─────────────────────────────────────────────────────────────
# AP 计算
# ─────────────────────────────────────────────────────────────
def compute_iou(box_a, box_b):
    """box: [x1,y1,x2,y2]"""
    xa = max(box_a[0], box_b[0]); ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2]); yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls, precisions):
    """VOC 11-point 插值 AP"""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p = max([p for r, p in zip(recalls, precisions) if r >= thr], default=0.0)
        ap += p / 11
    return ap


def evaluate_predictions(
    all_preds: Dict[str, List],   # {stem: [[cls, conf, x1, y1, x2, y2], ...]}
    all_labels: Dict[str, List],  # {stem: [[cls, cx, cy, w, h], ...]}
    img_sizes: Dict[str, tuple],  # {stem: (w, h)}
    iou_thr: float = 0.5,
) -> Dict:
    """计算每类 AP 和整体 mAP"""
    # 按类别收集 (tp/fp, conf) 对
    class_data = defaultdict(lambda: {"tp_fp": [], "n_gt": 0})

    for stem, gts in all_labels.items():
        preds = all_preds.get(stem, [])
        img_w, img_h = img_sizes.get(stem, (1280, 1280))

        # GT → xyxy
        gt_by_cls = defaultdict(list)
        for g in gts:
            cls = int(g[0])
            x1, y1, x2, y2 = yolo_to_xyxy(g[1], g[2], g[3], g[4], img_w, img_h)
            gt_by_cls[cls].append([x1, y1, x2, y2])
            class_data[cls]["n_gt"] += 1

        # 按类别匹配
        for cls, gt_boxes in gt_by_cls.items():
            matched = [False] * len(gt_boxes)
            cls_preds = sorted(
                [p for p in preds if int(p[0]) == cls],
                key=lambda x: -x[1]
            )
            for p in cls_preds:
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(gt_boxes):
                    iou = compute_iou(p[2:6], gt)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thr and not matched[best_j]:
                    class_data[cls]["tp_fp"].append((1, p[1]))
                    matched[best_j] = True
                else:
                    class_data[cls]["tp_fp"].append((0, p[1]))

        # 无 GT 的类别，预测框全算 FP
        for p in preds:
            cls = int(p[0])
            if cls not in gt_by_cls:
                class_data[cls]["tp_fp"].append((0, p[1]))

    # 计算每类 AP
    ap_per_class = {}
    for cls in range(len(CLASS_NAMES)):
        data = class_data[cls]
        n_gt = data["n_gt"]
        if n_gt == 0:
            ap_per_class[CLASS_NAMES[cls]] = 0.0
            continue
        tp_fp = sorted(data["tp_fp"], key=lambda x: -x[1])
        tp_cum = np.cumsum([x[0] for x in tp_fp])
        fp_cum = np.cumsum([1 - x[0] for x in tp_fp])
        recalls    = (tp_cum / n_gt).tolist()
        precisions = (tp_cum / (tp_cum + fp_cum + 1e-9)).tolist()
        ap_per_class[CLASS_NAMES[cls]] = compute_ap(recalls, precisions)

    map50 = float(np.mean(list(ap_per_class.values())))
    return {"map50": map50, "ap_per_class": ap_per_class}


# ─────────────────────────────────────────────────────────────
# 单格式评估
# ─────────────────────────────────────────────────────────────
def eval_single(
    fmt: str, model_path: str,
    image_files: List[Path],
    labels: Dict, imgsz: int, conf: float, iou_thr: float,
) -> dict:
    from ultralytics import YOLO

    print(f"\n  [{fmt.upper()}] 评估中... ({len(image_files)} 张)")
    model = YOLO(model_path)

    all_preds = {}
    img_sizes = {}
    t0 = time.time()

    for fp in image_files:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_sizes[fp.stem] = (w, h)

        results = model.predict(img, verbose=False, imgsz=imgsz, conf=conf)
        preds = []
        if results[0].boxes:
            for box in results[0].boxes:
                cls  = int(box.cls[0])
                c    = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                preds.append([cls, c] + xyxy)
        all_preds[fp.stem] = preds

    elapsed = time.time() - t0
    avg_latency = elapsed / len(image_files) * 1000

    metrics = evaluate_predictions(all_preds, labels, img_sizes, iou_thr)

    result = {
        "format":          fmt,
        "map50":           round(metrics["map50"] * 100, 2),
        "ap_per_class":    {k: round(v * 100, 2) for k, v in metrics["ap_per_class"].items()},
        "avg_latency_ms":  round(avg_latency, 2),
        "fps":             round(1000 / avg_latency, 1),
        "n_images":        len(image_files),
    }
    print(f"     mAP@0.5={result['map50']:.2f}%  FPS={result['fps']}")
    return result


# ─────────────────────────────────────────────────────────────
# 打印精度对比表
# ─────────────────────────────────────────────────────────────
def print_accuracy_table(all_results: List[dict]):
    pt_map = next((x["map50"] for x in all_results if x["format"] == "pt"), None)

    print(f"\n{'='*80}")
    print("  精度对比（mAP@0.5，DeepPCB test set）")
    print(f"{'='*80}")
    header = f"  {'格式':<14} {'mAP@0.5':<10} {'Δ vs PT':<10} " + \
             "  ".join(f"{c[:5]:<8}" for c in CLASS_NAMES)
    print(header)
    print(f"  {'-'*76}")

    for r in all_results:
        delta = f"{r['map50'] - pt_map:+.2f}" if pt_map else "—"
        cls_str = "  ".join(f"{r['ap_per_class'].get(c, 0):<8.2f}" for c in CLASS_NAMES)
        flag = ""
        if pt_map and abs(r["map50"] - pt_map) < 0.3 and r["format"] != "pt":
            flag = " ✅"
        elif pt_map and abs(r["map50"] - pt_map) >= 0.3 and r["format"] != "pt":
            flag = " ⚠️"
        print(f"  {r['format']:<14} {r['map50']:<10.2f} {delta:<10} {cls_str}{flag}")

    print(f"{'='*80}")
    print("  ✅ = 精度损失 < 0.3pp（合格）  ⚠️ = 精度损失 ≥ 0.3pp（需关注）")

    # 结论
    fp16 = next((x for x in all_results if x["format"] == "trt-fp16"), None)
    int8 = next((x for x in all_results if x["format"] == "trt-int8"), None)
    if fp16 and pt_map:
        drop = pt_map - fp16["map50"]
        print(f"\n  TRT-FP16 精度损失：{drop:.2f}pp  {'合格 ✅' if drop < 0.3 else '偏高 ⚠️'}")
    if int8 and pt_map:
        drop = pt_map - int8["map50"]
        print(f"  TRT-INT8 精度损失：{drop:.2f}pp  {'合格 ✅' if drop < 0.5 else '偏高 ⚠️'}")


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────
def find_model_path(fmt: str, pt_weights: str, export_dir: str) -> Optional[str]:
    fmt_file_map = {
        "pt":       pt_weights,
        "onnx":     str(Path(export_dir) / "model_fp32.onnx"),
        "trt-fp32": str(Path(export_dir) / "model_fp32.engine"),
        "trt-fp16": str(Path(export_dir) / "model_fp16.engine"),
        "trt-int8": str(Path(export_dir) / "model_int8.engine"),
    }
    p = fmt_file_map.get(fmt)
    return p if p and Path(p).exists() else None


def main():
    args = parse_args()

    # 加载 labels
    print(f"\n  加载标注... ({args.label_dir})")
    labels = load_labels(args.label_dir)
    print(f"  找到 {len(labels)} 个标注文件")

    # 收集图像
    img_dir = Path(args.image_dir)
    img_files = sorted(
        [f for f in img_dir.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if args.max_images > 0:
        img_files = img_files[:args.max_images]
    print(f"  评估图像：{len(img_files)} 张")

    # 收集要测试的格式
    to_test = []
    for fmt in args.formats:
        path = find_model_path(fmt, args.pt_weights, args.export_dir)
        if path:
            to_test.append((fmt, path))
        else:
            print(f"  ⚠️  [{fmt}] 模型文件不存在，跳过")

    # 逐格式评估
    all_results = []
    for fmt, path in to_test:
        try:
            r = eval_single(fmt, path, img_files, labels, args.imgsz, args.conf, args.iou_thr)
            all_results.append(r)
        except Exception as e:
            print(f"  ❌ [{fmt}] 评估失败：{e}")

    if not all_results:
        print("  ❌ 所有格式评估均失败")
        sys.exit(1)

    print_accuracy_table(all_results)

    # 保存报告
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "results": all_results,
    }
    report_path = out_dir / f"eval_exported_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  📄 报告已保存 → {report_path}")
    print(f"  ✅ 精度验证完成\n")


if __name__ == "__main__":
    main()
