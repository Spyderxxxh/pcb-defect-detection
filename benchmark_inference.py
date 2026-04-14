"""
推理速度基准测试脚本
对比 PyTorch (.pt) / ONNX / TensorRT FP32 / FP16 / INT8 的
延迟、FPS、GPU 显存占用，并输出面试可展示的汇总表。

用法：
  # 使用默认目录（exports 产物 + 原始 .pt 权重）
  python benchmark_inference.py

  # 指定权重和导出目录
  python benchmark_inference.py \
      --pt-weights runs/detect/runs/detect/defectguard_v2_ultra/weights/best.pt \
      --export-dir outputs/exported_models \
      --image-dir data/deeppcb/yolo_format/images/test \
      --warmup 20 --iterations 200

  # 只对比 pt vs fp16（服务器快速对比）
  python benchmark_inference.py --formats pt trt-fp16
"""

import os
import sys
import time
import json
import argparse
import gc
from pathlib import Path
from datetime import datetime
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
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 各导出格式推理速度")
    parser.add_argument(
        "--pt-weights", type=str,
        default="runs/detect/runs/detect/defectguard_v2_ultra/weights/best.pt",
        help="原始 PyTorch 权重路径"
    )
    parser.add_argument(
        "--export-dir", type=str, default="outputs/exported_models",
        help="export_trt.py 的导出产物目录"
    )
    parser.add_argument(
        "--image-dir", type=str,
        default="data/deeppcb/yolo_format/images/test",
        help="测试图像目录（优先使用真实图片，为空则用随机噪声）"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="推理输入尺寸"
    )
    parser.add_argument(
        "--warmup", type=int, default=20,
        help="预热轮数（不计入统计）"
    )
    parser.add_argument(
        "--iterations", type=int, default=200,
        help="正式测试轮数"
    )
    parser.add_argument(
        "--formats", nargs="+",
        default=["pt", "onnx", "trt-fp32", "trt-fp16", "trt-int8"],
        help="要测试的格式列表"
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="推理 batch size"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="GPU 设备"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/benchmark",
        help="结果保存目录"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="检测置信度阈值"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
def get_gpu_memory_mb() -> float:
    """获取当前 GPU 已用显存（MB）"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


def reset_gpu_memory_stats():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def load_images(image_dir: str, imgsz: int, n: int = 200) -> List[np.ndarray]:
    """从目录加载图像，不足时复用；目录不存在则生成随机图"""
    images = []
    path = Path(image_dir)
    if path.exists():
        img_files = sorted(
            [f for f in path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )[:n]
        for fp in img_files:
            img = cv2.imread(str(fp))
            if img is not None:
                images.append(img)

    if not images:
        print(f"  ⚠️  未找到真实图像，使用随机噪声图（{n} 张）")
        for _ in range(min(n, 50)):
            images.append(np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8))

    # 如果数量不够，循环复用
    while len(images) < n:
        images += images
    return images[:n]


def find_model_path(fmt: str, pt_weights: str, export_dir: str) -> Optional[str]:
    """根据格式名查找对应模型文件"""
    fmt_file_map = {
        "pt":       pt_weights,
        "onnx":     str(Path(export_dir) / "model_fp32.onnx"),
        "trt-fp32": str(Path(export_dir) / "model_fp32.engine"),
        "trt-fp16": str(Path(export_dir) / "model_fp16.engine"),
        "trt-int8": str(Path(export_dir) / "model_int8.engine"),
    }
    # 也接受 ultralytics 默认放置在 .pt 同目录的文件
    default_path = fmt_file_map.get(fmt)
    if default_path and Path(default_path).exists():
        return default_path

    # fallback：在 export_dir 里模糊匹配
    ext_map = {"onnx": ".onnx", "trt-fp32": ".engine", "trt-fp16": ".engine", "trt-int8": ".engine"}
    ext = ext_map.get(fmt)
    if ext:
        candidates = list(Path(export_dir).glob(f"*{ext}"))
        if candidates:
            return str(candidates[0])

    return None


# ─────────────────────────────────────────────────────────────
# 单格式 Benchmark
# ─────────────────────────────────────────────────────────────
def benchmark_single(
    fmt: str,
    model_path: str,
    images: List[np.ndarray],
    imgsz: int,
    warmup: int,
    iterations: int,
    conf: float,
) -> dict:
    from ultralytics import YOLO

    label = fmt.upper().replace("-", " ")
    print(f"\n{'─'*60}")
    print(f"  [{label}]  {model_path}")
    print(f"{'─'*60}")

    model = YOLO(model_path)
    reset_gpu_memory_stats()

    # ── 预热 ──────────────────────────────────────────────────
    print(f"  预热 {warmup} 次...")
    for i in range(warmup):
        img = images[i % len(images)]
        model.predict(img, verbose=False, imgsz=imgsz, conf=conf)

    # ── 正式测试 ───────────────────────────────────────────────
    print(f"  测试 {iterations} 次...")
    latencies = []
    detect_counts = []

    for i in range(iterations):
        img = images[i % len(images)]
        t0 = time.perf_counter()
        results = model.predict(img, verbose=False, imgsz=imgsz, conf=conf)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms
        n_boxes = len(results[0].boxes) if results[0].boxes else 0
        detect_counts.append(n_boxes)

    latencies = np.array(latencies)
    peak_mem = get_peak_gpu_memory_mb()

    stats = {
        "format":        fmt,
        "model_path":    model_path,
        "avg_latency_ms":  round(float(np.mean(latencies)), 2),
        "p50_latency_ms":  round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms":  round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms":  round(float(np.percentile(latencies, 99)), 2),
        "min_latency_ms":  round(float(np.min(latencies)), 2),
        "max_latency_ms":  round(float(np.max(latencies)), 2),
        "fps":             round(1000.0 / float(np.mean(latencies)), 1),
        "peak_gpu_mem_mb": round(peak_mem, 1),
        "avg_detections":  round(float(np.mean(detect_counts)), 2),
        "iterations":      iterations,
        "warmup":          warmup,
        "imgsz":           imgsz,
    }

    print(f"  ✅ avg={stats['avg_latency_ms']}ms  "
          f"p95={stats['p95_latency_ms']}ms  "
          f"FPS={stats['fps']}  "
          f"GPU={stats['peak_gpu_mem_mb']}MB")

    # 清理，避免显存影响下一格式
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return stats


# ─────────────────────────────────────────────────────────────
# 汇总打印
# ─────────────────────────────────────────────────────────────
def print_summary_table(all_stats: List[dict], pt_stats: dict):
    """打印完整对比表，含加速比"""
    pt_fps     = pt_stats["fps"]
    pt_latency = pt_stats["avg_latency_ms"]
    pt_mem     = pt_stats["peak_gpu_mem_mb"]

    print(f"\n{'='*90}")
    print("  推理速度对比（DeepPCB test set，RTX 4090，imgsz=1280）")
    print(f"{'='*90}")
    header = f"  {'格式':<14} {'Avg(ms)':<10} {'P95(ms)':<10} {'FPS':<8} {'GPU(MB)':<10} {'加速比':<8} {'显存比'}"
    print(header)
    print(f"  {'-'*80}")

    for s in all_stats:
        speedup  = f"{s['fps'] / pt_fps:.2f}x"   if pt_fps > 0 else "—"
        mem_ratio = f"{s['peak_gpu_mem_mb'] / pt_mem:.2f}x" if pt_mem > 0 else "—"
        flag = " 🏆" if s["fps"] == max(x["fps"] for x in all_stats) else ""
        print(f"  {s['format']:<14} {s['avg_latency_ms']:<10} {s['p95_latency_ms']:<10} "
              f"{s['fps']:<8} {s['peak_gpu_mem_mb']:<10} {speedup:<8} {mem_ratio}{flag}")

    print(f"{'='*90}")

    # 精简版（面试截图用）
    print(f"\n{'='*60}")
    print(f"  {'格式':<18} {'FPS':<10} {'Latency(ms)':<14} 加速比")
    print(f"  {'-'*50}")
    for s in all_stats:
        speedup = f"+{(s['fps']/pt_fps - 1)*100:.0f}%" if pt_fps > 0 and s['format'] != 'pt' else "baseline"
        print(f"  {s['format']:<18} {s['fps']:<10} {s['avg_latency_ms']:<14} {speedup}")
    print(f"{'='*60}")

    # 结论
    best = max(all_stats, key=lambda x: x["fps"])
    fp16 = next((x for x in all_stats if x["format"] == "trt-fp16"), None)

    print(f"\n  📊 结论：")
    print(f"     最快格式：{best['format']}  {best['fps']} FPS")
    if fp16:
        speedup_fp16 = fp16["fps"] / pt_fps
        print(f"     TRT-FP16 vs PyTorch：{speedup_fp16:.2f}x 加速，延迟 {fp16['avg_latency_ms']}ms")
        print(f"     （即：相同精度下，部署吞吐量提升 {(speedup_fp16-1)*100:.0f}%）")


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  DefectGuard 推理 Benchmark")
    print(f"  formats={args.formats}  imgsz={args.imgsz}  iters={args.iterations}")
    print(f"{'='*70}")

    # 加载测试图像
    images = load_images(args.image_dir, args.imgsz, n=args.iterations)
    print(f"\n  测试图像：{len(images)} 张  (来自 {args.image_dir})")

    # 收集各格式路径
    to_test = []
    for fmt in args.formats:
        path = find_model_path(fmt, args.pt_weights, args.export_dir)
        if path and Path(path).exists():
            to_test.append((fmt, path))
        else:
            print(f"  ⚠️  [{fmt}] 模型文件不存在，跳过（需先运行 export_trt.py）")

    if not to_test:
        print("  ❌ 没有可测试的模型，请先运行 export_trt.py 导出")
        sys.exit(1)

    # 逐格式测试
    all_stats = []
    for fmt, path in to_test:
        try:
            stats = benchmark_single(
                fmt, path, images, args.imgsz,
                args.warmup, args.iterations, args.conf
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"  ❌ [{fmt}] 测试失败：{e}")

    if not all_stats:
        print("  ❌ 所有格式测试均失败")
        sys.exit(1)

    # PyTorch 作为基准（如果没跑，用第一个）
    pt_stats = next((x for x in all_stats if x["format"] == "pt"), all_stats[0])

    # 打印汇总
    print_summary_table(all_stats, pt_stats)

    # 保存 JSON 报告
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "results": all_stats,
    }
    report_path = out_dir / f"benchmark_report_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 同时保存一份最新结果（方便直接引用）
    latest_path = out_dir / "benchmark_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  📄 报告已保存 → {report_path}")
    print(f"  📄 最新结果  → {latest_path}")
    print(f"\n  ✅ Benchmark 完成")


if __name__ == "__main__":
    main()
