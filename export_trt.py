"""
TensorRT / ONNX 导出脚本
支持：PyTorch .pt → ONNX → TensorRT (FP32 / FP16 / INT8)

用法示例：
  # 导出 ONNX
  python export_trt.py --weights runs/detect/xxx/weights/best.pt --format onnx

  # 导出 TensorRT FP16（推荐）
  python export_trt.py --weights runs/detect/xxx/weights/best.pt --format trt-fp16

  # 导出全部格式并对比
  python export_trt.py --weights runs/detect/xxx/weights/best.pt --format all

  # INT8 需要校准数据
  python export_trt.py --weights runs/detect/xxx/weights/best.pt --format trt-int8 \
      --calib-data data/deeppcb/yolo_format/images/val
"""

import os
import sys
import time
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np

# ─────────────────────────────────────────────────────────────
# CUDA 环境修复（必须在 import torch 之前）
# 空字符串 "" 会让 CUDA 认为没有 GPU，需提前修正
# ─────────────────────────────────────────────────────────────
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if _cvd == "":
    # 空字符串 = 屏蔽所有 GPU，修正为 "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("  ⚠️  检测到 CUDA_VISIBLE_DEVICES='' (空字符串)，已自动修正为 '0'")


# ─────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 → ONNX / TensorRT")
    parser.add_argument(
        "--weights", type=str,
        default="runs/detect/runs/detect/defectguard_v2_ultra/weights/best.pt",
        help="PyTorch 权重路径 (.pt)"
    )
    parser.add_argument(
        "--format", type=str, default="all",
        choices=["onnx", "trt-fp32", "trt-fp16", "trt-int8", "all"],
        help="导出格式：onnx / trt-fp32 / trt-fp16 / trt-int8 / all"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="导出时的输入尺寸（需与训练一致）"
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="导出 batch size（推理时固定 batch=1 最常用）"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="GPU 设备 ID"
    )
    parser.add_argument(
        "--calib-data", type=str,
        default="data/deeppcb/yolo_format/images/val",
        help="INT8 校准图像目录（仅 trt-int8 时使用）"
    )
    parser.add_argument(
        "--calib-num", type=int, default=200,
        help="INT8 校准图像数量"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/exported_models",
        help="导出产物保存目录"
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="导出后自动做精度一致性验证（对比 PyTorch 输出）"
    )
    parser.add_argument(
        "--simplify", action="store_true", default=True,
        help="ONNX 导出时使用 onnxsim 化简计算图"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
def get_file_size_mb(path: str) -> float:
    """返回文件大小（MB）"""
    return Path(path).stat().st_size / 1024 / 1024


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def save_report(report: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out / f"export_report_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 导出报告已保存 → {report_path}")
    return str(report_path)


# ─────────────────────────────────────────────────────────────
# 导出函数
# ─────────────────────────────────────────────────────────────
def export_onnx(model, weights_path: str, imgsz: int, batch: int,
                output_dir: str, simplify: bool = True) -> dict:
    """导出 ONNX 格式"""
    print_section("导出 ONNX")
    out_path = str(Path(output_dir) / "model_fp32.onnx")

    t0 = time.time()
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch,
        simplify=simplify,
        opset=17,           # 兼容 TensorRT 8.x+
        dynamic=False,      # 固定 batch=1 速度更快
    )
    elapsed = time.time() - t0

    # YOLOv8 export 把文件放在 weights 同目录，移到 output_dir
    src = str(exported) if exported else weights_path.replace(".pt", ".onnx")
    if Path(src).exists() and src != out_path:
        shutil.copy2(src, out_path)

    size_mb = get_file_size_mb(out_path) if Path(out_path).exists() else 0
    print(f"  ✅ ONNX 导出完成")
    print(f"     文件：{out_path}")
    print(f"     大小：{size_mb:.1f} MB")
    print(f"     耗时：{elapsed:.1f}s")

    return {
        "format": "onnx",
        "path": out_path,
        "size_mb": round(size_mb, 2),
        "export_time_s": round(elapsed, 2),
        "imgsz": imgsz,
        "batch": batch,
        "simplify": simplify,
    }


def export_trt(model, weights_path: str, precision: str,
               imgsz: int, batch: int, output_dir: str,
               device: str = "0",
               calib_data: str = None, calib_num: int = 200) -> dict:
    """
    导出 TensorRT 引擎
    precision: 'fp32' | 'fp16' | 'int8'
    """
    import torch

    # ─────────────────────────────────────────────────────────────
    # 每次调用前都检查并修复 CUDA 环境（Ultralytics 可能在运行中把变量改成 ""）
    # ─────────────────────────────────────────────────────────────
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if _cvd == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("  ⚠️  修复 CUDA_VISIBLE_DEVICES='' → '0'")

    # 同步 LD_LIBRARY_PATH（conda activate.d 只在 conda activate 时执行）
    _lib_path = "/usr/lib/x86_64-linux-gnu"
    if _lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = f"{_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        print(f"  ⚠️  添加 LD_LIBRARY_PATH={_lib_path}")

    print_section(f"导出 TensorRT {precision.upper()}")

    # CUDA 可用性检查
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA 不可用（torch.cuda.is_available()=False）\n"
            f"  torch.cuda.device_count(): {torch.cuda.device_count()}\n"
            f"  os.environ['CUDA_VISIBLE_DEVICES']: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}\n"
            f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')}\n"
            f"  请确认：(1) 驱动已安装 (2) conda activate mvp 已执行\n"
            f"  尝试：conda activate mvp && export CUDA_VISIBLE_DEVICES=0 && python export_trt.py ..."
        )

    # device 统一为整数形式（Ultralytics 更兼容）
    dev = int(device) if str(device).isdigit() else 0
    print(f"  使用 GPU: device={dev}  ({torch.cuda.get_device_name(dev)})")

    suffix_map = {"fp32": "fp32", "fp16": "fp16", "int8": "int8"}
    out_path = str(Path(output_dir) / f"model_{suffix_map[precision]}.engine")

    kwargs = dict(
        format="engine",
        imgsz=imgsz,
        batch=batch,
        device=dev,
    )
    if precision == "fp16":
        kwargs["half"] = True
    elif precision == "int8":
        kwargs["int8"] = True
        if calib_data and Path(calib_data).exists():
            kwargs["data"] = calib_data
        else:
            print(f"  ⚠️  INT8 校准数据目录不存在：{calib_data}")
            print(f"     将使用随机数据校准（精度可能略低）")

    t0 = time.time()
    exported = model.export(**kwargs)
    elapsed = time.time() - t0

    src = str(exported) if exported else weights_path.replace(".pt", ".engine")
    if Path(src).exists() and src != out_path:
        shutil.copy2(src, out_path)

    size_mb = get_file_size_mb(out_path) if Path(out_path).exists() else 0
    print(f"  ✅ TensorRT {precision.upper()} 导出完成")
    print(f"     文件：{out_path}")
    print(f"     大小：{size_mb:.1f} MB")
    print(f"     耗时：{elapsed:.1f}s")

    return {
        "format": f"trt-{precision}",
        "path": out_path,
        "size_mb": round(size_mb, 2),
        "export_time_s": round(elapsed, 2),
        "imgsz": imgsz,
        "batch": batch,
        "precision": precision,
    }


# ─────────────────────────────────────────────────────────────
# 精度一致性验证
# ─────────────────────────────────────────────────────────────
def verify_consistency(weights_pt: str, exported_path: str,
                        imgsz: int, n_samples: int = 10) -> dict:
    """
    用随机输入对比 PyTorch 与导出模型的输出差异（box confidence）
    """
    from ultralytics import YOLO
    import torch

    print(f"\n  🔍 精度一致性验证（{n_samples} 个随机输入）...")

    model_pt = YOLO(weights_pt)
    model_exp = YOLO(exported_path)

    diffs = []
    for _ in range(n_samples):
        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        r_pt  = model_pt.predict(dummy, verbose=False, imgsz=imgsz)
        r_exp = model_exp.predict(dummy, verbose=False, imgsz=imgsz)

        confs_pt  = sorted([float(b.conf) for b in r_pt[0].boxes],  reverse=True)[:5] if r_pt[0].boxes else []
        confs_exp = sorted([float(b.conf) for b in r_exp[0].boxes], reverse=True)[:5] if r_exp[0].boxes else []

        # 对齐长度后计算差值
        n = min(len(confs_pt), len(confs_exp))
        if n > 0:
            diff = np.mean(np.abs(np.array(confs_pt[:n]) - np.array(confs_exp[:n])))
            diffs.append(float(diff))

    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    status = "✅ 一致" if avg_diff < 0.02 else ("⚠️  轻微偏差" if avg_diff < 0.05 else "❌ 偏差较大")
    print(f"     平均 confidence 差异：{avg_diff:.4f}  {status}")

    return {"avg_conf_diff": round(avg_diff, 5), "status": status, "n_samples": n_samples}


# ─────────────────────────────────────────────────────────────
# 打印汇总
# ─────────────────────────────────────────────────────────────
def print_summary(results: list, pt_size_mb: float):
    """打印导出产物汇总表"""
    print_section("导出结果汇总")
    print(f"  {'格式':<18} {'文件大小(MB)':<16} {'导出耗时(s)':<14} {'vs PyTorch 大小'}")
    print(f"  {'-'*65}")
    print(f"  {'PyTorch (.pt)':<18} {pt_size_mb:<16.1f} {'—':<14} —")
    for r in results:
        ratio = r["size_mb"] / pt_size_mb if pt_size_mb > 0 else 0
        print(f"  {r['format']:<18} {r['size_mb']:<16.1f} {r['export_time_s']:<14.1f} {ratio:.2f}x")

    print(f"\n  💡 下一步：运行 benchmark_inference.py 对比各格式推理速度")
    print(f"     python benchmark_inference.py --output-dir outputs/exported_models")


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────
def main():
    # ─────────────────────────────────────────────────────────────
    # CUDA 环境修复（必须在 import torch/ultralytics 之前！）
    # ─────────────────────────────────────────────────────────────
    import torch
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if _cvd == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("  ⚠️  修复 CUDA_VISIBLE_DEVICES='' → '0'")

    _lib_path = "/usr/lib/x86_64-linux-gnu"
    if _lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = f"{_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        print(f"  ⚠️  添加 LD_LIBRARY_PATH={_lib_path}")

    # 验证 CUDA 可用
    if not torch.cuda.is_available():
        print(f"  ⚠️  torch.cuda.is_available()={torch.cuda.is_available()}，但 device_count={torch.cuda.device_count()}")
        print(f"     可能是 Ultralytics 导出时会重新检测，继续尝试...")

    args = parse_args()

    # 检查权重文件
    weights_path = args.weights
    if not Path(weights_path).exists():
        # 尝试自动查找
        candidates = list(Path("runs/detect").rglob("best.pt"))
        if candidates:
            weights_path = str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
            print(f"  ⚠️  指定权重不存在，自动使用最新权重：{weights_path}")
        else:
            print(f"  ❌ 找不到权重文件：{weights_path}")
            print(f"     请通过 --weights 参数指定正确路径")
            sys.exit(1)

    pt_size_mb = get_file_size_mb(weights_path)
    print_section("DefectGuard 模型导出")
    print(f"  权重：{weights_path}  ({pt_size_mb:.1f} MB)")
    print(f"  格式：{args.format}")
    print(f"  imgsz：{args.imgsz}   batch：{args.batch}")

    # 导入 YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    # 创建输出目录
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"\n  加载模型...")
    model = YOLO(weights_path)

    # 确定要导出的格式列表
    fmt = args.format
    if fmt == "all":
        formats = ["onnx", "trt-fp32", "trt-fp16", "trt-int8"]
    else:
        formats = [fmt]

    export_results = []
    report = {
        "timestamp": datetime.now().isoformat(),
        "weights": weights_path,
        "pt_size_mb": round(pt_size_mb, 2),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "exports": [],
    }

    for fmt_item in formats:
        try:
            if fmt_item == "onnx":
                r = export_onnx(model, weights_path, args.imgsz, args.batch,
                                output_dir, args.simplify)
            elif fmt_item in ("trt-fp32", "trt-fp16", "trt-int8"):
                precision = fmt_item.replace("trt-", "")
                r = export_trt(model, weights_path, precision,
                               args.imgsz, args.batch, output_dir,
                               args.device,
                               args.calib_data, args.calib_num)
            else:
                print(f"  ⚠️  未知格式：{fmt_item}，跳过")
                continue

            # 精度一致性验证
            if args.verify and Path(r["path"]).exists():
                ver = verify_consistency(weights_path, r["path"], args.imgsz)
                r["consistency"] = ver

            export_results.append(r)
            report["exports"].append(r)

        except Exception as e:
            print(f"\n  ❌ 导出 {fmt_item} 失败：{e}")
            report["exports"].append({"format": fmt_item, "error": str(e)})

    # 汇总
    if export_results:
        print_summary(export_results, pt_size_mb)

    # 保存报告
    save_report(report, output_dir)

    print(f"\n{'='*70}")
    print(f"  ✅ 全部导出完成，产物目录：{output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
