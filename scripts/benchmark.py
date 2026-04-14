"""
基准测试脚本
测试不同性能模式下的推理速度和精度
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import load_config
from inference import DefectGuardInference


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark DefectGuard')
    parser.add_argument('--weights', type=str, default='./outputs/defectguard_train/weights/best.pt',
                        help='模型权重')
    parser.add_argument('--config', type=str, default='configs/defectguard_yolov8.yaml',
                        help='配置文件')
    parser.add_argument('--data', type=str, default='./data/pcb_defect/images/test',
                        help='测试数据目录')
    parser.add_argument('--device', type=str, default='0',
                        help='设备')
    parser.add_argument('--iterations', type=int, default=100,
                        help='测试迭代次数')
    parser.add_argument('--warmup', type=int, default=10,
                        help='预热迭代次数')
    return parser.parse_args()


def benchmark_mode(inferencer, images, mode, iterations, warmup):
    """测试单个模式"""
    print(f"\n{'='*60}")
    print(f"Benchmarking Mode: {mode}")
    print(f"{'='*60}")
    
    # 切换模式
    inferencer.switch_mode(mode)
    
    # 预热
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        img_path = np.random.choice(images)
        inferencer.predict(img_path)
    
    # 重置计时器
    inferencer.inference_times = []
    inferencer.preprocess_times = []
    inferencer.postprocess_times = []
    
    # 正式测试
    print(f"Running benchmark ({iterations} iterations)...")
    
    all_detections = []
    
    for i in tqdm(range(iterations)):
        img_path = images[i % len(images)]
        detections = inferencer.predict(img_path)
        all_detections.append(len(detections))
    
    # 计算统计
    stats = {
        'mode': mode,
        'iterations': iterations,
        'avg_inference_time_ms': np.mean(inferencer.inference_times) * 1000,
        'std_inference_time_ms': np.std(inferencer.inference_times) * 1000,
        'min_inference_time_ms': np.min(inferencer.inference_times) * 1000,
        'max_inference_time_ms': np.max(inferencer.inference_times) * 1000,
        'p50_inference_time_ms': np.percentile(inferencer.inference_times, 50) * 1000,
        'p95_inference_time_ms': np.percentile(inferencer.inference_times, 95) * 1000,
        'p99_inference_time_ms': np.percentile(inferencer.inference_times, 99) * 1000,
        'fps': 1.0 / np.mean(inferencer.inference_times),
        'avg_preprocessing_ms': np.mean(inferencer.preprocess_times) * 1000 if inferencer.preprocess_times else 0,
        'avg_postprocessing_ms': np.mean(inferencer.postprocess_times) * 1000 if inferencer.postprocess_times else 0,
        'avg_detections': np.mean(all_detections),
    }
    
    return stats


def main():
    args = parse_args()
    
    # 加载图像列表
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data path not found: {args.data}")
        print("Please provide valid test images")
        return
    
    if data_path.is_dir():
        images = [
            str(f) for f in data_path.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ]
    else:
        images = [str(data_path)]
    
    if not images:
        print("No images found")
        return
    
    print(f"Found {len(images)} test images")
    
    # 创建推理器
    inferencer = DefectGuardInference(
        weights=args.weights,
        config_path=args.config,
        device=args.device,
        mode='balanced'
    )
    
    # 测试三种模式
    modes = ['high_precision', 'balanced', 'high_speed']
    all_stats = {}
    
    for mode in modes:
        stats = benchmark_mode(
            inferencer,
            images,
            mode,
            args.iterations,
            args.warmup
        )
        all_stats[mode] = stats
        
        # 打印结果
        print(f"\nResults for {mode}:")
        print(f"  Avg inference time: {stats['avg_inference_time_ms']:.1f}ms")
        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  P95 latency: {stats['p95_inference_time_ms']:.1f}ms")
        print(f"  Avg detections: {stats['avg_detections']:.1f}")
    
    # 保存结果
    output_dir = Path('./outputs/benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # 打印对比表
    print(f"\n{'='*80}")
    print("Performance Comparison")
    print(f"{'='*80}")
    print(f"{'Mode':<15} {'FPS':<10} {'Latency (ms)':<15} {'P95 (ms)':<15}")
    print(f"{'-'*80}")
    
    for mode in modes:
        stats = all_stats[mode]
        print(f"{mode:<15} {stats['fps']:<10.1f} "
              f"{stats['avg_inference_time_ms']:<15.1f} "
              f"{stats['p95_inference_time_ms']:<15.1f}")
    
    print(f"\nResults saved to: {output_dir / 'benchmark_results.json'}")


if __name__ == '__main__':
    main()
