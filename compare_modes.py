"""
三种性能模式对比脚本
对比 high_precision / balanced / high_speed 的精度和速度
"""

import os
import sys
import json
import subprocess
from tabulate import tabulate


def run_evaluation(mode, weights, data):
    """运行评估"""
    print(f"\n{'='*60}")
    print(f"Evaluating {mode} mode...")
    print(f"{'='*60}")
    
    cmd = [
        'python', 'evaluate.py',
        '--weights', weights,
        '--data', data,
        '--mode', mode,
        '--save-results'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {mode} mode:")
        print(result.stderr)
        return None
    
    # 读取结果
    result_file = f'evaluation_{mode}.json'
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Performance Modes')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data', type=str, required=True,
                        help='验证集目录')
    args = parser.parse_args()
    
    print("="*60)
    print("DefectGuard - Performance Mode Comparison")
    print("="*60)
    
    modes = ['high_precision', 'balanced', 'high_speed']
    results = {}
    
    for mode in modes:
        result = run_evaluation(mode, args.weights, args.data)
        if result:
            results[mode] = result['metrics']
    
    if len(results) < 3:
        print("\nWarning: Not all modes completed successfully!")
    
    # 打印对比表格
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)
    
    table_data = []
    for mode in modes:
        if mode in results:
            m = results[mode]
            table_data.append([
                mode,
                f"{m['map50']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['fps']:.1f}"
            ])
    
    headers = ['Mode', 'mAP@0.5', 'Precision', 'Recall', 'F1-Score', 'FPS']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # 分析结论
    print("\n" + "="*60)
    print("Analysis")
    print("="*60)
    
    if 'high_precision' in results and 'high_speed' in results:
        hp = results['high_precision']
        hs = results['high_speed']
        
        map_diff = (hp['map50'] - hs['map50']) / hs['map50'] * 100
        fps_diff = (hs['fps'] - hp['fps']) / hp['fps'] * 100
        
        print(f"High Precision vs High Speed:")
        print(f"  - mAP difference: {map_diff:+.1f}%")
        print(f"  - Speed difference: {fps_diff:+.1f}%")
        
        if map_diff > 5:
            print(f"  → High Precision mode provides significantly better accuracy")
        elif map_diff < -2:
            print(f"  → High Speed mode has comparable or better accuracy")
        else:
            print(f"  → Trade-off between modes is minimal")
    
    print("\nRecommendation:")
    if 'balanced' in results:
        bal = results['balanced']
        print(f"  Balanced mode: mAP={bal['map50']:.4f}, FPS={bal['fps']:.1f}")
        print(f"  → Recommended for most use cases")


if __name__ == "__main__":
    main()
