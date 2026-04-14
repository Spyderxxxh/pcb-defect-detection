"""
批量训练脚本：用不同随机种子训练多个模型，用于后续集成
训练完成后自动给出集成评估命令

Usage:
    # 训练 3 个不同 seed 的模型（推荐）
    python train_seed.py --seeds 42 123 456

    # 指定模型大小和 epoch
    python train_seed.py --seeds 42 123 --model yolov8l --epochs 300

    # 快速验证（50 epoch）
    python train_seed.py --seeds 42 123 --epochs 50 --name-prefix test
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds',       type=int, nargs='+', default=[42, 123, 456],
                   help='随机种子列表，每个种子训练一个模型')
    p.add_argument('--model',       type=str, default='yolov8l',
                   help='模型类型')
    p.add_argument('--epochs',      type=int, default=300)
    p.add_argument('--imgsz',       type=int, default=1280)
    p.add_argument('--batch',       type=int, default=8)
    p.add_argument('--data',        type=str, default='data/deeppcb/dataset.yaml')
    p.add_argument('--device',      type=str, default='0')
    p.add_argument('--name-prefix', type=str, default='defectguard_v2',
                   help='实验名前缀，最终名为 {prefix}_seed{seed}')
    p.add_argument('--image-dir',   type=str,
                   default='data/deeppcb/yolo_format/images/test')
    p.add_argument('--label-dir',   type=str,
                   default='data/deeppcb/yolo_format/labels/test')
    return p.parse_args()


def main():
    args = parse_args()
    
    print("=" * 65)
    print("  DefectGuard — 多 Seed 批量训练（用于集成）")
    print("=" * 65)
    print(f"  Seeds  : {args.seeds}")
    print(f"  Model  : {args.model}")
    print(f"  Epochs : {args.epochs}")
    print(f"  imgsz  : {args.imgsz}")
    print()
    
    trained_weights = []
    
    for seed in args.seeds:
        name = f"{args.name_prefix}_seed{seed}"
        weight_path = f"runs/detect/{name}/weights/best.pt"
        
        print(f"\n{'='*65}")
        print(f"  训练 seed={seed}  →  {name}")
        print(f"{'='*65}")
        
        # 检查是否已经训练过
        if Path(weight_path).exists():
            print(f"  ⚡ 已存在权重 {weight_path}，跳过训练")
            trained_weights.append(weight_path)
            continue
        
        # 调用 train_v2_ultra.py（复用已有训练配置）
        cmd = [
            sys.executable, "train_v2_ultra.py",
            "--model",   args.model,
            "--epochs",  str(args.epochs),
            "--imgsz",   str(args.imgsz),
            "--batch",   str(args.batch),
            "--data",    args.data,
            "--device",  args.device,
            "--name",    name,
            "--seed",    str(seed),
        ]
        
        print(f"  命令: {' '.join(cmd)}")
        ret = subprocess.run(cmd)
        
        if ret.returncode == 0 and Path(weight_path).exists():
            print(f"  ✅ 训练完成，权重: {weight_path}")
            trained_weights.append(weight_path)
        else:
            print(f"  ❌ 训练失败（seed={seed}），跳过")
    
    # 输出集成评估命令
    print(f"\n{'='*65}")
    print(f"  🎉 全部训练完成！共 {len(trained_weights)} 个权重")
    print(f"{'='*65}")
    
    if len(trained_weights) >= 2:
        weights_str = " \\\n          ".join(f"'{w}'" for w in trained_weights)
        print(f"\n  下一步：运行集成评估")
        print(f"\n  python eval_ensemble.py \\")
        print(f"      --weights {weights_str} \\")
        print(f"      --image-dir {args.image_dir} \\")
        print(f"      --label-dir {args.label_dir} \\")
        print(f"      --imgsz {args.imgsz} \\")
        print(f"      --wbf-iou 0.55")
    
    # 保存权重列表
    out = {'seeds': args.seeds, 'weights': trained_weights, 'model': args.model}
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/ensemble_weights.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  权重列表已保存 → outputs/ensemble_weights.json")


if __name__ == "__main__":
    main()
