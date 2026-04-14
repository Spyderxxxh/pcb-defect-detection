"""
Level-Set 边界精修消融实验
对比：YOLO原始框 vs SAM精修 vs SAM+LevelSet精修 vs 纯LevelSet精修

Usage:
    python ablation_levelset.py \
        --weights runs/detect/runs/detect/defectguard_v2_ultra/weights/best.pt \
        --data ./data/deeppcb/yolo_format \
        --imgsz 1280 \
        --device 0 \
        --max-images 100
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import cv2
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from src.models.sam_refinement import SAMRefinement, LevelSetRefinement


def parse_args():
    parser = argparse.ArgumentParser(description='Level-Set消融实验')
    parser.add_argument('--weights', type=str, required=True, help='YOLO权重路径')
    parser.add_argument('--data', type=str, default='./data/deeppcb/yolo_format', help='数据集路径')
    parser.add_argument('--imgsz', type=int, default=1280, help='推理分辨率')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--sam-type', type=str, default='vit_b', help='SAM模型类型')
    parser.add_argument('--sam-checkpoint', type=str, default=None, help='SAM权重路径')
    parser.add_argument('--max-images', type=int, default=100, help='评估图片数量')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--save-json', action='store_true', help='保存结果到JSON')
    parser.add_argument('--output-dir', type=str, default='outputs/levelset_ablation', help='输出目录')
    return parser.parse_args()


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-10)


def box_to_int_list(box) -> List[int]:
    """将任意格式的框坐标转为整数列表 [x1, y1, x2, y2]"""
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy()
    arr = np.array(box).flatten()
    return [int(float(arr[0])), int(float(arr[1])), int(float(arr[2])), int(float(arr[3]))]


def evaluate_method(
    yolo_model,
    refiner,
    image_paths: List[str],
    label_paths: List[str],
    imgsz: int,
    conf: float,
    device: str,
    method_name: str
) -> Dict:
    """
    评估单一方法

    Args:
        yolo_model: YOLO模型
        refiner: 精修器 (None表示不精修)
        image_paths: 图片路径列表
        label_paths: 标签路径列表
        imgsz: 推理分辨率
        conf: 置信度阈值
        device: 设备
        method_name: 方法名称

    Returns:
        评估指标字典
    """
    print(f"\n{'='*60}")
    print(f"评估方法: {method_name}")
    print(f"{'='*60}")

    total_gt = 0
    total_pred = 0
    total_tp = 0
    ious = []
    box_changes = []
    times = []

    for img_path, lbl_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc=method_name):
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # YOLO推理
        start_time = time.time()
        yolo_results = yolo_model(img_rgb, imgsz=imgsz, conf=conf, device=device, verbose=False)

        # 获取检测结果（Ultralytics xyxy 已自动还原到原图坐标）
        if len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()  # [N, 4] float
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        original_boxes = boxes.copy()

        # 精修（如果需要）
        if refiner is not None and len(boxes) > 0:
            try:
                # 准备输入：原图 tensor，框保持原图坐标
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)          # [1, C, H, W]
                boxes_tensor = torch.from_numpy(boxes).float()  # [N, 4]

                # 调用精修
                refined_results = refiner(img_tensor, boxes_tensor, original_size=(img_h, img_w))

                # 提取精修后的框，失败时保留对应原始框
                refined_boxes = []
                for idx, r in enumerate(refined_results):
                    box_raw = r.get('box')
                    if box_raw is not None:
                        try:
                            refined_boxes.append(box_to_int_list(box_raw))
                            continue
                        except Exception:
                            pass
                    # 回退：使用对应的原始 YOLO 框
                    refined_boxes.append(box_to_int_list(original_boxes[idx]))

                refined_boxes = np.array(refined_boxes, dtype=np.int32)  # [N, 4]

                # 计算框面积变化比例
                for orig, refined in zip(original_boxes, refined_boxes):
                    orig_area = max((orig[2] - orig[0]) * (orig[3] - orig[1]), 1e-6)
                    refined_area = (refined[2] - refined[0]) * (refined[3] - refined[1])
                    change_ratio = abs(refined_area - orig_area) / orig_area * 100
                    box_changes.append(change_ratio)

                boxes = refined_boxes

            except Exception as e:
                print(f"  [warn] 精修整体失败: {e}，使用原始框")
                # 不改动 boxes，使用原始 YOLO 框

        elapsed = time.time() - start_time
        times.append(elapsed)

        # 读取 GT（YOLO 格式，归一化 cx cy w h）
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        gx1 = int((cx - bw / 2) * img_w)
                        gy1 = int((cy - bh / 2) * img_h)
                        gx2 = int((cx + bw / 2) * img_w)
                        gy2 = int((cy + bh / 2) * img_h)
                        gt_boxes.append([gx1, gy1, gx2, gy2, cls])

        gt_boxes = np.array(gt_boxes) if len(gt_boxes) > 0 else np.zeros((0, 5), dtype=np.int32)

        # 计算 TP（IoU ≥ 0.5 即视为匹配）
        total_gt += len(gt_boxes)
        total_pred += len(boxes)

        if len(gt_boxes) > 0 and len(boxes) > 0:
            for gt_box in gt_boxes:
                best_iou = 0.0
                for pred_box in boxes:
                    iou = compute_iou(gt_box[:4], pred_box[:4])
                    best_iou = max(best_iou, iou)
                ious.append(best_iou)
                if best_iou >= 0.5:
                    total_tp += 1

    # 汇总指标
    precision = total_tp / (total_pred + 1e-10)
    recall = total_tp / (total_gt + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    mean_time = float(np.mean(times)) if len(times) > 0 else 0.0
    fps = 1.0 / mean_time if mean_time > 0 else 0.0
    mean_box_change = float(np.mean(box_changes)) if len(box_changes) > 0 else 0.0

    return {
        'method': method_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'fps': fps,
        'mean_time_ms': mean_time * 1000,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'total_tp': total_tp,
        'mean_box_change_percent': mean_box_change,
        'num_boxes_analyzed': len(box_changes),
    }


def print_comparison(results: List[Dict]):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("Level-Set 消融实验结果对比")
    print("=" * 80)

    print(f"{'方法':<25} {'Precision':<12} {'Recall':<10} {'F1':<10} {'mIoU':<10} {'FPS':<8} {'框变化':<12}")
    print("-" * 80)

    baseline = next((r for r in results if r['method'] == 'YOLO原始'), None)

    for r in results:
        box_change_str = f"{r.get('mean_box_change_percent', 0):.2f}%" if r.get('mean_box_change_percent', 0) > 0 else "-"
        print(f"{r['method']:<25} {r['precision']:<12.4f} {r['recall']:<10.4f} "
              f"{r['f1']:<10.4f} {r['mean_iou']:<10.4f} {r['fps']:<8.1f} {box_change_str:<12}")

    print("-" * 80)

    if baseline and len(results) >= 2:
        print("\n对比分析 (vs YOLO原始):")
        for r in results:
            if r['method'] == 'YOLO原始':
                continue
            prec_d  = (r['precision'] - baseline['precision']) * 100
            rec_d   = (r['recall']    - baseline['recall'])    * 100
            f1_d    = (r['f1']        - baseline['f1'])        * 100
            miou_d  = (r['mean_iou']  - baseline['mean_iou'])  * 100
            fps_d   = r['fps'] - baseline['fps']
            print(f"\n{r['method']}:")
            print(f"  Precision: {prec_d:+.2f} pp")
            print(f"  Recall:    {rec_d:+.2f} pp")
            print(f"  F1:        {f1_d:+.2f} pp")
            print(f"  mIoU:      {miou_d:+.2f} pp")
            print(f"  FPS:       {fps_d:+.1f}")


def make_ls_only_forward(ls_refiner):
    """
    生成一个跳过 SAM、直接用 LevelSet 精修的 forward 函数。
    ls_refiner 必须有 level_set 属性。
    """
    def ls_only_forward(image, boxes, original_size=None):
        # 图像转 numpy [H, W, C] uint8
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # boxes 转 numpy
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        results = []
        for box in boxes:
            box_int = box_to_int_list(box)
            try:
                refined = ls_refiner.level_set.refine(image, tuple(box_int))
                # refine 返回 tuple(int, int, int, int)，确认类型
                refined = tuple(int(c) for c in refined)
            except Exception as e:
                print(f"  [warn] LevelSet refine 失败: {e}，保留原始框")
                refined = tuple(box_int)
            results.append({
                'box': refined,
                'original_box': box_int,
            })
        return results

    return ls_only_forward


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"加载 YOLO 模型: {args.weights}")
    yolo_model = YOLO(args.weights)

    # 获取测试图片列表
    test_img_dir = Path(args.data) / 'images' / 'test'
    test_lbl_dir = Path(args.data) / 'labels' / 'test'

    image_paths = sorted(list(test_img_dir.glob('*.jpg')))[:args.max_images]
    label_paths = [test_lbl_dir / (p.stem + '.txt') for p in image_paths]
    print(f"评估图片数量: {len(image_paths)}")

    results = []

    # ── 实验1：YOLO 原始 ─────────────────────────────────────────
    result1 = evaluate_method(
        yolo_model, None,
        [str(p) for p in image_paths],
        [str(p) for p in label_paths],
        args.imgsz, args.conf, args.device,
        'YOLO原始'
    )
    results.append(result1)

    # ── 实验2：SAM 精修 ──────────────────────────────────────────
    print("\n加载 SAM 模型...")
    sam_refiner = SAMRefinement(
        model_type=args.sam_type,
        checkpoint=args.sam_checkpoint,
        level_set_cfg=None
    )
    sam_refiner.eval()

    result2 = evaluate_method(
        yolo_model, sam_refiner,
        [str(p) for p in image_paths],
        [str(p) for p in label_paths],
        args.imgsz, args.conf, args.device,
        'SAM精修'
    )
    results.append(result2)

    # ── 实验3：SAM + LevelSet ────────────────────────────────────
    print("\n加载 SAM + LevelSet...")
    sam_ls_refiner = SAMRefinement(
        model_type=args.sam_type,
        checkpoint=args.sam_checkpoint,
        level_set_cfg={
            'enabled': True,
            'max_iter': 100,
            'lambda1': 1.0,
            'lambda2': 1.0,
            'mu': 0.1,
            'dt': 0.1,
        }
    )
    sam_ls_refiner.eval()

    result3 = evaluate_method(
        yolo_model, sam_ls_refiner,
        [str(p) for p in image_paths],
        [str(p) for p in label_paths],
        args.imgsz, args.conf, args.device,
        'SAM+LevelSet'
    )
    results.append(result3)

    # ── 实验4：纯 LevelSet（跳过 SAM）───────────────────────────
    print("\n加载纯 LevelSet...")
    ls_refiner = SAMRefinement(
        model_type=args.sam_type,
        checkpoint=args.sam_checkpoint,
        level_set_cfg={
            'enabled': True,
            'max_iter': 100,
            'lambda1': 1.0,
            'lambda2': 1.0,
            'mu': 0.1,
            'dt': 0.1,
        }
    )
    # 用工厂函数生成 forward，避免闭包问题
    ls_refiner.forward = make_ls_only_forward(ls_refiner)

    result4 = evaluate_method(
        yolo_model, ls_refiner,
        [str(p) for p in image_paths],
        [str(p) for p in label_paths],
        args.imgsz, args.conf, args.device,
        '纯LevelSet'
    )
    results.append(result4)

    # 打印 & 保存
    print_comparison(results)

    if args.save_json:
        output_file = Path(args.output_dir) / 'levelset_ablation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {output_file}")

    print("\n" + "=" * 80)
    print("实验完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
