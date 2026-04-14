"""
SAM 细化前后对比可视化脚本

功能：
- 加载 YOLO + SAM 流水线
- 对比展示原始检测框 vs SAM 细化后的检测框
- 保存对比图片

用法:
    python visualize_sam_comparison.py \
        --weights runs/detect/runs/detect/defectguard_v2_ultra/weights/best.pt \
        --image-path data/deeppcb/yolo_format/images/val/xxx.jpg \
        --sam-type vit_b \
        --save-dir outputs/sam_comparison
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

from src.models.sam_refinement import SAMRefinement


def parse_args():
    parser = argparse.ArgumentParser(description='SAM 细化前后对比可视化')
    parser.add_argument('--weights', type=str, required=True, help='YOLO 权重路径')
    parser.add_argument('--image-path', type=str, required=True, help='输入图片路径')
    parser.add_argument('--sam-type', type=str, default='vit_b', choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument('--sam-checkpoint', type=str, default=None, help='SAM 权重路径')
    parser.add_argument('--imgsz', type=int, default=1280, help='推理分辨率')
    parser.add_argument('--save-dir', type=str, default='outputs/sam_comparison', help='保存目录')
    parser.add_argument('--device', type=str, default='0', help='设备')
    return parser.parse_args()


def draw_boxes(img, boxes, labels, scores, color, thickness=2):
    """绘制检测框"""
    img_draw = img.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        text = f'{label}: {score:.2f}'
        cv2.putText(img_draw, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_draw


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"加载 YOLO 模型: {args.weights}")
    yolo = YOLO(args.weights)
    
    print(f"加载 SAM 模型: {args.sam_type}")
    sam_refine = SAMRefinement(model_type=args.sam_type, checkpoint=args.sam_checkpoint)
    sam_refine.eval()
    # 处理设备字符串
    device = args.device if 'cuda' in args.device else f"cuda:{args.device}"
    sam_refine = sam_refine.to(device)
    
    # 读取图片
    img = cv2.imread(args.image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {args.image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 处理设备字符串
    device = args.device if 'cuda' in args.device else f"cuda:{args.device}"
    
    # 转换为 YOLO 需要的格式 (C, H, W)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # YOLO 推理
    results = yolo.predict(img, imgsz=args.imgsz, device=args.device, verbose=False)[0]
    
    # 提取检测结果
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    labels = [results.names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]
    
    print(f"检测到 {len(boxes)} 个缺陷")
    
    if len(boxes) == 0:
        print("没有检测到缺陷!")
        return
    
    # 原始检测框（绿色）
    img_original = draw_boxes(img, boxes, labels, scores, (0, 255, 0), 3)
    
    # SAM 细化
    print("正在 SAM 细化...")
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)
    refined_results = sam_refine.forward(img_tensor, boxes_tensor, original_size=(img_rgb.shape[0], img_rgb.shape[1]))
    
    # 提取细化后的框
    refined_boxes = []
    for r in refined_results:
        refined_boxes.append(r['box'])
    refined_boxes = np.array(refined_boxes)
    
    # SAM 细化后（红色）
    img_sam = draw_boxes(img.copy(), refined_boxes, labels, scores, (0, 0, 255), 3)
    
    # 对比图（左右拼接）
    h, w = img.shape[:2]
    comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    comparison[:, :w] = img_original
    comparison[:, w + 20:] = img_sam
    
    # 添加文字标签
    cv2.putText(comparison, 'Original (YOLO)', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(comparison, 'SAM Refined', (w + 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # 保存
    output_path = save_dir / f"{Path(args.image_path).stem}_comparison.jpg"
    cv2.imwrite(str(output_path), comparison)
    print(f"对比图保存至: {output_path}")
    
    # 打印框大小对比
    print("\n框大小对比:")
    print(f"{'类别':<10} {'原始框面积':>12} {'SAM框面积':>12} {'缩小比例':>10}")
    print("-" * 50)
    for i, (box, refined, label) in enumerate(zip(boxes, refined_boxes, labels)):
        orig_area = (box[2] - box[0]) * (box[3] - box[1])
        sam_area = (refined[2] - refined[0]) * (refined[3] - refined[1])
        reduction = (1 - sam_area / orig_area) * 100
        print(f"{label:<10} {orig_area:>12.0f} {sam_area:>12.0f} {reduction:>9.1f}%")


if __name__ == '__main__':
    main()