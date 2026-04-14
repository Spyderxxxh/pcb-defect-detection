"""
SAM边界精修可视化对比工具
放大展示精修前后的边界差异
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import sys
sys.path.append('.')
from src.models.sam_refinement import SAMRefinement
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize SAM Refinement')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--weights', type=str, required=True, help='YOLO模型权重')
    parser.add_argument('--sam-checkpoint', type=str, default='sam_vit_b.pth',
                        help='SAM模型权重')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--output', type=str, default='refinement_comparison.jpg',
                        help='输出图像路径')
    parser.add_argument('--zoom-size', type=int, default=200,
                        help='放大区域尺寸')
    return parser.parse_args()


def draw_box_with_label(img, bbox, label, color, thickness=2):
    """绘制带标签的边界框"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # 标签背景
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(img, (x1, y1 - label_size[1] - 4),
                  (x1 + label_size[0], y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img


def create_zoomed_comparison(img, bbox_original, bbox_refined, zoom_size=200):
    """创建局部放大对比图"""
    h, w = img.shape[:2]
    
    # 计算两个框的中心点
    cx1 = int((bbox_original[0] + bbox_original[2]) / 2)
    cy1 = int((bbox_original[1] + bbox_original[3]) / 2)
    cx2 = int((bbox_refined[0] + bbox_refined[2]) / 2)
    cy2 = int((bbox_refined[1] + bbox_refined[3]) / 2)
    
    # 取两个中心的平均作为放大区域中心
    cx = int((cx1 + cx2) / 2)
    cy = int((cy1 + cy2) / 2)
    
    # 计算放大区域
    x1 = max(0, cx - zoom_size // 2)
    y1 = max(0, cy - zoom_size // 2)
    x2 = min(w, x1 + zoom_size)
    y2 = min(h, y1 + zoom_size)
    
    # 裁剪区域
    roi = img[y1:y2, x1:x2].copy()
    
    # 在ROI上绘制两个框（相对坐标）
    roi_h, roi_w = roi.shape[:2]
    
    # 原始框（红色）
    ox1 = int(bbox_original[0] - x1)
    oy1 = int(bbox_original[1] - y1)
    ox2 = int(bbox_original[2] - x1)
    oy2 = int(bbox_original[3] - y1)
    cv2.rectangle(roi, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
    cv2.putText(roi, "YOLO", (ox1, max(oy1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # 精修框（绿色）
    rx1 = int(bbox_refined[0] - x1)
    ry1 = int(bbox_refined[1] - y1)
    rx2 = int(bbox_refined[2] - x1)
    ry2 = int(bbox_refined[3] - y1)
    cv2.rectangle(roi, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
    cv2.putText(roi, "SAM", (rx1, max(ry1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # 放大2倍
    roi_zoomed = cv2.resize(roi, (roi_w * 2, roi_h * 2), interpolation=cv2.INTER_CUBIC)
    
    return roi_zoomed, (x1, y1, x2, y2)


def main():
    args = parse_args()
    
    print("="*60)
    print("SAM Boundary Refinement Visualization")
    print("="*60)
    
    # 1. 加载图像
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Loaded image: {img.shape}")
    
    # 2. YOLO检测
    print("\nRunning YOLO detection...")
    model = YOLO(args.weights)
    results = model(args.image, conf=args.conf)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            detections.append({
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'confidence': float(box.conf),
                'class_id': int(box.cls),
                'class_name': result.names[int(box.cls)]
            })
    
    print(f"Detected {len(detections)} defects")
    
    if len(detections) == 0:
        print("No detections to refine!")
        return
    
    # 3. SAM精修
    print("\nRunning SAM refinement...")
    sam = SAMRefinement(
        model_type='vit_b',
        checkpoint=args.sam_checkpoint,
        level_set_cfg={
            'enabled': True,
            'max_iter': 150,
            'lambda1': 1.0,
            'lambda2': 1.0,
            'mu': 0.05,
            'dt': 0.05
        }
    )
    
    boxes = [d['bbox'] for d in detections]
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    
    try:
        refined = sam.forward(img_rgb, boxes_tensor)
        
        # 更新检测结果
        for i, ref in enumerate(refined):
            if i < len(detections):
                detections[i]['bbox_refined'] = ref['box']
                detections[i]['mask'] = ref.get('mask')
        
        print(f"Refined {len(refined)} detections")
    except Exception as e:
        print(f"SAM refinement failed: {e}")
        return
    
    # 4. 创建对比可视化
    print("\nCreating visualization...")
    
    # 创建三栏对比图
    h, w = img.shape[:2]
    
    # 左栏：原图
    col1 = img.copy()
    
    # 中栏：YOLO检测结果（红色框）
    col2 = img.copy()
    for det in detections:
        col2 = draw_box_with_label(col2, det['bbox'], 
                                   f"{det['class_name']}:{det['confidence']:.2f}",
                                   (0, 0, 255), 2)
    
    # 右栏：SAM精修结果（绿色框）
    col3 = img.copy()
    for det in detections:
        if 'bbox_refined' in det:
            col3 = draw_box_with_label(col3, det['bbox_refined'],
                                       f"{det['class_name']}:{det['confidence']:.2f}",
                                       (0, 255, 0), 2)
    
    # 添加标题栏
    title_h = 40
    header_color = (50, 50, 50)
    
    def add_header(img, text):
        header = np.full((title_h, img.shape[1], 3), header_color, dtype=np.uint8)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (title_h + text_size[1]) // 2
        cv2.putText(header, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return np.vstack([header, img])
    
    col1 = add_header(col1, "Original")
    col2 = add_header(col2, "YOLO Detection (Red)")
    col3 = add_header(col3, "SAM Refined (Green)")
    
    # 水平拼接
    main_comparison = np.hstack([col1, col2, col3])
    
    # 5. 为每个检测创建局部放大对比
    zoom_comparisons = []
    for i, det in enumerate(detections[:4]):  # 最多展示4个
        if 'bbox_refined' in det:
            zoom, coords = create_zoomed_comparison(
                img, det['bbox'], det['bbox_refined'], args.zoom_size
            )
            
            # 添加标题
            zoom_h, zoom_w = zoom.shape[:2]
            header = np.full((30, zoom_w, 3), (40, 40, 40), dtype=np.uint8)
            title = f"Defect {i+1}: {det['class_name']} (Zoomed 2x)"
            text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (zoom_w - text_size[0]) // 2
            cv2.putText(header, title, (text_x, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            zoom = np.vstack([header, zoom])
            zoom_comparisons.append(zoom)
    
    # 6. 组合最终输出
    if zoom_comparisons:
        # 将放大图排列成一行
        max_zoom_h = max(z.shape[0] for z in zoom_comparisons)
        max_zoom_w = max(z.shape[1] for z in zoom_comparisons)
        
        # 统一尺寸
        zooms_resized = []
        for zoom in zoom_comparisons:
            zooms_resized.append(cv2.resize(zoom, (max_zoom_w, max_zoom_h)))
        
        zoom_row = np.hstack(zooms_resized)
        
        # 添加分隔
        separator = np.full((20, main_comparison.shape[1], 3), (30, 30, 30), dtype=np.uint8)
        
        # 调整zoom_row宽度匹配主图
        if zoom_row.shape[1] != main_comparison.shape[1]:
            zoom_row = cv2.resize(zoom_row, (main_comparison.shape[1], 
                                              int(zoom_row.shape[0] * main_comparison.shape[1] / zoom_row.shape[1])))
        
        final_output = np.vstack([main_comparison, separator, zoom_row])
    else:
        final_output = main_comparison
    
    # 7. 保存结果
    cv2.imwrite(args.output, final_output)
    print(f"\nVisualization saved to: {args.output}")
    
    # 8. 打印精修统计
    print("\nRefinement Statistics:")
    for i, det in enumerate(detections):
        if 'bbox_refined' in det:
            orig = det['bbox']
            refined = det['bbox_refined']
            
            # 计算框的变化
            orig_w = orig[2] - orig[0]
            orig_h = orig[3] - orig[1]
            ref_w = refined[2] - refined[0]
            ref_h = refined[3] - refined[1]
            
            dx = refined[0] - orig[0]
            dy = refined[1] - orig[1]
            dw = ref_w - orig_w
            dh = ref_h - orig_h
            
            print(f"  Defect {i+1} ({det['class_name']}):")
            print(f"    Offset: ({dx:+.1f}, {dy:+.1f}) px")
            print(f"    Size change: ({dw:+.1f}, {dh:+.1f}) px")


if __name__ == "__main__":
    main()
