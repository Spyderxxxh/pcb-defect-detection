"""
推理部署脚本
支持单图推理、批量推理、视频推理
支持三种性能模式切换
"""

import os
import sys
import argparse
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Tuple
import json

from src.core.config import load_config
from src.models import YOLOv8NonLocal, SAMRefinement
from src.agents import PipelineAgentSystem


def parse_args():
    parser = argparse.ArgumentParser(description='DefectGuard Inference')
    parser.add_argument('--source', type=str, required=True,
                        help='输入源：图像路径、目录、视频路径或摄像头ID')
    parser.add_argument('--weights', type=str, default='./outputs/defectguard_train/weights/best.pt',
                        help='模型权重路径')
    parser.add_argument('--config', type=str, default='configs/defectguard_yolov8.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, default='high_precision',
                        choices=['high_precision', 'balanced', 'high_speed'],
                        help='性能模式')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='设备')
    parser.add_argument('--save', action='store_true',
                        help='保存结果')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    parser.add_argument('--agent', action='store_true',
                        help='启用Agent系统')
    return parser.parse_args()


class DefectGuardInference:
    """
    DefectGuard推理类
    支持YOLOv8 + NonLocal + SAM完整流程
    """
    
    def __init__(
        self,
        weights: str,
        config_path: str,
        device: str = '0',
        mode: str = 'balanced'
    ):
        self.weights = weights
        self.config = load_config(config_path)
        self.device = self._setup_device(device)
        self.mode = mode
        
        # 加载模型
        self.model = self._load_model()
        
        # SAM精修（可选）
        self.sam = self._load_sam()
        
        # Agent系统（可选）
        self.agent_system = None
        
        # 性能统计
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
    
    def _setup_device(self, device_str: str) -> torch.device:
        """设置设备"""
        if device_str == 'cpu':
            return torch.device('cpu')
        
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device_str}')
            torch.cuda.set_device(device)
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            print("CUDA not available, using CPU")
            device = torch.device('cpu')
        
        return device
    
    def _load_model(self):
        """加载YOLOv8模型"""
        from ultralytics import YOLO
        
        print(f"Loading model from: {self.weights}")
        
        if not os.path.exists(self.weights):
            print(f"Warning: Weights not found: {self.weights}")
            print("Using pretrained YOLOv8m instead")
            model = YOLO('yolov8m.pt')
        else:
            model = YOLO(self.weights)
        
        # 设置模式
        if self.mode == 'high_precision':
            model.conf = 0.15
            model.iou = 0.45
        elif self.mode == 'balanced':
            model.conf = 0.25
            model.iou = 0.45
        else:  # high_speed
            model.conf = 0.35
            model.iou = 0.50
        
        return model
    
    def _load_sam(self) -> Optional[SAMRefinement]:
        """加载SAM模型"""
        sam_cfg = self.config.get('model', {}).get('sam_refinement', {})
        
        if not sam_cfg.get('enabled', False) or self.mode == 'high_speed':
            return None
        
        try:
            print("Loading SAM for boundary refinement...")
            sam = SAMRefinement(
                model_type=sam_cfg.get('model_type', 'vit_b'),
                checkpoint=sam_cfg.get('checkpoint'),
                level_set_cfg=sam_cfg.get('level_set')
            )
            return sam
        except Exception as e:
            print(f"Failed to load SAM: {e}")
            return None
    
    def enable_agent_system(self):
        """启用Agent系统"""
        agent_cfg = self.config.get('agent', {})
        
        self.agent_system = PipelineAgentSystem(
            config=self.config.to_dict(),
            test_function=self._benchmark
        )
        self.agent_system.start()
        
        print("Agent system enabled")
    
    def _benchmark(self, config: dict) -> dict:
        """基准测试函数"""
        # 模拟测试
        return {
            'latency_ms': {'mean': 30 + np.random.randn() * 5},
            'throughput': {'mean': 100 + np.random.randn() * 10},
            'memory_mb': {'mean': 4000},
            'gpu_utilization': {'mean': 80}
        }
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """预处理"""
        start = time.time()
        
        # 调整尺寸
        img_size = self.config.get('dataset', {}).get('img_size', 640)
        h, w = img.shape[:2]
        
        # 保持长宽比缩放
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 填充到正方形
        pad_h = img_size - new_h
        pad_w = img_size - new_w
        
        img_padded = cv2.copyMakeBorder(
            img_resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        self.preprocess_times.append(time.time() - start)
        
        return img_padded
    
    def postprocess(
        self,
        results,
        orig_shape: Tuple[int, int]
    ) -> List[dict]:
        """后处理"""
        start = time.time()
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf),
                    'class_id': int(box.cls),
                    'class_name': result.names[int(box.cls)]
                }
                detections.append(detection)
        
        self.postprocess_times.append(time.time() - start)
        
        return detections
    
    def refine_with_sam(
        self,
        img: np.ndarray,
        detections: List[dict]
    ) -> List[dict]:
        """使用SAM精修边界"""
        if self.sam is None or not detections:
            return detections
        
        boxes = [d['bbox'] for d in detections]
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        
        try:
            refined = self.sam.forward(img, boxes_tensor)
            
            for i, ref in enumerate(refined):
                if i < len(detections):
                    # 确保 bbox_refined 是可序列化的格式
                    bbox_refined = ref['box']
                    if hasattr(bbox_refined, 'tolist'):
                        bbox_refined = bbox_refined.tolist()
                    detections[i]['bbox_refined'] = bbox_refined
                    
                    # mask 转为列表格式（或设为None如果不需要保存）
                    mask = ref['mask']
                    if mask is not None:
                        if isinstance(mask, np.ndarray):
                            # 将mask转为紧凑的RLE编码或简单列表
                            # 这里转为列表，如果需要更高效可以用其他格式
                            mask = mask.tolist()
                        detections[i]['mask'] = mask
        except Exception as e:
            print(f"SAM refinement failed: {e}")
        
        return detections
    
    def predict(
        self,
        source: Union[str, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45
    ) -> List[dict]:
        """
        单图推理
        
        Args:
            source: 图像路径或numpy数组
            conf: 置信度阈值
            iou: NMS IoU阈值
            
        Returns:
            检测结果列表
        """
        # 记录Stage时间
        if self.agent_system:
            self.agent_system.collector.start_stage_timer('preprocess')
        
        # 读取图像
        if isinstance(source, str):
            img = cv2.imread(source)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = source
        
        orig_shape = img.shape[:2]
        
        if self.agent_system:
            self.agent_system.collector.end_stage_timer('preprocess')
            self.agent_system.collector.start_stage_timer('inference')
        
        # 推理
        start = time.time()
        results = self.model.predict(
            img,
            conf=conf,
            iou=iou,
            verbose=False
        )
        inference_time = time.time() - start
        self.inference_times.append(inference_time)
        
        if self.agent_system:
            self.agent_system.collector.end_stage_timer('inference')
            self.agent_system.collector.start_stage_timer('postprocess')
        
        # 后处理
        detections = self.postprocess(results, orig_shape)
        
        # SAM精修
        if self.sam and self.mode != 'high_speed':
            detections = self.refine_with_sam(img, detections)
        
        if self.agent_system:
            self.agent_system.collector.end_stage_timer('postprocess')
        
        return detections
    
    def predict_batch(
        self,
        sources: List[Union[str, np.ndarray]],
        conf: float = 0.25,
        iou: float = 0.45
    ) -> List[List[dict]]:
        """批量推理"""
        results = []
        for source in sources:
            result = self.predict(source, conf, iou)
            results.append(result)
        return results
    
    def predict_video(
        self,
        source: str,
        output_path: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        show: bool = False
    ):
        """视频推理"""
        # 打开视频
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Failed to open video: {source}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建输出视频
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print(f"Processing video: {source}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 推理
            detections = self.predict(frame_rgb, conf, iou)
            
            # 绘制结果
            frame_result = self._draw_detections(frame, detections)
            
            # 保存
            if writer:
                writer.write(frame_result)
            
            # 显示
            if show:
                cv2.imshow('DefectGuard', frame_result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 打印进度
            if frame_count % 30 == 0:
                avg_time = np.mean(self.inference_times[-30:]) if self.inference_times else 0
                current_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"Frame {frame_count}, FPS: {current_fps:.1f}, Detections: {len(detections)}")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessed {frame_count} frames")
        self.print_stats()
    
    def _draw_detections(
        self,
        img: np.ndarray,
        detections: List[dict],
        color: tuple = None,
        label_prefix: str = "",
        show_refined: bool = False
    ) -> np.ndarray:
        """绘制检测结果
        
        Args:
            img: 输入图像
            detections: 检测结果列表
            color: 指定颜色，None则使用类别颜色
            label_prefix: 标签前缀（如"YOLO"、"SAM"）
            show_refined: 是否显示精修后的框
        """
        img_draw = img.copy()
        
        # 颜色映射
        colors = [
            (0, 255, 0),    # missing - 绿色
            (255, 0, 0),    # mousebite - 蓝色
            (0, 0, 255),    # open_circuit - 红色
            (255, 255, 0),  # short - 青色
            (255, 0, 255),  # spur - 紫色
        ]
        
        for det in detections:
            # 选择要显示的框
            if show_refined and 'bbox_refined' in det:
                bbox = det['bbox_refined']
                box_type = "Refined"
            else:
                bbox = det['bbox']
                box_type = "Original"
            
            conf = det['confidence']
            cls_id = det['class_id']
            cls_name = det.get('class_name', f'class_{cls_id}')
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 选择颜色
            if color is not None:
                box_color = color
            else:
                box_color = colors[cls_id % len(colors)]
            
            # 画框
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), box_color, 2)
            
            # 画标签
            if label_prefix:
                label = f"{label_prefix}: {cls_name} {conf:.2f}"
            else:
                label = f"{cls_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                img_draw,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                box_color, -1
            )
            cv2.putText(
                img_draw, label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        return img_draw
    
    def _create_comparison_image(
        self,
        img_original: np.ndarray,
        img_yolo: np.ndarray,
        img_refined: np.ndarray,
        title: str = ""
    ) -> np.ndarray:
        """创建三栏对比图
        
        Args:
            img_original: 原图
            img_yolo: YOLO检测结果
            img_refined: SAM精修结果
            title: 图像标题
            
        Returns:
            三栏拼接的对比图
        """
        h, w = img_original.shape[:2]
        
        # 创建标题栏
        title_height = 40
        header_color = (50, 50, 50)
        
        # 为每张图添加标题
        def add_header(img, header_text):
            header = np.full((title_height, w, 3), header_color, dtype=np.uint8)
            text_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (title_height + text_size[1]) // 2
            cv2.putText(header, header_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return np.vstack([header, img])
        
        img1 = add_header(img_original, "Original Image")
        img2 = add_header(img_yolo, "YOLO Detection (Green)")
        img3 = add_header(img_refined, "SAM Refined (Blue)")
        
        # 水平拼接
        comparison = np.hstack([img1, img2, img3])
        
        # 添加总标题
        if title:
            main_header = np.full((30, comparison.shape[1], 3), (30, 30, 30), dtype=np.uint8)
            text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = (comparison.shape[1] - text_size[0]) // 2
            cv2.putText(main_header, title, (text_x, 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            comparison = np.vstack([main_header, comparison])
        
        # 添加图例
        legend_height = 30
        legend = np.full((legend_height, comparison.shape[1], 3), (40, 40, 40), dtype=np.uint8)
        
        # YOLO图例（绿色）
        cv2.rectangle(legend, (10, 5), (30, 25), (0, 255, 0), -1)
        cv2.putText(legend, "YOLO Original", (35, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # SAM图例（蓝色）
        cv2.rectangle(legend, (200, 5), (220, 25), (255, 0, 0), -1)
        cv2.putText(legend, "SAM Refined", (225, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        comparison = np.vstack([comparison, legend])
        
        return comparison
    
    def print_stats(self):
        """打印性能统计"""
        if not self.inference_times:
            return
        
        print("\n" + "="*50)
        print("Performance Statistics")
        print("="*50)
        print(f"Mode: {self.mode}")
        print(f"Total inferences: {len(self.inference_times)}")
        print(f"Avg inference time: {np.mean(self.inference_times)*1000:.1f}ms")
        print(f"Avg FPS: {1.0/np.mean(self.inference_times):.1f}")
        
        if self.preprocess_times:
            print(f"Avg preprocess time: {np.mean(self.preprocess_times)*1000:.1f}ms")
        if self.postprocess_times:
            print(f"Avg postprocess time: {np.mean(self.postprocess_times)*1000:.1f}ms")
    
    def switch_mode(self, mode: str):
        """切换性能模式"""
        self.mode = mode
        
        # 更新模型参数
        if mode == 'high_precision':
            self.model.conf = 0.15
            self.model.iou = 0.45
        elif mode == 'balanced':
            self.model.conf = 0.25
            self.model.iou = 0.45
        else:  # high_speed
            self.model.conf = 0.35
            self.model.iou = 0.50
        
        # 更新SAM
        if mode == 'high_speed':
            self.sam = None
        elif self.sam is None:
            self.sam = self._load_sam()
        
        print(f"Switched to {mode} mode")


def main():
    args = parse_args()
    
    # 创建推理器
    inferencer = DefectGuardInference(
        weights=args.weights,
        config_path=args.config,
        device=args.device,
        mode=args.mode
    )
    
    # 启用Agent系统
    if args.agent:
        inferencer.enable_agent_system()
    
    # 判断输入类型
    source = args.source
    
    if source.isdigit():
        # 摄像头
        inferencer.predict_video(
            source,
            output_path='./outputs/camera_result.mp4' if args.save else None,
            conf=args.conf,
            iou=args.iou,
            show=args.show
        )
    elif source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # 视频文件
        output = './outputs/video_result.mp4' if args.save else None
        inferencer.predict_video(
            source,
            output_path=output,
            conf=args.conf,
            iou=args.iou,
            show=args.show
        )
    elif os.path.isdir(source):
        # 目录
        image_files = [
            f for f in os.listdir(source)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        
        print(f"Found {len(image_files)} images in {source}")
        
        all_results = []
        for img_file in image_files:
            img_path = os.path.join(source, img_file)
            detections = inferencer.predict(img_path, args.conf, args.iou)
            all_results.append({
                'image': img_file,
                'detections': detections
            })
            print(f"{img_file}: {len(detections)} detections")
        
        # 保存结果
        if args.save:
            # 创建输出目录
            os.makedirs('./outputs/visualizations', exist_ok=True)
            
            # 保存JSON结果
            with open('./outputs/batch_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to ./outputs/batch_results.json")
            
            # 保存可视化图像（三栏对比图）
            for img_file in image_files:
                img_path = os.path.join(source, img_file)
                
                # 读取原图
                img_orig = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                
                # 只运行YOLO检测（不精修）
                img_yolo = img_rgb.copy()
                results_yolo = inferencer.model.predict(img_path, conf=args.conf, iou=args.iou)
                detections_yolo = []
                for result in results_yolo:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        detections_yolo.append({
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'confidence': float(box.conf),
                            'class_id': int(box.cls),
                            'class_name': result.names[int(box.cls)]
                        })
                img_yolo_drawn = inferencer._draw_detections(img_yolo, detections_yolo, color=(0, 255, 0), label_prefix="YOLO")
                
                # 运行完整流程（YOLO + SAM精修）
                detections_refined = inferencer.predict(img_path, args.conf, args.iou)
                img_refined_drawn = inferencer._draw_detections(img_rgb.copy(), detections_refined, color=(255, 0, 0), label_prefix="SAM", show_refined=True)
                
                # 创建三栏对比图
                comparison = inferencer._create_comparison_image(img_rgb, img_yolo_drawn, img_refined_drawn, img_file)
                
                output_path = f'./outputs/visualizations/{img_file}'
                cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            
            print(f"Visualizations saved to ./outputs/visualizations/")
    else:
        # 单张图像
        detections = inferencer.predict(source, args.conf, args.iou)
        print(f"\nDetections: {len(detections)}")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.3f} at {det['bbox']}")
        
        # 保存可视化结果
        if args.save:
            img = cv2.imread(source)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result_img = inferencer._draw_detections(img_rgb, detections)
            result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            
            output_path = './outputs/single_result.jpg'
            cv2.imwrite(output_path, result_bgr)
            print(f"Result saved to {output_path}")
    
    # 打印统计
    inferencer.print_stats()
    
    # 关闭Agent系统
    if inferencer.agent_system:
        inferencer.agent_system.stop()


if __name__ == '__main__':
    main()
