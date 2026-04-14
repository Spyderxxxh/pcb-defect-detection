"""
数据验证工具
支持YOLO格式数据集验证、类别分布统计、标签一致性检查
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import json


class DataValidator:
    """
    数据集验证器
    
    功能：
    - 验证YOLO格式数据集完整性
    - 检查图像-标签对应关系
    - 统计类别分布
    - 检测异常标签（越界、负值、重复等）
    """
    
    def __init__(self, data_root: str, class_names: Optional[List[str]] = None):
        """
        Args:
            data_root: 数据集根目录
            class_names: 类别名称列表，可选
        """
        self.data_root = data_root
        self.class_names = class_names or []
        self.issues = []
    
    def validate_dataset(self, split: str = 'train') -> Dict:
        """
        完整验证数据集
        
        Returns:
            {
                'valid': bool,
                'issues': List[str],
                'stats': Dict,
                'class_distribution': Dict[int, int]
            }
        """
        self.issues = []
        
        # 基础路径检查
        img_dir = os.path.join(self.data_root, 'images', split)
        label_dir = os.path.join(self.data_root, 'labels', split)
        
        if not os.path.exists(img_dir):
            self.issues.append(f"图像目录不存在: {img_dir}")
            return self._build_result(split)
        
        if not os.path.exists(label_dir):
            self.issues.append(f"标签目录不存在: {label_dir}")
            return self._build_result(split)
        
        # 获取文件列表
        img_files = self._get_image_files(img_dir)
        label_files = set(os.listdir(label_dir))
        
        # 检查图像-标签对应
        matched, unmatched_imgs, unmatched_labels = self._check_pairs(
            img_files, label_files, img_dir, label_dir
        )
        
        if unmatched_imgs:
            self.issues.append(f"缺少标签的图像: {len(unmatched_imgs)}个")
        if unmatched_labels:
            self.issues.append(f"缺少图像的标签: {len(unmatched_labels)}个")
        
        # 验证每个标签文件
        total_boxes = 0
        class_distribution = defaultdict(int)
        invalid_boxes = 0
        
        for img_file, label_file in matched:
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, label_file)
            
            # 读取图像尺寸
            img = cv2.imread(img_path)
            if img is None:
                self.issues.append(f"无法读取图像: {img_file}")
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 验证标签
            boxes, labels = self._validate_label_file(
                label_path, img_file, img_w, img_h
            )
            
            total_boxes += len(boxes)
            for label in labels:
                class_distribution[label] += 1
            
            # 检查异常
            invalid_boxes += self._check_anomalies(
                boxes, labels, img_w, img_h, label_file
            )
        
        stats = {
            'total_images': len(img_files),
            'matched_pairs': len(matched),
            'unmatched_images': len(unmatched_imgs),
            'unmatched_labels': len(unmatched_labels),
            'total_boxes': total_boxes,
            'invalid_boxes': invalid_boxes,
            'avg_boxes_per_image': total_boxes / len(matched) if matched else 0
        }
        
        return {
            'valid': len(self.issues) == 0,
            'issues': self.issues,
            'stats': stats,
            'class_distribution': dict(class_distribution)
        }
    
    def _get_image_files(self, img_dir: str) -> List[str]:
        """获取图像文件列表"""
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = []
        for f in os.listdir(img_dir):
            if os.path.splitext(f.lower())[1] in valid_exts:
                files.append(f)
        return sorted(files)
    
    def _check_pairs(
        self,
        img_files: List[str],
        label_files: Set[str],
        img_dir: str,
        label_dir: str
    ) -> Tuple[List[Tuple], List[str], List[str]]:
        """检查图像-标签对应关系"""
        matched = []
        unmatched_imgs = []
        unmatched_labels = []
        
        for img_file in img_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            
            if label_file in label_files:
                matched.append((img_file, label_file))
            else:
                unmatched_imgs.append(img_file)
        
        # 检查多余的标签
        img_bases = {os.path.splitext(f)[0] for f in img_files}
        for label_file in label_files:
            if not label_file.endswith('.txt'):
                continue
            base_name = os.path.splitext(label_file)[0]
            if base_name not in img_bases:
                unmatched_labels.append(label_file)
        
        return matched, unmatched_imgs, unmatched_labels
    
    def _validate_label_file(
        self,
        label_path: str,
        img_file: str,
        img_w: int,
        img_h: int
    ) -> Tuple[List[List[float]], List[int]]:
        """验证单个标签文件"""
        boxes = []
        labels = []
        
        if not os.path.exists(label_path):
            return boxes, labels
        
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    self.issues.append(
                        f"{img_file}: 第{line_num}行格式错误，期望5个值，实际{len(parts)}个"
                    )
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    boxes.append([x_c, y_c, w, h])
                    labels.append(class_id)
                    
                except ValueError as e:
                    self.issues.append(f"{img_file}: 第{line_num}行数值解析错误: {e}")
        
        return boxes, labels
    
    def _check_anomalies(
        self,
        boxes: List[List[float]],
        labels: List[int],
        img_w: int,
        img_h: int,
        label_file: str
    ) -> int:
        """检查异常边界框"""
        invalid_count = 0
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x_c, y_c, w, h = box
            
            # 检查类别ID
            if label < 0:
                self.issues.append(f"{label_file}: 第{i+1}个框类别ID为负: {label}")
                invalid_count += 1
            
            if self.class_names and label >= len(self.class_names):
                self.issues.append(
                    f"{label_file}: 第{i+1}个框类别ID越界: {label} >= {len(self.class_names)}"
                )
                invalid_count += 1
            
            # 检查坐标范围 (YOLO格式应该在0-1之间)
            if not (0 <= x_c <= 1 and 0 <= y_c <= 1):
                self.issues.append(
                    f"{label_file}: 第{i+1}个框中心点越界: ({x_c:.3f}, {y_c:.3f})"
                )
                invalid_count += 1
            
            if w <= 0 or h <= 0:
                self.issues.append(
                    f"{label_file}: 第{i+1}个框宽高无效: w={w:.3f}, h={h:.3f}"
                )
                invalid_count += 1
            
            if w > 1 or h > 1:
                self.issues.append(
                    f"{label_file}: 第{i+1}个框宽高超过图像: w={w:.3f}, h={h:.3f}"
                )
                invalid_count += 1
            
            # 检查框是否太小
            pixel_w = w * img_w
            pixel_h = h * img_h
            if pixel_w < 2 or pixel_h < 2:
                self.issues.append(
                    f"{label_file}: 第{i+1}个框太小: {pixel_w:.1f}x{pixel_h:.1f}像素"
                )
        
        return invalid_count
    
    def _build_result(self, split: str) -> Dict:
        """构建验证结果"""
        return {
            'valid': len(self.issues) == 0,
            'issues': self.issues,
            'stats': {
                'total_images': 0,
                'matched_pairs': 0,
                'unmatched_images': 0,
                'unmatched_labels': 0,
                'total_boxes': 0,
                'invalid_boxes': 0,
                'avg_boxes_per_image': 0
            },
            'class_distribution': {}
        }
    
    def analyze_class_balance(self, split: str = 'train') -> Dict:
        """
        分析类别平衡性
        
        Returns:
            {
                'class_distribution': {class_id: count},
                'class_percentages': {class_id: percentage},
                'imbalance_ratio': float,  # 最大/最小
                'recommendations': List[str]
            }
        """
        result = self.validate_dataset(split)
        distribution = result['class_distribution']
        
        if not distribution:
            return {
                'class_distribution': {},
                'class_percentages': {},
                'imbalance_ratio': 1.0,
                'recommendations': ['数据集为空']
            }
        
        total = sum(distribution.values())
        percentages = {
            k: v / total * 100 for k, v in distribution.items()
        }
        
        # 计算不平衡比例
        max_count = max(distribution.values())
        min_count = min(distribution.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # 生成建议
        recommendations = []
        if imbalance_ratio > 10:
            recommendations.append(f"类别严重不平衡 (比例: {imbalance_ratio:.1f}:1)，建议使用过采样或数据增强")
        
        for class_id, count in distribution.items():
            if count < 10:
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                recommendations.append(f"类别 {class_name} 样本过少 ({count}个)，建议增加数据")
        
        return {
            'class_distribution': distribution,
            'class_percentages': percentages,
            'imbalance_ratio': imbalance_ratio,
            'recommendations': recommendations
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        生成验证报告
        
        Args:
            output_path: 报告输出路径，为None则返回字符串
        
        Returns:
            报告内容
        """
        splits = ['train', 'val', 'test']
        report_lines = ["=" * 60, "数据集验证报告", "=" * 60, ""]
        
        for split in splits:
            img_dir = os.path.join(self.data_root, 'images', split)
            if not os.path.exists(img_dir):
                continue
            
            report_lines.append(f"\n【{split.upper()} 集】")
            report_lines.append("-" * 40)
            
            result = self.validate_dataset(split)
            
            # 基础统计
            stats = result['stats']
            report_lines.append(f"图像总数: {stats['total_images']}")
            report_lines.append(f"匹配对数: {stats['matched_pairs']}")
            report_lines.append(f"边界框总数: {stats['total_boxes']}")
            report_lines.append(f"平均每图框数: {stats['avg_boxes_per_image']:.2f}")
            
            # 类别分布
            if result['class_distribution']:
                report_lines.append("\n类别分布:")
                for class_id, count in sorted(result['class_distribution'].items()):
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                    report_lines.append(f"  {class_name}: {count}")
            
            # 问题
            if result['issues']:
                report_lines.append("\n发现问题:")
                for issue in result['issues'][:10]:  # 最多显示10个
                    report_lines.append(f"  ! {issue}")
                if len(result['issues']) > 10:
                    report_lines.append(f"  ... 还有 {len(result['issues']) - 10} 个问题")
            else:
                report_lines.append("\n✓ 未发现问题")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


def validate_yolo_dataset(data_root: str, class_names: Optional[List[str]] = None) -> bool:
    """
    便捷函数：快速验证YOLO数据集
    
    Args:
        data_root: 数据集根目录
        class_names: 类别名称列表
    
    Returns:
        是否验证通过
    """
    validator = DataValidator(data_root, class_names)
    
    print(f"正在验证数据集: {data_root}")
    print("=" * 50)
    
    all_valid = True
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(data_root, 'images', split)
        if not os.path.exists(img_dir):
            continue
        
        print(f"\n【{split}】")
        result = validator.validate_dataset(split)
        
        stats = result['stats']
        print(f"  图像: {stats['total_images']}, 边界框: {stats['total_boxes']}")
        
        if result['issues']:
            print(f"  ⚠ 问题: {len(result['issues'])}个")
            for issue in result['issues'][:5]:
                print(f"    - {issue}")
            all_valid = False
        else:
            print(f"  ✓ 通过")
    
    return all_valid


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
        class_names = sys.argv[2].split(',') if len(sys.argv) > 2 else None
        
        validator = DataValidator(data_root, class_names)
        print(validator.generate_report())
    else:
        print("用法: python data_validator.py <data_root> [class_names]")
        print("示例: python data_validator.py ./data/pcb_defect missing,mousebite,open_circuit,short,spur")
