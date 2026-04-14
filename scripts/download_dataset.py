"""
数据集下载脚本
下载PCB缺陷检测数据集用于测试
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_file(url: str, output_path: str, desc: str = ""):
    """下载文件并显示进度"""
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\r{desc}: {percent:.1f}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    print()  # 换行


def download_pcb_defect_dataset(data_root: str = "./data/pcb_defect"):
    """
    下载PCB缺陷检测数据集
    
    使用DeepPCB数据集或类似的公开数据集
    """
    print("="*60)
    print("PCB Defect Detection Dataset Setup")
    print("="*60)
    
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    # 由于PCB数据集通常需要申请，这里创建一个示例结构
    # 实际使用时，请替换为真实的数据集下载链接
    
    print("\nNote: This is a template script.")
    print("Please download the PCB defect dataset manually from:")
    print("  - DeepPCB: http://www.imagecomputing.org/chenlab/resource/DeepPCB.html")
    print("  - Or use your own dataset")
    
    # 创建目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        (data_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (data_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 创建data.yaml
    yaml_content = """# PCB Defect Detection Dataset
path: ./data/pcb_defect  # dataset root dir
train: images/train  # train images
val: images/val  # val images
test: images/test  # test images (optional)

# Classes
names:
  0: missing
  1: mousebite
  2: open_circuit
  3: short
  4: spur
"""
    
    yaml_path = data_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated directory structure at: {data_root}")
    print(f"Created data.yaml at: {yaml_path}")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Place your images in:")
    print("   - data/pcb_defect/images/train/")
    print("   - data/pcb_defect/images/val/")
    print("   - data/pcb_defect/images/test/")
    print("\n2. Place your labels in:")
    print("   - data/pcb_defect/labels/train/")
    print("   - data/pcb_defect/labels/val/")
    print("   - data/pcb_defect/labels/test/")
    print("\n3. Label format (YOLO):")
    print("   <class_id> <x_center> <y_center> <width> <height>")
    print("\n4. Start training:")
    print("   python train.py --config configs/defectguard_yolov8.yaml")


def create_sample_data(data_root: str = "./data/sample"):
    """创建示例数据用于测试"""
    import numpy as np
    import cv2
    
    print("\nCreating sample data for testing...")
    
    data_root = Path(data_root)
    
    # 创建目录
    for split in ['train', 'val']:
        (data_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (data_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 生成示例图像
    np.random.seed(42)
    
    for split in ['train', 'val']:
        num_samples = 10 if split == 'train' else 5
        
        for i in range(num_samples):
            # 创建空白图像
            img = np.ones((640, 640, 3), dtype=np.uint8) * 200
            
            # 随机添加一些"缺陷"（黑色矩形）
            num_defects = np.random.randint(1, 5)
            labels = []
            
            for j in range(num_defects):
                # 随机位置
                x = np.random.randint(50, 550)
                y = np.random.randint(50, 550)
                w = np.random.randint(20, 100)
                h = np.random.randint(20, 100)
                
                # 绘制缺陷
                color = (0, 0, 0)  # 黑色
                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
                
                # 生成YOLO格式标签
                x_center = (x + w/2) / 640
                y_center = (y + h/2) / 640
                width = w / 640
                height = h / 640
                class_id = np.random.randint(0, 5)
                
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # 保存图像
            img_path = data_root / 'images' / split / f'{split}_{i:04d}.jpg'
            cv2.imwrite(str(img_path), img)
            
            # 保存标签
            label_path = data_root / 'labels' / split / f'{split}_{i:04d}.txt'
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
    
    # 创建data.yaml
    yaml_content = """# Sample Dataset
path: ./data/sample
train: images/train
val: images/val

names:
  0: missing
  1: mousebite
  2: open_circuit
  3: short
  4: spur
"""
    
    with open(data_root / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"Sample data created at: {data_root}")
    print("  Train samples: 10")
    print("  Val samples: 5")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download or create dataset')
    parser.add_argument('--dataset', type=str, default='pcb_defect',
                        choices=['pcb_defect', 'sample'],
                        help='Dataset to download/create')
    parser.add_argument('--output', type=str, default='./data',
                        help='Output directory')
    
    args = parser.parse_args()
    
    if args.dataset == 'pcb_defect':
        download_pcb_defect_dataset(os.path.join(args.output, 'pcb_defect'))
    elif args.dataset == 'sample':
        create_sample_data(os.path.join(args.output, 'sample'))


if __name__ == '__main__':
    main()
