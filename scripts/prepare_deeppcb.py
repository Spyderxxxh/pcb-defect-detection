"""
DeepPCB 数据集准备脚本
下载并转换为YOLO格式

DeepPCB数据集结构:
- PCBData/
  - groupXXXXX/
    - XXXXX/          # 缺陷图像 (00041000_temp.jpg, 00041000_test.jpg)
    - XXXXX_not/      # 模板图像
  - trainval.txt      # 训练集列表
  - test.txt          # 测试集列表
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import random

# DeepPCB 数据集URL
DEEPPCB_URL = "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"


def download_file(url, dest_path):
    """下载文件并显示进度"""
    print(f"Downloading from {url}...")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)
    
    print(f"Downloaded to {dest_path}")


def parse_deeppcb_split_line(line, pcbdata_dir):
    """
    解析trainval.txt/test.txt中的一行
    
    格式: image_path annotation_path
    示例: group20085/20085/20085000.jpg group20085/20085_not/20085000.txt
    
    注意: trainval.txt中的图像路径后缀是.jpg，但实际文件可能是_temp.jpg或_test.jpg
    """
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    
    img_rel_path = parts[0]  # e.g., "group20085/20085/20085000.jpg"
    ann_rel_path = parts[1]  # e.g., "group20085/20085_not/20085000.txt"
    
    # 构建标注完整路径
    ann_path = pcbdata_dir / ann_rel_path
    
    # 处理图像路径: 将 .jpg 替换为实际存在的文件 (_temp.jpg 或 _test.jpg)
    img_path_base = pcbdata_dir / img_rel_path.replace('.jpg', '')
    img_dir = img_path_base.parent
    img_stem = img_path_base.name  # e.g., "20085000"
    
    # 尝试 _test.jpg 或 _temp.jpg
    img_path_test = img_dir / f"{img_stem}_test.jpg"
    img_path_temp = img_dir / f"{img_stem}_temp.jpg"
    
    if img_path_test.exists():
        img_path = img_path_test
    elif img_path_temp.exists():
        img_path = img_path_temp
    else:
        # 回退到原始路径（可能不存在）
        img_path = pcbdata_dir / img_rel_path
    
    return {
        'img_path': img_path,
        'ann_path': ann_path,
        'img_rel_path': img_rel_path
    }


def parse_deeppcb_annotation(ann_line):
    """
    解析单个标注行
    
    格式: x1 y1 x2 y2 class_id
    注意: class_id从1开始，需要转换为0-based
    示例: 409 394 435 422 3
    """
    parts = ann_line.strip().split()
    if len(parts) < 5:
        return None
    
    x1, y1, x2, y2 = map(int, parts[:4])
    class_id = int(parts[4]) - 1  # 转换为0-based
    
    return {
        'bbox': [x1, y1, x2, y2],
        'class_id': class_id
    }


def prepare_deeppcb_dataset(data_root="./data/deeppcb"):
    """
    准备DeepPCB数据集
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_root / "deeppcb.zip"
    extract_path = data_root / "extracted"
    
    # 1. 下载数据集
    if not zip_path.exists():
        try:
            download_file(DEEPPCB_URL, zip_path)
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please manually download from: https://github.com/tangsanli5201/DeepPCB")
            print(f"And place the zip file at: {zip_path}")
            return False
    
    # 2. 解压
    if not extract_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete")
    
    # 3. 定位数据集目录
    deeppcb_dir = extract_path / "DeepPCB-master"
    pcbdata_dir = deeppcb_dir / "PCBData"
    
    if not pcbdata_dir.exists():
        print(f"Error: PCBData directory not found at {pcbdata_dir}")
        return False
    
    print(f"\nUsing dataset directory: {pcbdata_dir}")
    
    # 4. 读取trainval.txt和test.txt
    trainval_file = pcbdata_dir / "trainval.txt"
    test_file = pcbdata_dir / "test.txt"
    
    if not trainval_file.exists() or not test_file.exists():
        print(f"Error: Split files not found!")
        print(f"  trainval.txt exists: {trainval_file.exists()}")
        print(f"  test.txt exists: {test_file.exists()}")
        return False
    
    # 5. 解析数据集划分文件
    print("\nParsing dataset splits...")
    
    # 训练集
    train_image_list = []
    with open(trainval_file, 'r') as f:
        for line in f:
            entry = parse_deeppcb_split_line(line, pcbdata_dir)
            if entry:
                train_image_list.append(entry)
    
    # 测试集
    test_image_list = []
    with open(test_file, 'r') as f:
        for line in f:
            entry = parse_deeppcb_split_line(line, pcbdata_dir)
            if entry:
                test_image_list.append(entry)
    
    print(f"Train/Val images: {len(train_image_list)}")
    print(f"Test images: {len(test_image_list)}")

    # 6. 从训练集划分出验证集 (20%)
    random.seed(42)
    random.shuffle(train_image_list)
    split_idx = int(len(train_image_list) * 0.8)
    train_list = train_image_list[:split_idx]
    val_list = train_image_list[split_idx:]

    print(f"Split: {len(train_list)} train, {len(val_list)} val, {len(test_image_list)} test")
    
    # 7. 创建YOLO格式目录结构
    yolo_root = data_root / "yolo_format"
    for split in ['train', 'val', 'test']:
        (yolo_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # DeepPCB缺陷类别映射 (根据论文)
    class_names = {
        0: "open",        # 开路
        1: "short",       # 短路
        2: "mousebite",   # 缺口
        3: "spur",        # 毛刺
        4: "copper",      # 余铜
        5: "pinhole",     # 针孔
    }
    
    # 8. 处理每个数据集划分
    splits_data = {
        'train': train_list,
        'val': val_list,
        'test': test_image_list
    }

    stats = {split: {'images': 0, 'annotations': 0} for split in splits_data}

    for split, image_entries in splits_data.items():
        print(f"\nProcessing {split} set ({len(image_entries)} images)...")

        for entry in tqdm(image_entries, desc=f"{split}"):
            img_path = entry['img_path']
            ann_path = entry['ann_path']

            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue

            # 读取图像获取尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            h, w = img.shape[:2]

            # 生成输出文件名
            img_stem = img_path.stem  # e.g., "20085000"
            group_name = img_path.parent.parent.name  # e.g., "group20085"
            out_img_name = f"{group_name}_{img_stem}.jpg"

            # 复制图像
            dst_img_path = yolo_root / 'images' / split / out_img_name
            shutil.copy2(img_path, dst_img_path)
            stats[split]['images'] += 1

            # 读取并转换标注
            yolo_labels = []
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    for line in f:
                        ann = parse_deeppcb_annotation(line)
                        if ann:
                            x1, y1, x2, y2 = ann['bbox']
                            cls_id = ann['class_id']

                            # 转换为YOLO格式 (x_center, y_center, width, height) - 归一化
                            x_center = ((x1 + x2) / 2) / w
                            y_center = ((y1 + y2) / 2) / h
                            box_w = (x2 - x1) / w
                            box_h = (y2 - y1) / h

                            # 限制在[0,1]范围内
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            box_w = max(0, min(1, box_w))
                            box_h = max(0, min(1, box_h))

                            yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
                            stats[split]['annotations'] += 1

            # 写入YOLO格式标注
            dst_label_path = yolo_root / 'labels' / split / f"{group_name}_{img_stem}.txt"
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
    
    # 9. 创建dataset.yaml配置文件
    yaml_content = f"""# DeepPCB Dataset
path: {yolo_root.absolute()}  # dataset root dir
train: images/train  # train images
val: images/val  # val images
test: images/test  # test images

# Classes
nc: 6  # number of classes
names:
  0: open
  1: short
  2: mousebite
  3: spur
  4: copper
  5: pinhole
"""
    
    yaml_path = data_root / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # 10. 打印统计信息
    print("\n" + "="*50)
    print("Dataset Preparation Complete!")
    print("="*50)
    for split, stat in stats.items():
        print(f"{split}: {stat['images']} images, {stat['annotations']} annotations")
    print(f"\nDataset YAML: {yaml_path}")
    print(f"YOLO format data: {yolo_root}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare DeepPCB Dataset')
    parser.add_argument('--data-root', type=str, default='./data/deeppcb',
                        help='Root directory for dataset')
    args = parser.parse_args()
    
    success = prepare_deeppcb_dataset(args.data_root)
    sys.exit(0 if success else 1)
