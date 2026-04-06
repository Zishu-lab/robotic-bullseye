#!/usr/bin/env python3
"""
YOLO 数据集准备脚本
自动生成标注 + 数据增强
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import Tuple, List
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLODatasetPreparer:
    """YOLO 数据集准备器"""

    def __init__(
        self,
        image_dir: Path,
        output_dir: Path,
        class_name: str = "bullseye",
        class_id: int = 0,
    ):
        """
        初始化数据集准备器

        Args:
            image_dir: 输入图片目录
            output_dir: 输出目录
            class_name: 类别名称
            class_id: 类别ID
        """
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.class_name = class_name
        self.class_id = class_id

        # 创建输出目录结构
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def generate_auto_annotations(self) -> List[dict]:
        """
        自动生成标注

        从 bullseye_positions.json 读取正确的靶心坐标
        每张原图生成一个样本，包含该图中所有靶心的位置

        Returns:
            标注信息列表
        """
        annotations = []

        # 读取靶心位置配置文件
        positions_file = self.image_dir / "bullseye_positions.json"
        if not positions_file.exists():
            logger.error(f"找不到靶心位置配置文件: {positions_file}")
            return annotations

        with open(positions_file, 'r') as f:
            positions_data = json.load(f)

        # 构建文件名到位置的映射
        positions_map = {item['file']: item for item in positions_data}
        logger.info(f"加载了 {len(positions_map)} 个文件的靶心位置配置")

        # 获取所有图片
        image_files = list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        logger.info(f"找到 {len(image_files)} 张图片")

        for image_file in image_files:
            filename = image_file.name

            # 跳过非原始图片文件
            if filename in ['correct_cropped_upper.png', 'correct_cropped_lower.png',
                           'position_comparison.png', 'debug_bullseye_boxes.png',
                           'debug_cropped_from_json_upper.png', 'debug_json_pos1.png',
                           'debug_json_pos2.png']:
                continue

            # 检查是否有配置
            if filename not in positions_map:
                logger.warning(f"未找到 {filename} 的靶心位置配置，跳过")
                continue

            # 读取图片获取尺寸
            img = cv2.imread(str(image_file))
            if img is None:
                logger.warning(f"无法读取图片: {image_file}")
                continue

            h, w = img.shape[:2]
            file_config = positions_map[filename]

            # 收集该图片中所有靶心的位置
            boxes = []
            for bullseye in file_config['bullseyes']:
                bbox = bullseye['bbox_pixel']  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox

                # 转换为 YOLO 格式 (归一化的 x_center, y_center, width, height)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                bbox_width = (x2 - x1) / w
                bbox_height = (y2 - y1) / h

                boxes.append({
                    "class_id": self.class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": bbox_width,
                    "height": bbox_height,
                })

            # 每张原图一个样本
            base_name = image_file.stem
            annotations.append({
                "source_file": image_file,
                "output_name": base_name,
                "boxes": boxes,  # 多个靶心
            })

        logger.info(f"生成 {len(annotations)} 个图片标注")
        return annotations

    def save_yolo_labels(self, annotations: List[dict]):
        """
        保存 YOLO 格式的标注文件

        Args:
            annotations: 标注信息列表
        """
        for ann in annotations:
            label_file = self.labels_dir / f"{ann['output_name']}.txt"

            with open(label_file, 'w') as f:
                for box in ann['boxes']:
                    f.write(f"{box['class_id']} {box['x_center']} {box['y_center']} {box['width']} {box['height']}\n")

        logger.info(f"已保存 {len(annotations)} 个标注文件")

    def save_dataset_yaml(self):
        """
        保存数据集配置文件 (dataset.yaml)
        """
        yaml_content = f"""# YOLO 数据集配置
path: {self.output_dir.absolute()}
train: images
val: images

# 类别
names:
  {self.class_id}: {self.class_name}
"""

        yaml_file = self.output_dir / "dataset.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        logger.info(f"已保存数据集配置: {yaml_file}")

    def apply_data_augmentation(
        self,
        annotations: List[dict],
        augment_count: int = 3,
    ) -> List[dict]:
        """
        应用数据增强

        Args:
            annotations: 原始标注列表
            augment_count: 每张图片增强的数量

        Returns:
            包含增强数据的标注列表
        """
        augmented = []

        for ann in annotations:
            # 保留原始
            augmented.append(ann)

            # 读取图片
            img = cv2.imread(str(ann['source_file']))
            if img is None:
                continue

            h, w = img.shape[:2]

            # 生成增强数据
            for i in range(augment_count):
                aug_img, aug_boxes = self._augment_image(img, ann['boxes'], i)

                # 保存增强图片
                aug_name = f"{ann['output_name']}_aug_{i+1}"
                aug_file = self.images_dir / f"{aug_name}.png"
                cv2.imwrite(str(aug_file), aug_img)

                # 保存增强标注
                label_file = self.labels_dir / f"{aug_name}.txt"
                with open(label_file, 'w') as f:
                    for box in aug_boxes:
                        f.write(f"{box['class_id']} {box['x_center']} {box['y_center']} {box['width']} {box['height']}\n")

                augmented.append({
                    'source_file': aug_file,
                    'output_name': aug_name,
                    'boxes': aug_boxes,
                    'is_augmented': True,
                })

        logger.info(f"数据增强完成：原始 {len(annotations)} → 增强 {len(augmented)}")
        return augmented

    def _augment_image(
        self,
        image: np.ndarray,
        boxes: List[dict],
        aug_type: int
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        对单张图片进行数据增强

        Args:
            image: 输入图像
            boxes: 边界框列表（归一化坐标）
            aug_type: 增强类型 (0-3)

        Returns:
            (增强后的图像, 增强后的边界框列表)
        """
        h, w = image.shape[:2]
        aug_img = image.copy()
        aug_boxes = [box.copy() for box in boxes]

        # 根据类型应用不同的增强
        if aug_type == 0:
            # 降低分辨率（模拟远距离）
            scale = 0.5
            small = cv2.resize(aug_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            aug_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        elif aug_type == 1:
            # 添加高斯噪声
            noise = np.random.normal(0, 15, aug_img.shape).astype(np.uint8)
            aug_img = cv2.add(aug_img, noise)

        elif aug_type == 2:
            # 降低对比度（模拟光照不足）
            aug_img = cv2.convertScaleAbs(aug_img, alpha=0.7, beta=20)

        elif aug_type == 3:
            # 轻微模糊（模拟空气扰动）
            aug_img = cv2.GaussianBlur(aug_img, (5, 5), 1)

        # 边界框保持不变（因为我们没有改变图片的宽高比或位置）
        return aug_img, aug_boxes

    def copy_images_for_detection(self, annotations: List[dict]):
        """
        复制原始图片用于检测训练（保留原始尺寸和标签坐标）

        Args:
            annotations: 标注信息列表
        """
        for ann in annotations:
            # 复制原图到输出目录
            src = ann['source_file']
            dst = self.images_dir / f"{ann['output_name']}.png"
            if not dst.exists():
                shutil.copy(src, dst)

        logger.info(f"已复制 {len(annotations)} 张原始图片")

    def create_cropped_images_for_recognition(self, annotations: List[dict], output_subdir: str = "cropped"):
        """
        创建裁剪后的图片用于内容识别（第二步）

        Args:
            annotations: 标注信息列表
            output_subdir: 裁剪图片输出子目录
        """
        cropped_dir = self.output_dir / output_subdir
        cropped_dir.mkdir(parents=True, exist_ok=True)

        cropped_count = 0
        for ann in annotations:
            # 读取原图
            img = cv2.imread(str(ann['source_file']))
            if img is None:
                continue

            h, w = img.shape[:2]

            # 为每个靶心创建裁剪图片
            for i, box in enumerate(ann['boxes']):
                # 转换为像素坐标
                x_center = int(box['x_center'] * w)
                y_center = int(box['y_center'] * h)
                box_w = int(box['width'] * w)
                box_h = int(box['height'] * h)

                # 计算裁剪区域
                x1 = max(0, x_center - box_w // 2)
                y1 = max(0, y_center - box_h // 2)
                x2 = min(w, x_center + box_w // 2)
                y2 = min(h, y_center + box_h // 2)

                # 裁剪
                cropped = img[y1:y2, x1:x2]

                # 保存裁剪后的图片
                output_file = cropped_dir / f"{ann['output_name']}_{i+1:02d}.png"
                cv2.imwrite(str(output_file), cropped)
                cropped_count += 1

        logger.info(f"已创建 {cropped_count} 张裁剪图片到 {cropped_dir}")

    def generate_classification_labels(self, annotations: List[dict], output_subdir: str = "cropped"):
        """
        为裁剪图片生成分类标签（从 bullseye_classes.json 读取类别）

        Args:
            annotations: 标注信息列表
            output_subdir: 裁剪图片目录
        """
        # 读取类别标注
        classes_file = self.image_dir / "bullseye_classes.json"
        if not classes_file.exists():
            logger.warning(f"找不到类别标注文件: {classes_file}")
            return

        with open(classes_file, 'r') as f:
            classes_data = json.load(f)

        class_to_idx = classes_data['class_to_idx']
        class_annotations = classes_data['annotations']

        cropped_dir = self.output_dir / output_subdir
        labels_file = self.output_dir / "classification_labels.json"

        # 生成分类标签
        classification_labels = {}
        for ann in annotations:
            source_name = ann['source_file'].stem
            for i in range(len(ann['boxes'])):
                # 查找对应的类别
                key = f"{source_name}_target{i+1}"
                if key in class_annotations:
                    class_name = class_annotations[key]
                    class_idx = class_to_idx.get(class_name, -1)

                    output_name = f"{ann['output_name']}_{i+1:02d}"
                    classification_labels[output_name] = {
                        'class_name': class_name,
                        'class_idx': class_idx
                    }

        # 保存标签文件
        with open(labels_file, 'w') as f:
            json.dump({
                'classes': classes_data['classes'],
                'class_to_idx': class_to_idx,
                'labels': classification_labels
            }, f, indent=2)

        logger.info(f"已生成分类标签文件: {labels_file}")
        logger.info(f"共有 {len(classification_labels)} 个样本，{len(classes_data['classes'])} 个类别")


def main():
    """主函数"""
    # 配置路径
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "yolo_dataset"

    logger.info("=" * 60)
    logger.info("开始准备 YOLO 检测数据集")
    logger.info("=" * 60)

    # 创建准备器
    preparer = YOLODatasetPreparer(
        image_dir=input_dir,
        output_dir=output_dir,
        class_name="bullseye",
        class_id=0,
    )

    # 生成自动标注
    logger.info("步骤 1: 生成自动标注...")
    annotations = preparer.generate_auto_annotations()

    # 复制原始图片用于检测训练（保留原始尺寸和坐标）
    logger.info("步骤 2: 复制原始图片...")
    preparer.copy_images_for_detection(annotations)

    # 保存 YOLO 格式标注（使用原始坐标）
    logger.info("步骤 3: 保存 YOLO 标注...")
    preparer.save_yolo_labels(annotations)

    # 应用数据增强
    logger.info("步骤 4: 应用数据增强...")
    annotations = preparer.apply_data_augmentation(annotations, augment_count=3)

    # 创建裁剪图片用于内容识别（第二步）
    logger.info("步骤 5: 创建裁剪图片用于内容识别...")
    preparer.create_cropped_images_for_recognition(annotations)

    # 生成分类标签
    logger.info("步骤 6: 生成分类标签...")
    preparer.generate_classification_labels(annotations)

    # 保存数据集配置
    logger.info("步骤 7: 保存数据集配置...")
    preparer.save_dataset_yaml()

    logger.info("=" * 60)
    logger.info(f"数据集准备完成!")
    logger.info(f"检测训练样本: {len(annotations)} (images/ 目录)")
    logger.info(f"内容识别样本: cropped/ 目录")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
