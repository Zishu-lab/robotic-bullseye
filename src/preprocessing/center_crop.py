

#!/usr/bin/env python3
"""
靶心中心区域裁剪模块
支持基于图像处理和基于 YOLO 检测两种裁剪策略
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)




class BaseCropStrategy(ABC):
    """裁剪策略基类"""

    @abstractmethod
    def crop_center(
        self,
        image: np.ndarray,
        crop_size: Tuple[int, int] = (64, 64),
        **kwargs
    ) -> Dict[str, any]:
        """
        裁剪中心区域

        Args:
            image: 输入图像
            crop_size: 裁剪尺寸 (width, height)
            **kwargs: 其他参数

        Returns:
            包含裁剪结果和元数据的字典:
            - 'cropped_image': 裁剪后的图像
            - 'center_point': 中心点坐标 (x, y)
            - 'confidence': 置信度/质量分数
            - 'method': 使用的方法名
        """
        pass


class ImageBasedCrop(BaseCropStrategy):
    """
    基于图像处理的中心裁剪策略

    原理：
    1. 使用霍夫圆变换检测圆
    2. 计算所有圆的中心点作为靶心中心
    3. 裁剪中心区域
    """

    def __init__(
        self,
        dp: float = 1.0,              # 霍夫圆变换累加器分辨率
        min_dist: int = 100,          # 圆心之间的最小距离
        param1: int = 50,             # Canny 边缘检测的高阈值
        param2: int = 30,             # 累加器阈值
        min_radius: int = 20,         # 最小圆半径
        max_radius: int = 200,        # 最大圆半径
    ):
        """
        初始化基于图像处理的裁剪策略

        Args:
            dp: 累加器分辨率（1.0 表示与图像相同分辨率）
            min_dist: 检测到的圆心之间的最小距离
            param1: Canny 边缘检测的高阈值
            param2: 累加器阈值（越小检测到的圆越多）
            min_radius: 最小圆半径
            max_radius: 最大圆半径
        """
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def _detect_circles(self, gray_image: np.ndarray) -> np.ndarray:
        """
        使用霍夫圆变换检测圆

        Args:
            gray_image: 灰度图像

        Returns:
            检测到的圆数组，每行格式为 (x, y, radius)
        """
        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is None:
            return np.array([])

        return np.round(circles[0, :]).astype("int")

    def _find_center_point(self, circles: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        从检测到的圆中找到中心点

        Args:
            circles: 检测到的圆数组
            image_shape: 图像尺寸 (height, width)

        Returns:
            中心点坐标 (x, y)
        """
        if len(circles) == 0:
            # 如果没有检测到圆，返回图像中心
            h, w = image_shape[:2]
            return (w // 2, h // 2)

        if len(circles) == 1:
            # 只有一个圆，直接使用
            return (circles[0][0], circles[0][1])

        # 多个圆，计算平均中心点（同心圆的情况）
        center_x = int(np.mean(circles[:, 0]))
        center_y = int(np.mean(circles[:, 1]))

        return (center_x, center_y)

    def crop_center(
        self,
        image: np.ndarray,
        crop_size: Tuple[int, int] = (64, 64),
        **kwargs
    ) -> Dict[str, any]:
        """
        裁剪中心区域

        Args:
            image: 输入图像
            crop_size: 裁剪尺寸 (width, height)
            **kwargs: 其他参数（可选 preprocess=True 来预处理图像）

        Returns:
            包含裁剪结果和元数据的字典
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 可选：预处理（增强圆环特征）
        preprocess = kwargs.get('preprocess', True)
        if preprocess:
            # 使用 CLAHE 增强对比度
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            # 高斯模糊降噪
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 检测圆
        circles = self._detect_circles(gray)

        # 找到中心点
        center_point = self._find_center_point(circles, image.shape)

        # 计算置信度（检测到的圆数量越多，置信度越高）
        confidence = min(len(circles) / 5.0, 1.0)  # 最多 5 个圆就认为置信度为 1

        # 裁剪中心区域
        cropped = self._crop_at_center(image, center_point, crop_size)

        return {
            'cropped_image': cropped,
            'center_point': center_point,
            'confidence': confidence,
            'method': 'image_based',
            'circles_detected': len(circles),
        }

    def _crop_at_center(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        size: Tuple[int, int]
    ) -> np.ndarray:
        """
        在指定中心点裁剪图像

        Args:
            image: 输入图像
            center: 中心点 (x, y)
            size: 裁剪尺寸 (width, height)

        Returns:
            裁剪后的图像
        """
        x, y = center
        w, h = size

        # 计算裁剪区域
        x1 = max(0, x - w // 2)
        y1 = max(0, y - h // 2)
        x2 = min(image.shape[1], x + w // 2)
        y2 = min(image.shape[0], y + h // 2)

        # 裁剪
        cropped = image[y1:y2, x1:x2]

        # 如果裁剪区域不够大，填充黑色
        if cropped.shape[0] != h or cropped.shape[1] != w:
            padded = np.zeros((h, w, 3) if len(image.shape) == 3 else (h, w), dtype=image.dtype)
            # 计算粘贴位置
            y_offset = (h - cropped.shape[0]) // 2
            x_offset = (w - cropped.shape[1]) // 2
            padded[y_offset:y_offset + cropped.shape[0],
                   x_offset:x_offset + cropped.shape[1]] = cropped
            cropped = padded

        # 调整到指定尺寸
        cropped = cv2.resize(cropped, size)

        return cropped


class YOLOBasedCrop(BaseCropStrategy):
    """
    基于 YOLO 检测的中心裁剪策略

    原理：
    1. YOLO 检测靶心，返回边界框
    2. 计算边界框的中心点
    3. 裁剪中心区域
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化基于 YOLO 的裁剪策略

        Args:
            model_path: YOLO 模型路径（如果为 None，需要后续加载）
        """
        self.model_path = model_path
        self.model = None

    def load_model(self, model_path: str = None):
        """
        加载 YOLO 模型

        Args:
            model_path: 模型路径（如果为 None，使用初始化时的路径）
        """
        if model_path is not None:
            self.model_path = model_path

        if self.model_path is None:
            raise ValueError("模型路径未指定")

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO 模型加载成功: {self.model_path}")
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")
            raise

    def crop_center(
        self,
        image: np.ndarray,
        crop_size: Tuple[int, int] = (64, 64),
        **kwargs
    ) -> Dict[str, any]:
        """
        裁剪中心区域

        Args:
            image: 输入图像
            crop_size: 裁剪尺寸 (width, height)
            **kwargs: 其他参数（可选 conf=0.25 设置置信度阈值）

        Returns:
            包含裁剪结果和元数据的字典
        """
        if self.model is None:
            raise RuntimeError("YOLO 模型未加载，请先调用 load_model()")

        # YOLO 检测
        conf_threshold = kwargs.get('conf', 0.25)
        results = self.model(image, conf=conf_threshold)

        # 获取检测结果
        if len(results[0].boxes) == 0:
            # 没有检测到靶心，使用图像中心
            h, w = image.shape[:2]
            center_point = (w // 2, h // 2)
            confidence = 0.0
        else:
            # 取置信度最高的检测框
            boxes = results[0].boxes
            best_idx = int(boxes.conf.argmax())
            box = boxes.xyxy[best_idx].cpu().numpy()

            # 计算中心点
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_point = (center_x, center_y)
            confidence = float(boxes.conf[best_idx])

        # 裁剪中心区域
        cropped = self._crop_at_center(image, center_point, crop_size)

        return {
            'cropped_image': cropped,
            'center_point': center_point,
            'confidence': confidence,
            'method': 'yolo_based',
            'detection_box': results[0].boxes[0].xyxy.cpu().numpy() if len(results[0].boxes) > 0 else None,
        }

    def _crop_at_center(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        size: Tuple[int, int]
    ) -> np.ndarray:
        """
        在指定中心点裁剪图像

        Args:
            image: 输入图像
            center: 中心点 (x, y)
            size: 裁剪尺寸 (width, height)

        Returns:
            裁剪后的图像
        """
        x, y = center
        w, h = size

        # 计算裁剪区域
        x1 = max(0, x - w // 2)
        y1 = max(0, y - h // 2)
        x2 = min(image.shape[1], x + w // 2)
        y2 = min(image.shape[0], y + h // 2)

        # 裁剪
        cropped = image[y1:y2, x1:x2]

        # 如果裁剪区域不够大，填充黑色
        if cropped.shape[0] != h or cropped.shape[1] != w:
            padded = np.zeros((h, w, 3) if len(image.shape) == 3 else (h, w), dtype=image.dtype)
            # 计算粘贴位置
            y_offset = (h - cropped.shape[0]) // 2
            x_offset = (w - cropped.shape[1]) // 2
            padded[y_offset:y_offset + cropped.shape[0],
                   x_offset:x_offset + cropped.shape[1]] = cropped
            cropped = padded

        # 调整到指定尺寸
        cropped = cv2.resize(cropped, size)

        return cropped


class CenterCropper:
    """
    中心裁剪器

    支持多种裁剪策略，可根据需要切换
    """

    def __init__(self, strategy: BaseCropStrategy = None):
        """
        初始化裁剪器

        Args:
            strategy: 裁剪策略（默认使用基于图像处理的策略）
        """
        if strategy is None:
            strategy = ImageBasedCrop()
        self.strategy = strategy

    def set_strategy(self, strategy: BaseCropStrategy):
        """
        切换裁剪策略

        Args:
            strategy: 新的裁剪策略
        """
        self.strategy = strategy

    def crop(
        self,
        image: np.ndarray,
        crop_size: Tuple[int, int] = (64, 64),
        **kwargs
    ) -> Dict[str, any]:
        """
        裁剪中心区域

        Args:
            image: 输入图像
            crop_size: 裁剪尺寸 (width, height)
            **kwargs: 传递给策略的其他参数

        Returns:
            包含裁剪结果和元数据的字典
        """
        return self.strategy.crop_center(image, crop_size, **kwargs)


def process_single_image(
    image_path: str | Path,
    output_dir: str | Path,
    cropper: CenterCropper = None,
    crop_size: Tuple[int, int] = (64, 64),
    save_visual: bool = True,
) -> Dict[str, any]:
    """
    处理单张图片

    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        cropper: 裁剪器实例
        crop_size: 裁剪尺寸
        save_visual: 是否保存可视化结果

    Returns:
        处理结果字典
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"无法读取图片: {image_path}")
        return None

    # 创建裁剪器
    if cropper is None:
        cropper = CenterCropper(ImageBasedCrop())

    # 裁剪中心区域
    result = cropper.crop(image, crop_size=crop_size)

    # 保存裁剪结果
    base_name = image_path.stem
    cv2.imwrite(str(output_dir / f"{base_name}_center.png"), result['cropped_image'])

    # 保存可视化结果
    if save_visual:
        visual = _create_visualization(image, result)
        cv2.imwrite(str(output_dir / f"{base_name}_crop_visual.png"), visual)

    logger.info(f"已处理: {image_path.name}, 中心点: {result['center_point']}, 置信度: {result['confidence']:.2f}")

    return result


def _create_visualization(image: np.ndarray, result: Dict[str, any]) -> np.ndarray:
    """
    创建可视化结果

    Args:
        image: 原始图像
        result: 裁剪结果字典

    Returns:
        可视化图像
    """
    vis = image.copy()

    # 绘制中心点
    center_x, center_y = result['center_point']
    cv2.circle(vis, (center_x, center_y), 5, (0, 255, 0), -1)
    cv2.circle(vis, (center_x, center_y), 30, (0, 255, 0), 2)

    # 绘制裁剪区域
    crop_size = result['cropped_image'].shape[:2][::-1]  # (width, height)
    x1 = max(0, center_x - crop_size[0] // 2)
    y1 = max(0, center_y - crop_size[1] // 2)
    x2 = min(vis.shape[1], center_x + crop_size[0] // 2)
    y2 = min(vis.shape[0], center_y + crop_size[1] // 2)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 添加文字
    text = f"Center: ({center_x}, {center_y}), Conf: {result['confidence']:.2f}"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return vis


def main():
    """主函数：测试中心裁剪功能"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 配置路径
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"

    logger.info("=" * 60)
    logger.info("开始中心区域裁剪")
    logger.info("=" * 60)

    # 创建裁剪器
    cropper = CenterCropper(ImageBasedCrop())

    # 处理所有图片
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

    if not image_files:
        logger.warning(f"未找到图片文件: {input_dir}")
        return

    logger.info(f"找到 {len(image_files)} 张图片")

    for image_file in image_files:
        process_single_image(image_file, output_dir, cropper, crop_size=(64, 64))

    logger.info("=" * 60)
    logger.info(f"处理完成! 结果保存在: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
