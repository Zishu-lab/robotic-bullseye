#!/usr/bin/env python3
"""
预处理效果验证脚本
对比处理前后的圆环检测效果，评估预处理算法的性能
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.ring_extractor import BullseyeRingExtractor
from src.preprocessing.center_crop import ImageBasedCrop

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingValidator:
    """预处理效果验证器"""

    def __init__(
        self,
        hough_dp: float = 1.0,
        hough_min_dist: int = 100,
        hough_param1: int = 50,
        hough_param2: int = 30,
        hough_min_radius: int = 20,
        hough_max_radius: int = 200,
    ):
        """
        初始化验证器

        Args:
            hough_*: 霍夫圆变换参数
        """
        self.hough_params = {
            'dp': hough_dp,
            'min_dist': hough_min_dist,
            'param1': hough_param1,
            'param2': hough_param2,
            'min_radius': hough_min_radius,
            'max_radius': hough_max_radius,
        }

        # 初始化预处理器
        self.extractor = BullseyeRingExtractor(
            black_threshold=(0, 80),
            gray_threshold=(80, 180),
            kernel_size=(7, 7),
        )
        self.cropper = ImageBasedCrop(**self.hough_params)

    def detect_circles(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        使用霍夫圆变换检测圆

        Args:
            image: 输入图像（灰度或彩色）

        Returns:
            (圆数组, 统计信息)
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 霍夫圆变换
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_params['dp'],
            minDist=self.hough_params['min_dist'],
            param1=self.hough_params['param1'],
            param2=self.hough_params['param2'],
            minRadius=self.hough_params['min_radius'],
            maxRadius=self.hough_params['max_radius']
        )

        stats = {
            'image_shape': image.shape,
            'circles_detected': 0 if circles is None else len(circles[0]),
        }

        if circles is None:
            return np.array([]), stats

        circles = np.round(circles[0, :]).astype("int")
        stats['circles_detected'] = len(circles)

        if len(circles) > 0:
            stats['avg_radius'] = float(np.mean(circles[:, 2]))
            stats['std_radius'] = float(np.std(circles[:, 2]))
            stats['min_radius_detected'] = int(np.min(circles[:, 2]))
            stats['max_radius_detected'] = int(np.max(circles[:, 2]))

        return circles, stats

    def evaluate_single_image(self, image_path: Path) -> Dict:
        """
        评估单张图像的预处理效果

        Args:
            image_path: 图像路径

        Returns:
            评估结果字典
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return None

        result = {
            'image_name': image_path.name,
            'original': {},
            'enhanced': {},
            'black_region': {},
            'gray_region': {},
            'center_crop': {},
        }

        # 1. 在原始图像上检测圆
        circles_original, stats_original = self.detect_circles(image)
        result['original'] = stats_original

        # 2. 预处理
        processed = self.extractor.process(image, return_visual=False)

        # 3. 在增强后的图像上检测圆
        circles_enhanced, stats_enhanced = self.detect_circles(processed['enhanced'])
        result['enhanced'] = stats_enhanced

        # 4. 在黑色区域上检测圆
        circles_black, stats_black = self.detect_circles(processed['black_region'])
        result['black_region'] = stats_black

        # 5. 在灰色区域上检测圆
        circles_gray, stats_gray = self.detect_circles(processed['gray_region'])
        result['gray_region'] = stats_gray

        # 6. 中心裁剪
        crop_result = self.cropper.crop_center(image, crop_size=(64, 64), preprocess=True)
        result['center_crop'] = {
            'center_point': crop_result['center_point'],
            'confidence': crop_result['confidence'],
            'circles_detected': crop_result['circles_detected'],
        }

        # 7. 计算改进指标
        result['improvement'] = {
            'enhanced_vs_original': stats_enhanced['circles_detected'] - stats_original['circles_detected'],
            'black_vs_original': stats_black['circles_detected'] - stats_original['circles_detected'],
            'gray_vs_original': stats_gray['circles_detected'] - stats_original['circles_detected'],
        }

        # 保存可视化
        result['visualization'] = self._create_comparison_visual(
            image, processed, crop_result,
            circles_original, circles_enhanced, circles_black, circles_gray
        )

        return result

    def _create_comparison_visual(
        self,
        image: np.ndarray,
        processed: Dict,
        crop_result: Dict,
        circles_original: np.ndarray,
        circles_enhanced: np.ndarray,
        circles_black: np.ndarray,
        circles_gray: np.ndarray,
    ) -> np.ndarray:
        """创建对比可视化图像"""
        # 准备子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.patch.set_facecolor('white')

        images = [
            (image.copy(), circles_original, 'Original', self._draw_circles),
            (processed['enhanced'], circles_enhanced, 'Enhanced', self._draw_circles_binary),
            (processed['black_region'], circles_black, 'Black Region', self._draw_circles_binary),
            (processed['gray_region'], circles_gray, 'Gray Region', self._draw_circles_binary),
            (crop_result['cropped_image'], None, 'Center Crop', None),
        ]

        # 填充前5个子图
        for idx, (img, circles, title, draw_func) in enumerate(images):
            if idx >= 5:
                break
            ax = axes[idx // 3, idx % 3]

            if draw_func is not None:
                img = draw_func(img.copy(), circles)

            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            ax.set_title(f'{title}\nCircles: {len(circles)}' if circles is not None else title,
                        fontsize=10, fontweight='bold')
            ax.axis('off')

        # 第6个子图：裁剪可视化
        ax = axes[1, 2]
        vis = image.copy()
        center_x, center_y = crop_result['center_point']
        cv2.circle(vis, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.circle(vis, (center_x, center_y), 30, (0, 255, 0), 2)
        cv2.rectangle(vis, (center_x - 32, center_y - 32), (center_x + 32, center_y + 32), (255, 0, 0), 2)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Crop Detection\nConf: {crop_result["confidence"]:.2f}', fontsize=10, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        # 转换回 OpenCV 格式
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        vis = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        vis = vis.reshape(canvas.get_width_height()[::-1] + (4,))
        vis = vis[:, :, :3]
        plt.close(fig)

        return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    def _draw_circles(self, image: np.ndarray, circles: np.ndarray) -> np.ndarray:
        """在彩色图像上绘制检测到的圆"""
        if len(circles) == 0:
            return image

        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

        return image

    def _draw_circles_binary(self, image: np.ndarray, circles: np.ndarray) -> np.ndarray:
        """在二值图像上绘制检测到的圆"""
        if len(circles) == 0:
            return image

        # 转换为彩色以便绘制
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()

        for (x, y, r) in circles:
            cv2.circle(image_color, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image_color, (x, y), 2, (0, 0, 255), 3)

        return image_color

    def generate_report(self, results: List[Dict], output_dir: Path) -> str:
        """
        生成验证报告

        Args:
            results: 所有图像的评估结果列表
            output_dir: 输出目录

        Returns:
            报告文本
        """
        # 计算统计指标
        total_images = len(results)

        stats = {
            'total_images': total_images,
            'original_detection_rate': 0,
            'enhanced_detection_rate': 0,
            'black_region_detection_rate': 0,
            'gray_region_detection_rate': 0,
            'avg_circles_original': 0,
            'avg_circles_enhanced': 0,
            'avg_circles_black': 0,
            'avg_circles_gray': 0,
            'avg_crop_confidence': 0,
        }

        # 统计检测率（至少检测到一个圆）
        detected_original = sum(1 for r in results if r['original']['circles_detected'] > 0)
        detected_enhanced = sum(1 for r in results if r['enhanced']['circles_detected'] > 0)
        detected_black = sum(1 for r in results if r['black_region']['circles_detected'] > 0)
        detected_gray = sum(1 for r in results if r['gray_region']['circles_detected'] > 0)

        stats['original_detection_rate'] = detected_original / total_images * 100
        stats['enhanced_detection_rate'] = detected_enhanced / total_images * 100
        stats['black_region_detection_rate'] = detected_black / total_images * 100
        stats['gray_region_detection_rate'] = detected_gray / total_images * 100

        # 平均圆数量
        stats['avg_circles_original'] = np.mean([r['original']['circles_detected'] for r in results])
        stats['avg_circles_enhanced'] = np.mean([r['enhanced']['circles_detected'] for r in results])
        stats['avg_circles_black'] = np.mean([r['black_region']['circles_detected'] for r in results])
        stats['avg_circles_gray'] = np.mean([r['gray_region']['circles_detected'] for r in results])

        # 平均裁剪置信度
        stats['avg_crop_confidence'] = np.mean([r['center_crop']['confidence'] for r in results])

        # 生成报告文本
        report_lines = [
            "=" * 70,
            "预处理效果验证报告",
            "=" * 70,
            "",
            f"评估图像数量: {total_images}",
            "",
            "1. 圆环检测成功率",
            "-" * 70,
            f"  原始图像:           {stats['original_detection_rate']:.1f}%",
            f"  增强图像:           {stats['enhanced_detection_rate']:.1f}%",
            f"  黑色区域:           {stats['black_region_detection_rate']:.1f}%",
            f"  灰色区域:           {stats['gray_region_detection_rate']:.1f}%",
            "",
            "2. 平均检测到的圆数量",
            "-" * 70,
            f"  原始图像:           {stats['avg_circles_original']:.2f}",
            f"  增强图像:           {stats['avg_circles_enhanced']:.2f}",
            f"  黑色区域:           {stats['avg_circles_black']:.2f}",
            f"  灰色区域:           {stats['avg_circles_gray']:.2f}",
            "",
            "3. 中心裁剪性能",
            "-" * 70,
            f"  平均置信度:         {stats['avg_crop_confidence']:.2f}",
            "",
            "4. 改进效果",
            "-" * 70,
            f"  增强改进:           +{stats['avg_circles_enhanced'] - stats['avg_circles_original']:.2f} 圆/图",
            f"  黑色区域改进:       +{stats['avg_circles_black'] - stats['avg_circles_original']:.2f} 圆/图",
            f"  灰色区域改进:       +{stats['avg_circles_gray'] - stats['avg_circles_original']:.2f} 圆/图",
            "",
            "=" * 70,
        ]

        report = "\n".join(report_lines)

        # 保存报告
        report_file = output_dir / "validation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # 保存JSON统计数据
        stats_file = output_dir / "validation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"报告已保存到: {report_file}")
        logger.info(f"统计数据已保存到: {stats_file}")

        return report


def main():
    """主函数"""
    # 配置路径
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "validation_results"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("开始预处理效果验证")
    logger.info("=" * 70)

    # 创建验证器
    validator = PreprocessingValidator(
        hough_dp=1.0,
        hough_min_dist=100,
        hough_param1=50,
        hough_param2=30,
        hough_min_radius=20,
        hough_max_radius=200,
    )

    # 获取所有图像
    image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))

    if not image_files:
        logger.error(f"未找到图像文件: {input_dir}")
        return

    logger.info(f"找到 {len(image_files)} 张图像")

    # 评估所有图像
    results = []
    for i, image_file in enumerate(image_files, 1):
        logger.info(f"[{i}/{len(image_files)}] 评估: {image_file.name}")

        result = validator.evaluate_single_image(image_file)
        if result is not None:
            results.append(result)

            # 保存可视化结果
            vis_file = output_dir / f"{image_file.stem}_comparison.png"
            cv2.imwrite(str(vis_file), result['visualization'])

    # 生成报告
    report = validator.generate_report(results, output_dir)

    logger.info("=" * 70)
    logger.info("验证完成!")
    logger.info("=" * 70)
    logger.info("\n" + report)


if __name__ == "__main__":
    main()
