

#!/usr/bin/env python3
"""
靶心圆环特征提取器
提取黑色外圈和灰色圆环特征，增强小目标检测
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BullseyeRingExtractor:
    """靶心圆环特征提取器"""

    def __init__(
        self,
        black_threshold=(0, 50),      # 黑色阈值范围 (低, 高)
        gray_threshold=(50, 150),      # 灰色阈值范围 (低, 高)
        kernel_size=(5, 5),            # 形态学核大小
        blur_kernel=(5, 5),            # 高斯模糊核大小
    ):
        """
        初始化提取器

        Args:
            black_threshold: 黑色区域的阈值范围
            gray_threshold: 灰色区域的阈值范围
            kernel_size: 形态学操作的核大小
            blur_kernel: 高斯模糊的核大小
        """
        self.black_threshold = black_threshold
        self.gray_threshold = gray_threshold
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        self.blur_kernel = blur_kernel

    def extract_black_region(self, gray_image: np.ndarray) -> np.ndarray:
        """
        提取黑色外圈区域

        Args:
            gray_image: 灰度图像

        Returns:
            黑色区域的二值图像
        """
        # 阈值分割提取黑色区域
        low, high = self.black_threshold
        _, black_mask = cv2.threshold(gray_image, high, 255, cv2.THRESH_BINARY_INV)

        # 形态学闭运算：填充小孔洞
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, self.kernel)

        return black_mask

    def extract_gray_region(self, gray_image: np.ndarray) -> np.ndarray:
        """
        提取灰色圆环区域

        Args:
            gray_image: 灰度图像

        Returns:
            灰色区域的二值图像
        """
        # 阈值分割提取灰色区域
        low, high = self.gray_threshold

        # 双阈值分割
        _, binary_low = cv2.threshold(gray_image, low, 255, cv2.THRESH_BINARY)
        _, binary_high = cv2.threshold(gray_image, high, 255, cv2.THRESH_BINARY)

        # 灰色区域 = 低阈值以上 - 高阈值以上
        gray_mask = cv2.subtract(binary_low, binary_high)

        # 形态学闭运算：填充小孔洞
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, self.kernel)

        return gray_mask

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        增强对比度

        Args:
            image: 输入图像

        Returns:
            对比度增强后的图像
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        if len(image.shape) == 3:
            # 彩色图像，转换到 LAB 空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图像
            return clahe.apply(image)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        降噪处理

        Args:
            image: 输入图像

        Returns:
            降噪后的图像
        """
        return cv2.GaussianBlur(image, self.blur_kernel, 0)

    def process(
        self,
        image: np.ndarray,
        return_visual: bool = False
    ) -> dict[str, np.ndarray]:
        """
        完整的预处理流程

        Args:
            image: 输入图像 (BGR 或灰度)
            return_visual: 是否返回可视化结果

        Returns:
            包含处理结果的字典:
            - 'original': 原始图像
            - 'gray': 灰度图像
            - 'black_region': 黑色区域
            - 'gray_region': 灰色区域
            - 'enhanced': 增强后的图像
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 降噪
        gray_denoised = self.denoise(gray)

        # 对比度增强
        enhanced = self.enhance_contrast(gray_denoised)

        # 提取黑色区域
        black_region = self.extract_black_region(enhanced)

        # 提取灰色区域
        gray_region = self.extract_gray_region(enhanced)

        results = {
            'original': image,
            'gray': gray,
            'enhanced': enhanced,
            'black_region': black_region,
            'gray_region': gray_region,
        }

        if return_visual:
            results['visual'] = self._create_visualization(results)

        return results

    def _create_visualization(self, results: dict[str, np.ndarray]) -> np.ndarray:
        """
        创建可视化结果

        Args:
            results: 处理结果字典

        Returns:
            可视化图像
        """
        images = [
            results['gray'],
            results['enhanced'],
            results['black_region'],
            results['gray_region'],
        ]

        titles = ['Grayscale', 'Enhanced', 'Black Region', 'Gray Region']

        # 创建 2x2 网格
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.patch.set_facecolor('white')

        for idx, (img, title) in enumerate(zip(images, titles)):
            ax = axes[idx // 2, idx % 2]

            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()

        # 转换回 OpenCV 格式
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        vis = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        vis = vis.reshape(canvas.get_width_height()[::-1] + (4,))
        vis = vis[:, :, :3]  # 移除 alpha 通道
        plt.close(fig)

        return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def process_single_image(
    image_path: str | Path,
    output_dir: str | Path,
    extractor: BullseyeRingExtractor = None
) -> dict[str, np.ndarray]:
    """
    处理单张图片

    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        extractor: 特征提取器实例

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

    # 创建提取器
    if extractor is None:
        extractor = BullseyeRingExtractor()

    # 处理图片
    results = extractor.process(image, return_visual=True)

    # 保存结果
    base_name = image_path.stem

    # 保存处理后的图像
    cv2.imwrite(str(output_dir / f"{base_name}_enhanced.png"), results['enhanced'])
    cv2.imwrite(str(output_dir / f"{base_name}_black.png"), results['black_region'])
    cv2.imwrite(str(output_dir / f"{base_name}_gray.png"), results['gray_region'])
    cv2.imwrite(str(output_dir / f"{base_name}_visual.png"), results['visual'])

    logger.info(f"已处理: {image_path.name}")

    return results


def main():
    """主函数"""

    # 配置路径
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"

    logger.info("=" * 60)
    logger.info("开始靶心圆环特征提取")
    logger.info("=" * 60)

    # 创建提取器
    extractor = BullseyeRingExtractor(
        black_threshold=(0, 80),     # 可根据实际效果调整
        gray_threshold=(80, 180),    # 可根据实际效果调整
        kernel_size=(7, 7),          # 闭运算核大小
    )

    # 处理所有图片
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

    if not image_files:
        logger.warning(f"未找到图片文件: {input_dir}")
        return

    logger.info(f"找到 {len(image_files)} 张图片")

    for image_file in image_files:
        process_single_image(image_file, output_dir, extractor)

    logger.info("=" * 60)
    logger.info(f"处理完成! 结果保存在: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
