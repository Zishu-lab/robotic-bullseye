#!/usr/bin/env python3
"""
PDF 图集转换脚本
将 targets_40_images_booklet.pdf 转换为图片并按类别组织
"""

import fitz  # PyMuPDF
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    image_format: str = "png"
) -> list[Path]:
    """
    将 PDF 的每一页转换为图片

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
        dpi: 分辨率（默认 300）
        image_format: 图片格式（png/jpg）

    Returns:
        生成的图片路径列表
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开 PDF
    doc = fitz.open(pdf_path)
    logger.info(f"PDF 文件: {pdf_path}")
    logger.info(f"总页数: {len(doc)}")

    image_paths = []

    # 转换每一页
    for page_num in range(len(doc)):
        page = doc[page_num]

        # 设置缩放因子（基于 DPI）
        zoom = dpi / 72  # 72 是 PDF 的默认 DPI
        mat = fitz.Matrix(zoom, zoom)

        # 渲染页面为图片
        pix = page.get_pixmap(matrix=mat)

        # 生成文件名
        image_name = f"page_{page_num + 1:03d}.{image_format}"
        image_path = output_dir / image_name

        # 保存图片
        pix.save(image_path)
        image_paths.append(image_path)

        logger.info(f"已转换: 页面 {page_num + 1}/{len(doc)} -> {image_name}")

        # 获取页面尺寸信息
        logger.info(f"  尺寸: {pix.width} x {pix.height} 像素")

    doc.close()

    return image_paths


def main():
    """主函数"""
    # 配置路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    pdf_path = project_root / "targets_40_images_booklet.pdf"
    output_dir = project_root / "data" / "raw"

    logger.info("=" * 60)
    logger.info("开始 PDF 转换")
    logger.info("=" * 60)

    # 执行转换
    image_paths = convert_pdf_to_images(
        pdf_path=pdf_path,
        output_dir=output_dir,
        dpi=300,  # 高分辨率
        image_format="png"
    )

    logger.info("=" * 60)
    logger.info(f"转换完成! 共生成 {len(image_paths)} 张图片")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
