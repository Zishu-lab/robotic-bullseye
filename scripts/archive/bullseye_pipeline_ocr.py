#!/usr/bin/env python3
"""
靶心检测+识别完整流程（离线版）
1. YOLO检测靶心位置
2. 用OCR读取靶心旁边的Class标注
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BullseyePipelineOCR:
    """靶心检测+识别流程（使用OCR）"""

    def __init__(self, yolo_model_path: str):
        """初始化流程"""
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)
        logger.info(f"加载YOLO模型: {yolo_model_path}")
        self._load_ocr()

    def _load_ocr(self):
        """加载OCR引擎"""
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=True)
            self.pytesseract = None
            logger.info("加载EasyOCR成功")
        except Exception as e:
            logger.warning(f"EasyOCR加载失败: {e}")
            self.reader = None
            try:
                import pytesseract
                self.pytesseract = pytesseract
                logger.info("使用pytesseract")
            except ImportError:
                logger.error("OCR引擎未安装")
                self.pytesseract = None

    def detect_bullseyes(self, image_path: str, conf_threshold: float = 0.5):
        """检测图像中的靶心"""
        results = self.yolo_model(image_path, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return detections

    def read_class_label(self, image, bbox):
        """读取靶心旁边的Class标签"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Class标签在靶心右侧
        text_x1 = min(x2 + 20, w)
        text_x2 = min(x2 + 500, w)
        text_y1 = y1
        text_y2 = y2

        if text_x1 >= text_x2 or text_x2 > w:
            return "unknown"

        text_region = image[text_y1:text_y2, text_x1:text_x2]

        try:
            if self.reader:
                results = self.reader.readtext(text_region)
                if results:
                    full_text = ' '.join([r[1] for r in results])
                    match = re.search(r'Class[:\s]+(\w+)', full_text, re.IGNORECASE)
                    if match:
                        return match.group(1).lower()
                    for r in results:
                        text = r[1].strip()
                        if text and len(text) > 2:
                            if not re.match(r'^Target|^#|^\\d', text, re.IGNORECASE):
                                return text.lower().replace(' ', '_')
            elif self.pytesseract:
                text = self.pytesseract.image_to_string(text_region)
                match = re.search(r'Class[:\s]+(\w+)', text, re.IGNORECASE)
                if match:
                    return match.group(1).lower()
        except Exception as e:
            logger.warning(f"OCR识别失败: {e}")

        return "unknown"

    def process_image(self, image_path: str, conf_threshold: float = 0.5, visualize: bool = True):
        """完整处理流程"""
        logger.info(f"处理图像: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        detections = self.detect_bullseyes(image_path, conf_threshold)
        logger.info(f"检测到 {len(detections)} 个靶心")

        results = []
        for i, (x1, y1, x2, y2, conf) in enumerate(detections):
            class_name = self.read_class_label(image, (x1, y1, x2, y2))

            result = {
                'target_id': i + 1,
                'bbox': (x1, y1, x2, y2),
                'detection_conf': conf,
                'class_name': class_name,
            }
            results.append(result)
            logger.info(f"  靶心 {i+1}: 位置=({x1},{y1},{x2},{y2}), 类别={class_name}")

            if visualize:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"#{i+1}: {class_name}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if visualize and results:
            output_path = Path(image_path).stem + "_result.jpg"
            cv2.imwrite(output_path, image)
            logger.info(f"保存可视化结果: {output_path}")

        return results


def main():
    """测试流程"""
    import sys

    project_root = Path(__file__).parent.parent
    yolo_model_path = project_root / "runs/detect/runs/detect/bullseye_train/weights/best.pt"

    if not yolo_model_path.exists():
        logger.error(f"YOLO模型不存在: {yolo_model_path}")
        return

    pipeline = BullseyePipelineOCR(str(yolo_model_path))

    test_image = sys.argv[1] if len(sys.argv) > 1 else str(project_root / "data/raw/page_001.png")

    if not Path(test_image).exists():
        logger.error(f"测试图像不存在: {test_image}")
        return

    results = pipeline.process_image(test_image)

    print("\n" + "=" * 60)
    print("检测结果:")
    print("=" * 60)
    for r in results:
        print(f"\n靶心 #{r['target_id']}:")
        print(f"  位置: {r['bbox']}")
        print(f"  检测置信度: {r['detection_conf']:.2%}")
        print(f"  识别类别: {r['class_name']}")


if __name__ == "__main__":
    main()
