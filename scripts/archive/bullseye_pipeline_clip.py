#!/usr/bin/env python3
"""
靶心检测+识别完整流程
1. YOLO检测靶心位置
2. 使用CLIP零样本分类识别内容
"""

import cv2
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BullseyePipeline:
    """靶心检测+识别流程"""

    def __init__(
        self,
        yolo_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化流程

        Args:
            yolo_model_path: YOLO模型路径
            device: 运行设备
        """
        self.device = device

        # 加载YOLO检测模型
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)
        logger.info(f"加载YOLO模型: {yolo_model_path}")

        # 加载CLIP模型
        self._load_clip_model()
        logger.info("加载CLIP模型")

        # CIFAR-100类别名称
        self.cifar100_classes = [
            'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear',
            'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank',
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm'
        ]

        # 预计算文本特征
        self._precompute_text_features()

    def _load_clip_model(self):
        """加载CLIP模型"""
        try:
            import open_clip
        except ImportError:
            logger.info("安装open_clip...")
            import subprocess
            subprocess.run(['pip', 'install', 'open-clip-torch', '-q'])
            import open_clip

        # 加载CLIP模型
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def _precompute_text_features(self):
        """预计算CIFAR-100类别的文本特征"""
        # 创建文本提示
        text_prompts = [f"a photo of a {cls}" for cls in self.cifar100_classes]

        # 编码文本
        text_tokens = self.tokenizer(text_prompts)
        text_tokens = text_tokens.to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

    def detect_bullseyes(self, image_path: str, conf_threshold: float = 0.5):
        """
        检测图像中的靶心

        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值

        Returns:
            检测结果列表，每个包含 (x1, y1, x2, y2, confidence)
        """
        results = self.yolo_model(image_path, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

        return detections

    def classify_content(self, cropped_image):
        """
        使用CLIP识别靶心内容

        Args:
            cropped_image: 裁剪的靶心图像 (numpy array, BGR格式)

        Returns:
            (类别名称, 置信度, top5预测)
        """
        # 转换为PIL图像 (RGB)
        if isinstance(cropped_image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        else:
            image = cropped_image

        # 预处理
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # 编码图像
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算相似度
            similarities = (image_features @ self.text_features.T).softmax(dim=-1)

        # 获取top5预测
        top5_prob, top5_indices = torch.topk(similarities[0], 5)
        top5_prob = top5_prob.cpu().numpy()
        top5_indices = top5_indices.cpu().numpy()

        top5_predictions = [
            (self.cifar100_classes[idx].replace(' ', '_'), float(prob))
            for idx, prob in zip(top5_indices, top5_prob)
        ]

        # 最佳预测
        best_class = top5_predictions[0][0]
        best_conf = top5_predictions[0][1]

        return best_class, best_conf, top5_predictions

    def process_image(self, image_path: str, conf_threshold: float = 0.5, visualize: bool = True):
        """
        完整处理流程：检测靶心 -> 识别内容

        Args:
            image_path: 输入图像路径
            conf_threshold: 检测置信度阈值
            visualize: 是否生成可视化结果

        Returns:
            处理结果
        """
        logger.info(f"处理图像: {image_path}")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 检测靶心
        detections = self.detect_bullseyes(image_path, conf_threshold)
        logger.info(f"检测到 {len(detections)} 个靶心")

        results = []
        for i, (x1, y1, x2, y2, conf) in enumerate(detections):
            # 裁剪靶心
            cropped = image[y1:y2, x1:x2]

            # 识别内容
            class_name, class_conf, top5 = self.classify_content(cropped)

            result = {
                'target_id': i + 1,
                'bbox': (x1, y1, x2, y2),
                'detection_conf': conf,
                'class_name': class_name,
                'class_conf': class_conf,
                'top5_predictions': top5,
            }
            results.append(result)

            logger.info(f"  靶心 {i+1}: 位置=({x1},{y1},{x2},{y2}), 类别={class_name}, 置信度={class_conf:.2%}")

            # 可视化
            if visualize:
                # 绘制检测框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制标签
                label = f"{class_name} ({class_conf:.1%})"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 保存可视化结果
        if visualize and results:
            output_path = Path(image_path).stem + "_result.jpg"
            cv2.imwrite(output_path, image)
            logger.info(f"保存可视化结果: {output_path}")

        return results


def main():
    """测试流程"""
    import sys

    # 配置
    project_root = Path(__file__).parent.parent
    yolo_model_path = project_root / "runs/detect/runs/detect/bullseye_train/weights/best.pt"

    if not yolo_model_path.exists():
        logger.error(f"YOLO模型不存在: {yolo_model_path}")
        logger.info("请先运行训练: python scripts/train_yolo.py")
        return

    # 创建流程
    pipeline = BullseyePipeline(str(yolo_model_path))

    # 测试图像
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = str(project_root / "data/raw/page_001.png")

    if not Path(test_image).exists():
        logger.error(f"测试图像不存在: {test_image}")
        return

    # 处理图像
    results = pipeline.process_image(test_image)

    # 打印结果
    print("\n" + "=" * 60)
    print("检测结果:")
    print("=" * 60)
    for r in results:
        print(f"\n靶心 #{r['target_id']}:")
        print(f"  位置: {r['bbox']}")
        print(f"  检测置信度: {r['detection_conf']:.2%}")
        print(f"  识别类别: {r['class_name']}")
        print(f"  类别置信度: {r['class_conf']:.2%}")
        print(f"  Top5预测:")
        for cls, prob in r['top5_predictions']:
            print(f"    - {cls}: {prob:.2%}")


if __name__ == "__main__":
    main()
