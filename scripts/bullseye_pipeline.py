#!/usr/bin/env python3
"""
靶心检测+识别完整流程
1. YOLO检测靶心位置
2. 使用本地训练的CIFAR-100分类模型识别内容
"""

import cv2
import torch
import torchvision
import torchvision.transforms as transforms
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
        classifier_model_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化流程

        Args:
            yolo_model_path: YOLO模型路径
            classifier_model_path: 分类器模型路径（可选，默认使用本地训练的模型）
            device: 运行设备
        """
        self.device = device

        # 默认CIFAR-100类别名称（会被模型中的类别覆盖）
        self.cifar100_classes = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

        # 加载YOLO检测模型
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)
        logger.info(f"加载YOLO模型: {yolo_model_path}")


        # 加载CIFAR-100分类模型（优先使用本地训练的模型）
        if classifier_model_path is None:
            project_root = Path(__file__).parent.parent
            classifier_model_path = project_root / "models/classifier/final_classifier.pt"
        self.classifier = self._load_cifar100_classifier(classifier_model_path)
        logger.info(f"加载CIFAR-100分类模型: {classifier_model_path}")

        # 图像预处理（使用ImageNet归一化参数，因为基于预训练模型）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet标准输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_cifar100_classifier(self, model_path: str):
        """
        加载CIFAR-100分类模型

        Args:
            model_path: 本地训练的模型路径
        """
        # 使用ResNet-50架构（与训练脚本一致）
        model = torchvision.models.resnet50(weights=None)

        # 加载checkpoint获取类别信息
        checkpoint = torch.load(model_path, map_location='cpu')
        num_classes = len(checkpoint.get('classes', ['unknown']))
        self.cifar100_classes = checkpoint.get('classes', self.cifar100_classes)

        # 修改最后一层（与train_bullseye_classifier.py一致）
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
        logger.info(f"分类器类别数: {num_classes}")

        # 加载本地训练的权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"加载模型权重: epoch {checkpoint.get('epoch', 'N/A')}, acc {checkpoint.get('acc', 0):.2f}%")
        else:
            model.load_state_dict(checkpoint)
            logger.info("加载模型权重完成")

        model = model.to(self.device)
        model.eval()

        logger.info("CIFAR-100分类器加载成功")
        return model

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
        识别靶心内容

        Args:
            cropped_image: 裁剪的靶心图像 (numpy array)

        Returns:
            (类别名称, 置信度, top5预测)
        """
        # 转换为PIL图像
        if isinstance(cropped_image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        else:
            image = cropped_image

        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # 获取top5预测
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        top5_prob = top5_prob.cpu().numpy()[0]
        top5_indices = top5_indices.cpu().numpy()[0]

        top5_predictions = [
            (self.cifar100_classes[idx], prob)
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
    classifier_path = project_root / "models/classifier/final_classifier.pt"

    if not yolo_model_path.exists():
        logger.error(f"YOLO模型不存在: {yolo_model_path}")
        logger.info("请先运行训练: python scripts/train_yolo.py")
        return

    if not classifier_path.exists():
        logger.warning(f"分类器模型不存在: {classifier_path}")
        logger.info("将使用ImageNet预训练模型（效果可能较差）")

    # 创建流程
    pipeline = BullseyePipeline(
        str(yolo_model_path),
        classifier_model_path=str(classifier_path) if classifier_path.exists() else None
    )

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
