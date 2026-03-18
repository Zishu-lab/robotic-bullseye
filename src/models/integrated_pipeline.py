#!/usr/bin/env python3
"""
集成推理管道
连接 YOLO 检测 + CIFAR-100 分类
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BullseyePipeline:
    """
    靶心检测与内容识别管道

    流程：
    1. YOLO 检测靶心位置
    2. 裁剪中心区域
    3. CIFAR-100 识别物品
    """

    def __init__(
        self,
        yolo_model_path: str | Path = "runs/detect/runs/detect/bullseye_optimized/weights/best.pt",
        cifar_model_path: str | Path = "experiments/runs/cifar100_resnet18/best.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold: float = 0.5,
        crop_size: Tuple[int, int] = (64, 64),
    ):
        """
        初始化管道

        Args:
            yolo_model_path: YOLO 模型路径
            cifar_model_path: CIFAR-100 模型路径
            device: 运行设备
            conf_threshold: YOLO 检测置信度阈值
            crop_size: 中心区域裁剪尺寸
        """
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.crop_size = crop_size

        # 加载 YOLO 模型
        logger.info("加载 YOLO 模型...")
        from ultralytics import YOLO
        self.yolo_model = YOLO(str(yolo_model_path))
        logger.info("✅ YOLO 模型加载完成")

        # 加载 CIFAR-100 模型
        logger.info("加载 CIFAR-100 模型...")
        checkpoint = torch.load(str(cifar_model_path), map_location=self.device)
        # 提取模型状态字典
        if 'model_state_dict' in checkpoint:
            # 加载ResNet模型结构
            import torchvision.models as models
            self.cifar_model = models.resnet18(weights=None)
            self.cifar_model.fc = torch.nn.Linear(self.cifar_model.fc.in_features, 100)
            self.cifar_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.cifar_model = checkpoint
        self.cifar_model = self.cifar_model.to(self.device)
        self.cifar_model.eval()
        logger.info("✅ CIFAR-100 模型加载完成")

        # CIFAR-100 类别名称
        self.cifar100_classes = self._get_cifar100_classes()

    def _get_cifar100_classes(self) -> List[str]:
        """获取 CIFAR-100 类别名称"""
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
            'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

    def detect_bullseye(self, image: np.ndarray) -> List[Dict]:
        """
        使用 YOLO 检测靶心

        Args:
            image: 输入图像

        Returns:
            检测结果列表
        """
        results = self.yolo_model(image, conf=self.conf_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())

                x1, y1, x2, y2 = box
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                })

        return detections

    def crop_center_region(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        size: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        裁剪中心区域

        Args:
            image: 输入图像
            center: 中心点 (x, y)
            size: 裁剪尺寸

        Returns:
            裁剪后的图像
        """
        if size is None:
            size = self.crop_size

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
            y_offset = (h - cropped.shape[0]) // 2
            x_offset = (w - cropped.shape[1]) // 2
            padded[y_offset:y_offset + cropped.shape[0],
                   x_offset:x_offset + cropped.shape[1]] = cropped
            cropped = padded

        # 调整到指定尺寸
        cropped = cv2.resize(cropped, size)

        return cropped

    def classify_content(self, image: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        使用 CIFAR-100 模型识别物品

        Args:
            image: 输入图像（64x64 或其他尺寸）
            top_k: 返回前 k 个预测

        Returns:
            预测结果列表
        """
        # 调整到 CIFAR-100 输入尺寸 (32x32)
        image_resized = cv2.resize(image, (32, 32))

        # 转换为 Tensor
        # CIFAR-100 的均值和标准差
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

        # 转换为Tensor: HWC -> CHW
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        # 标准化
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.cifar_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_classes = torch.topk(probabilities, top_k)

        # 转换为结果
        results = []
        for i in range(top_k):
            results.append({
                'class_id': int(top_classes[0][i]),
                'class_name': self.cifar100_classes[int(top_classes[0][i])],
                'probability': float(top_probs[0][i]),
            })

        return results

    def process(self, image: np.ndarray) -> Dict:
        """
        完整的处理流程

        Args:
            image: 输入图像

        Returns:
            处理结果
        """
        # Step 1: 检测靶心
        detections = self.detect_bullseye(image)

        if len(detections) == 0:
            return {
                'status': 'no_detection',
                'message': '未检测到靶心',
                'detections': [],
            }

        # 处理每个检测
        results = []
        for i, detection in enumerate(detections):
            # Step 2: 裁剪中心区域
            center_crop = self.crop_center_region(image, detection['center'])

            # Step 3: 识别物品
            classification = self.classify_content(center_crop)

            results.append({
                'detection_id': i,
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'center': detection['center'],
                'classification': classification,
            })

        return {
            'status': 'success',
            'detections': results,
            'count': len(results),
        }

    def process_single(self, image_path: str | Path, visualize: bool = True) -> Dict:
        """
        处理单张图片

        Args:
            image_path: 图片路径
            visualize: 是否可视化结果

        Returns:
            处理结果
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            return {
                'status': 'error',
                'message': f'无法读取图片: {image_path}',
            }

        # 处理
        result = self.process(image)

        # 可视化
        if visualize and result['status'] == 'success':
            vis_image = self._visualize_results(image, result)
            # 保存可视化结果
            output_path = image_path.parent / f"{image_path.stem}_result.png"
            cv2.imwrite(str(output_path), vis_image)
            result['visualization'] = str(output_path)

        return result

    def _visualize_results(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        可视化结果

        Args:
            image: 原始图像
            result: 处理结果

        Returns:
            可视化图像
        """
        vis = image.copy()

        for detection in result['detections']:
            # 绘制边界框
            bbox = detection['bbox']
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # 绘制中心点
            center = detection['center']
            cv2.circle(vis, center, 5, (0, 0, 255), -1)

            # 绘制分类结果
            classification = detection['classification'][0]
            label = f"{classification['class_name']}: {classification['probability']:.2f}"
            cv2.putText(vis, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis


def main():
    """主函数：测试集成管道"""
    import matplotlib
    matplotlib.use('Agg')

    logging.basicConfig(level=logging.INFO)

    # 配置路径
    project_root = Path(__file__).parent.parent.parent
    yolo_model_path = project_root / "runs/detect/runs/detect/bullseye_optimized/weights/best.pt"
    cifar_model_path = project_root / "experiments/runs/cifar100_resnet18/best.pt"
    test_image = project_root / "data/raw/page_001.png"

    logger.info("=" * 60)
    logger.info("集成管道测试")
    logger.info("=" * 60)

    # 检查模型是否存在
    if not yolo_model_path.exists():
        logger.error(f"YOLO 模型不存在: {yolo_model_path}")
        logger.info("请先完成 YOLO 训练")
        return

    if not cifar_model_path.exists():
        logger.error(f"CIFAR-100 模型不存在: {cifar_model_path}")
        logger.info("请先完成 CIFAR-100 训练")
        return

    # 创建管道
    pipeline = BullseyePipeline(
        yolo_model_path=yolo_model_path,
        cifar_model_path=cifar_model_path,
        conf_threshold=0.25,  # 降低置信度阈值
    )

    # 处理测试图片
    logger.info(f"处理测试图片: {test_image}")
    result = pipeline.process_single(test_image, visualize=True)

    # 打印结果
    logger.info("=" * 60)
    logger.info("处理结果:")
    logger.info(f"状态: {result['status']}")
    if result['status'] == 'success':
        logger.info(f"检测到 {result['count']} 个靶心")
        for i, detection in enumerate(result['detections']):
            logger.info(f"\n靶心 {i+1}:")
            logger.info(f"  位置: {detection['bbox']}")
            logger.info(f"  置信度: {detection['confidence']:.2f}")
            logger.info(f"  分类结果: {detection['classification'][0]['class_name']}")
            logger.info(f"  概率: {detection['classification'][0]['probability']:.2f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
