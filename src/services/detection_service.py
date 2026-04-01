#!/usr/bin/env python3
"""
检测服务模块
封装 YOLO 检测 + CIFAR-100 分类的完整逻辑
"""

import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from queue import Queue
from threading import Thread, Lock
import logging
import time
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """检测结果数据类"""
    bbox: List[int]
    confidence: float
    class_name: str
    probability: float


class ModelConfig:
    """模型配置管理"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/model_config.yaml")
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return self._default_config()

    def _default_config(self) -> dict:
        """默认配置"""
        return {
            'models': {
                'yolo': {
                    'path': 'runs/detect/runs/detect/bullseye_train/weights/best.pt',
                    'confidence_threshold': 0.3
                },
                'classifier': {
                    'path': 'experiments/runs/cifar100_optimized/best.pt',
                    'fallback_path': 'models/classifier/final_classifier.pt'
                }
            },
            'device': {
                'type': 'auto'
            },
            'camera': {
                'default_url': 'http://10.24.100.139:8080/video',
                'timeout': 10
            }
        }

    @property
    def yolo_path(self) -> Path:
        return Path(self._config['models']['yolo']['path'])

    @property
    def classifier_path(self) -> Path:
        path = Path(self._config['models']['classifier']['path'])
        if not path.exists():
            fallback = self._config['models']['classifier'].get('fallback_path')
            if fallback:
                path = Path(fallback)
        return path

    @property
    def confidence_threshold(self) -> float:
        return self._config['models']['yolo']['confidence_threshold']

    @property
    def device(self) -> str:
        device_type = self._config['device']['type']
        if device_type == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_type

    @property
    def camera_url(self) -> str:
        return self._config['camera']['default_url']


class DetectionService:
    """
    检测服务类
    封装模型加载、推理、视频流处理等逻辑
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[ModelConfig] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.config = config or ModelConfig()
        self.device = torch.device(self.config.device)

        # 模型
        self.yolo_model = None
        self.classifier = None
        self.classes: List[str] = []

        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])

        # 状态
        self._running = False
        self._camera_url = self.config.camera_url
        self._cap = None
        self._frame = None
        self._results: List[DetectionResult] = []
        self._fps = 0.0
        self._frame_lock = Lock()
        self._detection_thread: Optional[Thread] = None

        self._initialized = True
        logger.info(f"DetectionService 初始化完成, device={self.device}")

    def load_models(self) -> bool:
        """加载所有模型"""
        try:
            self._load_yolo()
            self._load_classifier()
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def _load_yolo(self):
        """加载 YOLO 模型"""
        from ultralytics import YOLO

        yolo_path = self.config.yolo_path
        if not yolo_path.exists():
            # 尝试备选路径
            alt_paths = [
                Path("runs/detect/runs/detect/bullseye_optimized/weights/best.pt"),
                Path("yolov8s.pt")
            ]
            for alt in alt_paths:
                if alt.exists():
                    yolo_path = alt
                    break

        logger.info(f"加载 YOLO 模型: {yolo_path}")
        self.yolo_model = YOLO(str(yolo_path))
        logger.info("✅ YOLO 模型加载完成")

    def _load_classifier(self):
        """加载分类器模型"""
        classifier_path = self.config.classifier_path
        logger.info(f"加载分类器: {classifier_path}")

        checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)
        self.classes = checkpoint.get('classes', [])

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        is_efficientnet = any('classifier' in k or 'features.0.0' in k for k in state_dict.keys())

        if is_efficientnet:
            logger.info("检测到 EfficientNet-B3 模型")
            model = torchvision.models.efficientnet_b3(weights=None)
            model.features[0][0] = torch.nn.Conv2d(
                3, model.features[0][0].out_channels, 3, 1, 1, bias=False
            )
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2, inplace=True),
                torch.nn.Linear(model.classifier[1].in_features, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.SiLU(inplace=True),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, 100)
            )
        else:
            logger.info("检测到 ResNet 模型")
            model = torchvision.models.resnet50(weights=None)
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(model.fc.in_features, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, len(self.classes) if self.classes else 100)
            )

        # 加载权重
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            new_state_dict = {
                k.replace('model.', '').replace('_orig_mod.', ''): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(new_state_dict, strict=False)

        model.to(self.device).eval()
        self.classifier = model

        acc = checkpoint.get('acc', 'N/A')
        logger.info(f"✅ 分类器加载完成 (准确率: {acc}%, 类别数: {len(self.classes) if self.classes else 100})")

    def detect_single(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        对单帧进行检测

        Args:
            frame: BGR 格式的图像

        Returns:
            检测结果列表
        """
        if self.yolo_model is None:
            return []

        results = []
        conf_threshold = self.config.confidence_threshold

        for r in self.yolo_model(frame, verbose=False):
            for box in r.boxes:
                if box.conf[0] >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    # 分类
                    class_name, prob = self._classify_crop(crop)

                    results.append(DetectionResult(
                        bbox=[x1, y1, x2, y2],
                        confidence=float(box.conf[0]),
                        class_name=class_name,
                        probability=prob
                    ))

        return results

    def _classify_crop(self, crop: np.ndarray) -> tuple:
        """对裁剪区域进行分类"""
        if self.classifier is None:
            return "unknown", 0.0

        try:
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.classifier(tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, idx = torch.max(probs, 1)

            class_name = self.classes[idx.item()] if idx.item() < len(self.classes) else "unknown"
            return class_name, conf.item()
        except Exception as e:
            logger.error(f"分类错误: {e}")
            return "error", 0.0

    def visualize_results(self, frame: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """在帧上绘制检测结果"""
        vis = frame.copy()

        for r in results:
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{r.class_name} {r.probability*100:.0f}%"
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return vis

    # ========== 视频流处理 ==========

    def start_stream(self, url: Optional[str] = None) -> bool:
        """启动视频流处理"""
        if self._running:
            logger.warning("视频流已在运行")
            return False

        if url:
            self._camera_url = url

        # 确保模型已加载
        if self.yolo_model is None:
            if not self.load_models():
                return False

        self._running = True
        self._detection_thread = Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()
        logger.info(f"视频流处理已启动: {self._camera_url}")
        return True

    def stop_stream(self):
        """停止视频流处理"""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("视频流处理已停止")

    def _detection_loop(self):
        """检测循环（在独立线程中运行）"""
        self._cap = cv2.VideoCapture(self._camera_url)

        if not self._cap.isOpened():
            logger.error(f"无法连接摄像头: {self._camera_url}")
            self._running = False
            return

        logger.info("摄像头连接成功")

        start_time = time.time()
        frame_count = 0

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue

            frame_count += 1

            # 检测
            results = self.detect_single(frame)

            # 可视化
            vis_frame = self.visualize_results(frame, results)

            # 计算 FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # 添加 FPS 显示
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 更新状态
            with self._frame_lock:
                self._frame = vis_frame
                self._results = results
                self._fps = fps

        if self._cap:
            self._cap.release()
        logger.info("检测循环结束")

    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def get_current_results(self) -> List[Dict]:
        """获取当前检测结果"""
        with self._frame_lock:
            return [
                {
                    'bbox': r.bbox,
                    'conf': r.confidence,
                    'class': r.class_name,
                    'prob': r.probability
                }
                for r in self._results
            ]

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        with self._frame_lock:
            return {
                'running': self._running,
                'fps': round(self._fps, 1),
                'results': self.get_current_results(),
                'camera_url': self._camera_url,
                'device': str(self.device)
            }

    @property
    def is_running(self) -> bool:
        return self._running


# 便捷函数
def get_service() -> DetectionService:
    """获取 DetectionService 单例"""
    return DetectionService()
