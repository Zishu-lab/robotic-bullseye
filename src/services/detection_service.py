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
from threading import Thread, Lock, Event
import logging
import time
import yaml
import socket
import urllib.parse

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
        self._stop_event = Event()  # 用于优雅停止线程
        self._camera_url = self.config.camera_url
        self._cap = None
        self._frame = None
        self._results: List[DetectionResult] = []
        self._fps = 0.0
        self._frame_lock = Lock()
        self._detection_thread: Optional[Thread] = None

        # 帧队列用于解耦读取和处理
        self._frame_queue: Queue = Queue(maxsize=2)
        self._reader_thread: Optional[Thread] = None

        # 缓存靶心中心点，避免每帧都做霍夫圆检测
        self._cached_center: Optional[tuple] = None
        self._center_update_counter: int = 0
        self._center_update_interval: int = 10  # 每10帧更新一次中心点

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

    def _find_bullseye_center(self, frame: np.ndarray, force_update: bool = False) -> tuple:
        """
        使用霍夫圆变换找到靶心中心点（带缓存优化）

        Args:
            frame: BGR 格式的图像
            force_update: 强制更新中心点

        Returns:
            (center_x, center_y, confidence)
        """
        h, w = frame.shape[:2]

        # 使用缓存，每隔 _center_update_interval 帧更新一次
        self._center_update_counter += 1
        if not force_update and self._cached_center is not None:
            if self._center_update_counter < self._center_update_interval:
                return self._cached_center
        self._center_update_counter = 0

        # 缩小图像加速检测（缩小到 1/4）
        scale = 0.25
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_h, small_w = small_frame.shape[:2]

        # 转灰度
        if len(small_frame.shape) == 3:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = small_frame.copy()

        # CLAHE 增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 高斯模糊降噪
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 霍夫圆变换（在缩小后的图像上检测）
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=50,  # 缩小后调整
            param1=50,
            param2=30,
            minRadius=10,  # 缩小后调整
            maxRadius=min(small_h, small_w) // 2
        )

        if circles is None or len(circles) == 0:
            # 没检测到圆，用图像中心
            self._cached_center = (w // 2, h // 2, 0.0)
            return self._cached_center

        circles = np.round(circles[0, :]).astype("int")

        # 多个圆取平均中心（还原到原始尺寸）
        center_x = int(np.mean(circles[:, 0]) / scale)
        center_y = int(np.mean(circles[:, 1]) / scale)

        # 置信度：检测到的圆越多越可信
        confidence = min(len(circles) / 5.0, 1.0)

        self._cached_center = (center_x, center_y, confidence)
        return self._cached_center

    def detect_single(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        对单帧进行检测

        Args:
            frame: BGR 格式的图像

        Returns:
            检测结果列表
        """
        if self.classifier is None:
            return []

        h, w = frame.shape[:2]

        # 更新计数器，决定是否需要重新检测中心点
        self._center_update_counter += 1
        if self._center_update_counter >= self._center_update_interval or self._cached_center is None:
            # 重新检测中心点
            self._find_bullseye_center(frame)
            self._center_update_counter = 0

        # 使用缓存的中心点
        center_x, center_y, confidence = self._cached_center

        # 裁剪尺寸（取图像较小边的 40%）
        crop_size = min(h, w) * 40 // 100
        half_size = crop_size // 2

        # 计算裁剪区域
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(w, center_x + half_size)
        y2 = min(h, center_y + half_size)

        # 裁剪中心区域
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return []

        # 分类
        class_name, prob = self._classify_crop(crop)

        return [DetectionResult(
            bbox=[x1, y1, x2, y2],
            confidence=confidence,
            class_name=class_name,
            probability=prob
        )]

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

        self._stop_event.clear()
        self._running = True

        # 启动帧读取线程
        self._reader_thread = Thread(target=self._frame_reader_loop, daemon=True)
        self._reader_thread.start()

        # 启动检测处理线程
        self._detection_thread = Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()

        logger.info(f"视频流处理已启动: {self._camera_url}")
        return True

    def stop_stream(self):
        """停止视频流处理"""
        self._stop_event.set()  # 发送停止信号
        self._running = False

        # 清空帧队列以解除阻塞
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except:
                break

        if self._cap:
            self._cap.release()
            self._cap = None

        # 等待线程结束（最多等待2秒）
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=1.0)

        logger.info("视频流处理已停止")

    def _frame_reader_loop(self):
        """帧读取线程 - 独立读取摄像头帧，避免阻塞检测"""
        logger.info(f"帧读取线程启动: {self._camera_url}")
        reconnect_delay = 1
        max_reconnect_delay = 10
        consecutive_failures = 0

        while not self._stop_event.is_set():
            try:
                # 连接或重连摄像头
                if self._cap is None or not self._cap.isOpened():
                    logger.info(f"尝试连接摄像头: {self._camera_url}")

                    # 释放旧的 VideoCapture
                    if self._cap:
                        self._cap.release()

                    self._cap = cv2.VideoCapture(self._camera_url)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if not self._cap.isOpened():
                        logger.warning(f"摄像头连接失败，{reconnect_delay}秒后重试")
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                        continue

                    logger.info("摄像头连接成功")
                    reconnect_delay = 1
                    consecutive_failures = 0

                # 读取帧
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        logger.warning("连续读取失败，尝试重连...")
                        self._cap.release()
                        self._cap = None
                        consecutive_failures = 0
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0

                # 将帧放入队列（非阻塞）
                try:
                    # 如果队列满了，丢弃旧帧
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except:
                            pass
                    self._frame_queue.put(frame, block=False)
                except:
                    pass

            except Exception as e:
                logger.error(f"帧读取错误: {e}")
                time.sleep(0.1)

        # 清理
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("帧读取线程结束")

    def _detection_loop(self):
        """检测循环（在独立线程中运行）- 从队列获取帧进行处理"""
        logger.info("检测处理线程启动")

        # 初始化计数器
        start_time = time.time()
        frame_count = 0
        last_frame_time = time.time()

        while not self._stop_event.is_set():
            try:
                # 从队列获取帧（带超时）
                try:
                    frame = self._frame_queue.get(timeout=0.5)
                except:
                    # 队列为空，继续等待
                    continue

                frame_count += 1
                last_frame_time = time.time()

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

                # 更新状态（使用锁保护）
                with self._frame_lock:
                    self._frame = vis_frame
                    self._results = results
                    self._fps = fps

                if frame_count % 100 == 0:
                    logger.info(f"已处理 {frame_count} 帧, FPS: {fps:.1f}")

            except Exception as e:
                logger.error(f"检测循环错误: {e}")
                time.sleep(0.1)

        self._running = False
        logger.info("检测处理线程结束")

    def get_current_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """获取当前帧（带超时保护）"""
        try:
            if self._frame_lock.acquire(timeout=timeout):
                try:
                    if self._frame is not None:
                        return self._frame.copy()
                    return None
                finally:
                    self._frame_lock.release()
        except Exception as e:
            logger.warning(f"获取帧超时或失败: {e}")
            return None

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
            results = [
                {
                    'bbox': r.bbox,
                    'conf': r.confidence,
                    'class': r.class_name,
                    'prob': r.probability
                }
                for r in self._results
            ]
            return {
                'running': self._running,
                'fps': round(self._fps, 1),
                'results': results,
                'camera_url': self._camera_url,
                'device': str(self.device)
            }

    @property
    def is_running(self) -> bool:
        return self._running

    def set_camera_url(self, url: str):
        """设置摄像头 URL"""
        self._camera_url = url
        logger.info(f"摄像头 URL 已更新: {url}")


# 便捷函数
def get_service() -> DetectionService:
    """获取 DetectionService 单例"""
    return DetectionService()
