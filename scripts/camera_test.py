#!/usr/bin/env python3
"""
实时摄像头靶心识别测试
集成 IP Webcam 视频流 + 靶心检测管道
"""

import sys
import cv2
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

# 添加当前项目路径（优先）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入当前项目的摄像头配置
try:
    from config.camera_config import VIDEO_URL as CAMERA_VIDEO_URL
    camera_config_available = True
except ImportError:
    camera_config_available = False

# 添加 ip-webcam-opencv 项目路径（备用）
sys.path.insert(0, '/home/zishu/.openclaw/workspace/ip-webcam-opencv')

# 如果当前项目没有配置，尝试导入 ip-webcam-opencv 的配置
if not camera_config_available:
    try:
        import config as webcam_config
    except ImportError:
        webcam_config = None

# 导入靶心识别管道
from src.models.integrated_pipeline import BullseyePipeline

logger = logging.getLogger(__name__)


class CameraBullseyeTester:
    """实时摄像头靶心检测测试器"""

    def __init__(
        self,
        yolo_model_path: str | Path,
        cifar_model_path: str | Path,
        camera_url: Optional[str] = None,
        conf_threshold: float = 0.25,
    ):
        """
        初始化测试器

        Args:
            yolo_model_path: YOLO 模型路径
            cifar_model_path: CIFAR-100 模型路径
            camera_url: 摄像头 URL（默认使用 ip-webcam-opencv 的配置）
            conf_threshold: 检测置信度阈值
        """
        # 初始化靶心检测管道
        logger.info("初始化靶心检测管道...")
        self.pipeline = BullseyePipeline(
            yolo_model_path=yolo_model_path,
            cifar_model_path=cifar_model_path,
            conf_threshold=conf_threshold,
        )
        logger.info("✅ 管道初始化完成")

        # 设置摄像头 URL（优先级：参数 > 当前项目配置 > ip-webcam-opencv配置 > 默认值）
        if camera_url:
            self.camera_url = camera_url
        elif camera_config_available:
            self.camera_url = CAMERA_VIDEO_URL
        elif webcam_config:
            self.camera_url = webcam_config.VIDEO_URL
        else:
            self.camera_url = "http://10.24.100.139:8080/video"

        # 统计数据
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None

    def test_connection(self) -> bool:
        """测试摄像头连接"""
        logger.info(f"测试连接: {self.camera_url}")
        cap = cv2.VideoCapture(self.camera_url)

        if not cap.isOpened():
            logger.error("❌ 无法连接到摄像头")
            logger.error("请检查：")
            logger.error("1. IP Webcam App 是否已启动")
            logger.error("2. 手机和电脑是否在同一 Wi-Fi 网络")
            logger.error("3. 摄像头 URL 是否正确")
            return False

        logger.info("✅ 连接成功！")
        cap.release()
        return True

    def capture_snapshot(self, output_file: str = "snapshot.jpg") -> bool:
        """捕获单张快照"""
        logger.info(f"捕获快照...")
        cap = cv2.VideoCapture(self.camera_url)

        if not cap.isOpened():
            logger.error("❌ 无法连接到摄像头")
            return False

        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(output_file, frame)
            logger.info(f"✅ 快照已保存: {output_file}")
            return True
        else:
            logger.error("❌ 无法捕获帧")
            return False

    def run_realtime_detection(
        self,
        display: bool = True,
        save_video: bool = False,
        output_file: str = "detection_output.mp4",
        fps_target: int = 30,
    ) -> Tuple[int, int]:
        """
        实时靶心检测

        Args:
            display: 是否显示视频窗口
            save_video: 是否保存检测结果视频
            output_file: 输出视频文件名
            fps_target: 目标帧率

        Returns:
            (总帧数, 检测成功帧数)
        """
        logger.info("=" * 60)
        logger.info("开始实时靶心检测")
        logger.info("=" * 60)
        logger.info(f"摄像头 URL: {self.camera_url}")
        logger.info(f"显示窗口: {'是' if display else '否'}")
        logger.info(f"保存视频: {'是' if save_video else '否'}")

        # 连接摄像头
        cap = cv2.VideoCapture(self.camera_url)

        if not cap.isOpened():
            logger.error("❌ 无法连接到摄像头")
            return 0, 0

        # 获取视频参数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or fps_target

        logger.info(f"📹 视频参数: {width}x{height} @ {actual_fps}fps")

        # 视频写入器
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_file, fourcc, actual_fps, (width, height))
            logger.info(f"💾 正在保存到: {output_file}")

        # 重置统计
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()

        logger.info("按 'q' 退出, 's' 保存当前帧, 'p' 暂停/继续")

        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()

                if not ret:
                    logger.warning("⚠️ 无法读取帧，连接可能中断")
                    break

                self.frame_count += 1

                # 执行靶心检测
                result = self.pipeline.process(frame)

                # 统计检测成功
                if result['status'] == 'success':
                    self.detection_count += 1

                # 可视化
                if display or save_video:
                    vis_frame = self._draw_results(frame, result)

                    # 添加统计信息
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    success_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0

                    info_text = f"FPS: {fps:.1f} | Frames: {self.frame_count} | Detection Rate: {success_rate:.1f}%"
                    cv2.putText(vis_frame, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if display:
                        cv2.imshow('Bullseye Detection - IP Webcam', vis_frame)

                    if save_video and writer:
                        writer.write(vis_frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("\n👋 用户退出")
                break
            elif key == ord('s'):
                # 保存当前帧
                snapshot_file = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snapshot_file, frame if paused else vis_frame)
                logger.info(f"📸 已保存: {snapshot_file}")
            elif key == ord('p'):
                paused = not paused
                logger.info(f"{'⏸️ 暂停' if paused else '▶️ 继续'}")

        # 清理
        cap.release()
        if writer:
            writer.release()
            logger.info(f"✅ 视频已保存: {output_file}")
        cv2.destroyAllWindows()

        # 打印统计
        self._print_statistics()

        return self.frame_count, self.detection_count

    def _draw_results(self, frame, result):
        """绘制检测结果"""
        vis = frame.copy()

        if result['status'] == 'success':
            for detection in result['detections']:
                # 绘制边界框
                bbox = detection['bbox']
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                # 绘制中心点
                center = detection['center']
                cv2.circle(vis, center, 5, (0, 0, 255), -1)

                # 绘制裁剪区域
                crop_size = self.pipeline.crop_size
                cv2.rectangle(
                    vis,
                    (center[0] - crop_size[0] // 2, center[1] - crop_size[1] // 2),
                    (center[0] + crop_size[0] // 2, center[1] + crop_size[1] // 2),
                    (255, 0, 0), 1
                )

                # 绘制分类结果
                classification = detection['classification'][0]
                label = f"{classification['class_name']}: {classification['probability']:.2f}"
                cv2.putText(vis, label, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 显示置信度
                conf_label = f"Conf: {detection['confidence']:.2f}"
                cv2.putText(vis, conf_label, (bbox[0], bbox[3] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        else:
            # 未检测到靶心
            cv2.putText(vis, "No Bullseye Detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return vis

    def _print_statistics(self):
        """打印统计信息"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        logger.info("=" * 60)
        logger.info("检测统计")
        logger.info("=" * 60)
        logger.info(f"总帧数: {self.frame_count}")
        logger.info(f"检测成功: {self.detection_count}")
        logger.info(f"检测率: {(self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0:.2f}%")
        logger.info(f"运行时间: {elapsed:.2f}秒")
        logger.info(f"平均FPS: {(self.frame_count / elapsed) if elapsed > 0 else 0:.2f}")
        logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时摄像头靶心识别测试')

    # 模型路径
    parser.add_argument('--yolo', type=str,
                       default='runs/detect/runs/detect/bullseye_optimized/weights/best.pt',
                       help='YOLO 模型路径')
    parser.add_argument('--cifar', type=str,
                       default='experiments/runs/cifar100_resnet18/best.pt',
                       help='CIFAR-100 模型路径')

    # 摄像头配置
    parser.add_argument('--camera-url', type=str, default=None,
                       help='摄像头 URL (默认使用 ip-webcam-opencv 配置)')
    parser.add_argument('--test-connection', action='store_true',
                       help='仅测试摄像头连接')

    # 检测参数
    parser.add_argument('--conf', type=float, default=0.25,
                       help='检测置信度阈值')

    # 显示和保存
    parser.add_argument('--no-display', action='store_true',
                       help='不显示视频窗口')
    parser.add_argument('--save', action='store_true',
                       help='保存检测结果视频')
    parser.add_argument('--output', type=str, default='detection_output.mp4',
                       help='输出视频文件名')

    # 快照模式
    parser.add_argument('--snapshot', action='store_true',
                       help='捕获单张快照')

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 确定项目根目录
    project_root = Path(__file__).parent.parent
    yolo_path = project_root / args.yolo
    cifar_path = project_root / args.cifar

    # 检查模型文件
    if not yolo_path.exists():
        logger.error(f"YOLO 模型不存在: {yolo_path}")
        logger.info("请先运行训练脚本")
        return

    if not cifar_path.exists():
        logger.error(f"CIFAR-100 模型不存在: {cifar_path}")
        logger.info("请先运行训练脚本")
        return

    # 创建测试器
    tester = CameraBullseyeTester(
        yolo_model_path=yolo_path,
        cifar_model_path=cifar_path,
        camera_url=args.camera_url,
        conf_threshold=args.conf,
    )

    # 测试连接模式
    if args.test_connection:
        success = tester.test_connection()
        sys.exit(0 if success else 1)

    # 快照模式
    if args.snapshot:
        success = tester.capture_snapshot()
        sys.exit(0 if success else 1)

    # 实时检测模式
    display = not args.no_display
    tester.run_realtime_detection(
        display=display,
        save_video=args.save,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
