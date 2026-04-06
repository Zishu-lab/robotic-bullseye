#!/usr/bin/env python3
"""
实战场景测试脚本
测试模型在不同场景下的表现：
1. 不同距离（缩放测试）
2. 不同光照（亮度/对比度变化）
3. 不同背景（噪声干扰）
4. 批量评估所有测试图像
"""

import cv2
import numpy as np
import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import time
from datetime import datetime

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.integrated_pipeline import BullseyePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果"""
    scenario: str
    image_name: str
    detection_count: int
    detection_confidence: float
    classification_result: str
    classification_confidence: float
    inference_time_ms: float
    success: bool
    notes: str = ""


class ScenarioTester:
    """实战场景测试器"""

    def __init__(
        self,
        yolo_model_path: str | Path,
        cifar_model_path: str | Path,
        output_dir: str | Path = "data/test_results"
    ):
        self.pipeline = BullseyePipeline(
            yolo_model_path=yolo_model_path,
            cifar_model_path=cifar_model_path,
            conf_threshold=0.25
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.output_dir / "distance").mkdir(exist_ok=True)
        (self.output_dir / "lighting").mkdir(exist_ok=True)
        (self.output_dir / "noise").mkdir(exist_ok=True)
        (self.output_dir / "batch").mkdir(exist_ok=True)

    def test_distance(self, image: np.ndarray, image_name: str) -> List[TestResult]:
        """
        测试不同距离（通过缩放模拟）

        Args:
            image: 输入图像
            image_name: 图像名称

        Returns:
            测试结果列表
        """
        results = []
        scales = [1.0, 0.75, 0.5, 0.35, 0.25]  # 模拟不同距离
        scale_names = ["近距离(100%)", "中距离(75%)", "远距离(50%)", "超远距离(35%)", "极远距离(25%)"]

        logger.info(f"\n{'='*60}")
        logger.info(f"距离测试: {image_name}")
        logger.info(f"{'='*60}")

        for scale, scale_name in zip(scales, scale_names):
            # 缩放图像
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(image, (new_w, new_h))

            # 测试
            start_time = time.time()
            result = self.pipeline.process(scaled_image)
            inference_time = (time.time() - start_time) * 1000

            # 记录结果
            if result['status'] == 'success' and len(result['detections']) > 0:
                detection = result['detections'][0]
                test_result = TestResult(
                    scenario=f"distance_{scale_name}",
                    image_name=image_name,
                    detection_count=len(result['detections']),
                    detection_confidence=detection['confidence'],
                    classification_result=detection['classification'][0]['class_name'],
                    classification_confidence=detection['classification'][0]['probability'],
                    inference_time_ms=inference_time,
                    success=True
                )

                # 保存可视化结果
                vis_image = self._visualize_result(scaled_image, result)
                output_path = self.output_dir / "distance" / f"{image_name}_{scale_name.replace('(', '_').replace(')', '').replace('%', '')}.png"
                cv2.imwrite(str(output_path), vis_image)
            else:
                test_result = TestResult(
                    scenario=f"distance_{scale_name}",
                    image_name=image_name,
                    detection_count=0,
                    detection_confidence=0.0,
                    classification_result="N/A",
                    classification_confidence=0.0,
                    inference_time_ms=inference_time,
                    success=False,
                    notes="未检测到目标"
                )

            results.append(test_result)
            logger.info(f"  {scale_name}: 检测={test_result.success}, 置信度={test_result.detection_confidence:.2f}, 时间={inference_time:.1f}ms")

        return results

    def test_lighting(self, image: np.ndarray, image_name: str) -> List[TestResult]:
        """
        测试不同光照条件

        Args:
            image: 输入图像
            image_name: 图像名称

        Returns:
            测试结果列表
        """
        results = []
        lighting_conditions = [
            ("正常光照", 1.0, 0),
            ("明亮", 1.5, 50),
            ("暗淡", 0.5, -50),
            ("高对比度", 1.2, 30),
            ("低对比度", 0.8, 20),
            ("过曝", 2.0, 100),
            ("欠曝", 0.3, -80),
        ]

        logger.info(f"\n{'='*60}")
        logger.info(f"光照测试: {image_name}")
        logger.info(f"{'='*60}")

        for condition_name, alpha, beta in lighting_conditions:
            # 调整亮度和对比度
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            # 测试
            start_time = time.time()
            result = self.pipeline.process(adjusted)
            inference_time = (time.time() - start_time) * 1000

            # 记录结果
            if result['status'] == 'success' and len(result['detections']) > 0:
                detection = result['detections'][0]
                test_result = TestResult(
                    scenario=f"lighting_{condition_name}",
                    image_name=image_name,
                    detection_count=len(result['detections']),
                    detection_confidence=detection['confidence'],
                    classification_result=detection['classification'][0]['class_name'],
                    classification_confidence=detection['classification'][0]['probability'],
                    inference_time_ms=inference_time,
                    success=True
                )

                # 保存可视化结果
                vis_image = self._visualize_result(adjusted, result)
                output_path = self.output_dir / "lighting" / f"{image_name}_{condition_name}.png"
                cv2.imwrite(str(output_path), vis_image)
            else:
                test_result = TestResult(
                    scenario=f"lighting_{condition_name}",
                    image_name=image_name,
                    detection_count=0,
                    detection_confidence=0.0,
                    classification_result="N/A",
                    classification_confidence=0.0,
                    inference_time_ms=inference_time,
                    success=False,
                    notes="未检测到目标"
                )

            results.append(test_result)
            logger.info(f"  {condition_name}: 检测={test_result.success}, 置信度={test_result.detection_confidence:.2f}")

        return results

    def test_noise(self, image: np.ndarray, image_name: str) -> List[TestResult]:
        """
        测试不同噪声干扰

        Args:
            image: 输入图像
            image_name: 图像名称

        Returns:
            测试结果列表
        """
        results = []
        noise_levels = [
            ("无噪声", 0),
            ("轻微噪声", 10),
            ("中等噪声", 25),
            ("强噪声", 50),
            ("极强噪声", 75),
        ]

        logger.info(f"\n{'='*60}")
        logger.info(f"噪声测试: {image_name}")
        logger.info(f"{'='*60}")

        for noise_name, sigma in noise_levels:
            # 添加高斯噪声
            if sigma > 0:
                noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
                noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            else:
                noisy_image = image.copy()

            # 测试
            start_time = time.time()
            result = self.pipeline.process(noisy_image)
            inference_time = (time.time() - start_time) * 1000

            # 记录结果
            if result['status'] == 'success' and len(result['detections']) > 0:
                detection = result['detections'][0]
                test_result = TestResult(
                    scenario=f"noise_{noise_name}",
                    image_name=image_name,
                    detection_count=len(result['detections']),
                    detection_confidence=detection['confidence'],
                    classification_result=detection['classification'][0]['class_name'],
                    classification_confidence=detection['classification'][0]['probability'],
                    inference_time_ms=inference_time,
                    success=True
                )

                # 保存可视化结果
                vis_image = self._visualize_result(noisy_image, result)
                output_path = self.output_dir / "noise" / f"{image_name}_{noise_name}.png"
                cv2.imwrite(str(output_path), vis_image)
            else:
                test_result = TestResult(
                    scenario=f"noise_{noise_name}",
                    image_name=image_name,
                    detection_count=0,
                    detection_confidence=0.0,
                    classification_result="N/A",
                    classification_confidence=0.0,
                    inference_time_ms=inference_time,
                    success=False,
                    notes="未检测到目标"
                )

            results.append(test_result)
            logger.info(f"  {noise_name}: 检测={test_result.success}, 置信度={test_result.detection_confidence:.2f}")

        return results

    def batch_test(self, image_dir: str | Path) -> List[TestResult]:
        """
        批量测试所有图像

        Args:
            image_dir: 图像目录

        Returns:
            测试结果列表
        """
        results = []
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

        logger.info(f"\n{'='*60}")
        logger.info(f"批量测试: {len(image_files)} 张图像")
        logger.info(f"{'='*60}")

        for i, image_path in enumerate(image_files):
            if "_result" in image_path.stem or "_comparison" in image_path.stem:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            start_time = time.time()
            result = self.pipeline.process(image)
            inference_time = (time.time() - start_time) * 1000

            if result['status'] == 'success' and len(result['detections']) > 0:
                detection = result['detections'][0]
                test_result = TestResult(
                    scenario="batch_test",
                    image_name=image_path.name,
                    detection_count=len(result['detections']),
                    detection_confidence=detection['confidence'],
                    classification_result=detection['classification'][0]['class_name'],
                    classification_confidence=detection['classification'][0]['probability'],
                    inference_time_ms=inference_time,
                    success=True
                )

                # 保存可视化结果
                vis_image = self._visualize_result(image, result)
                output_path = self.output_dir / "batch" / f"{image_path.stem}_result.png"
                cv2.imwrite(str(output_path), vis_image)
            else:
                test_result = TestResult(
                    scenario="batch_test",
                    image_name=image_path.name,
                    detection_count=0,
                    detection_confidence=0.0,
                    classification_result="N/A",
                    classification_confidence=0.0,
                    inference_time_ms=inference_time,
                    success=False,
                    notes="未检测到目标"
                )

            results.append(test_result)
            logger.info(f"  [{i+1}/{len(image_files)}] {image_path.name}: 检测={test_result.success}, 置信度={test_result.detection_confidence:.2f}")

        return results

    def _visualize_result(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """可视化结果"""
        vis = image.copy()

        if result['status'] != 'success':
            return vis

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

    def generate_report(self, all_results: Dict[str, List[TestResult]]) -> str:
        """
        生成测试报告

        Args:
            all_results: 所有测试结果

        Returns:
            报告文本
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("实战场景测试报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)

        for scenario_name, results in all_results.items():
            if not results:
                continue

            report_lines.append(f"\n## {scenario_name.upper()} 测试结果")
            report_lines.append("-" * 60)

            total = len(results)
            success = sum(1 for r in results if r.success)
            success_rate = (success / total * 100) if total > 0 else 0

            avg_confidence = np.mean([r.detection_confidence for r in results if r.success]) if success > 0 else 0
            avg_time = np.mean([r.inference_time_ms for r in results])

            report_lines.append(f"总测试数: {total}")
            report_lines.append(f"成功检测: {success}")
            report_lines.append(f"检测率: {success_rate:.1f}%")
            report_lines.append(f"平均置信度: {avg_confidence:.2f}")
            report_lines.append(f"平均推理时间: {avg_time:.1f}ms")

            # 失败详情
            failed = [r for r in results if not r.success]
            if failed:
                report_lines.append(f"\n失败场景 ({len(failed)}):")
                for f in failed[:5]:  # 只显示前5个
                    report_lines.append(f"  - {f.scenario}: {f.image_name} - {f.notes}")

        # 总结
        report_lines.append("\n" + "=" * 80)
        report_lines.append("## 总结")
        report_lines.append("-" * 60)

        total_all = sum(len(r) for r in all_results.values())
        success_all = sum(sum(1 for r in results if r.success) for results in all_results.values())
        overall_rate = (success_all / total_all * 100) if total_all > 0 else 0

        report_lines.append(f"总测试场景数: {total_all}")
        report_lines.append(f"总成功检测数: {success_all}")
        report_lines.append(f"总体检测率: {overall_rate:.1f}%")

        report_lines.append("\n### 建议")
        if overall_rate >= 90:
            report_lines.append("- 模型表现优秀，可用于实际部署")
        elif overall_rate >= 70:
            report_lines.append("- 模型表现良好，建议针对失败场景进行优化")
        else:
            report_lines.append("- 模型需要进一步训练或调整参数")

        report_text = "\n".join(report_lines)

        # 保存报告
        report_path = self.output_dir / "test_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 保存JSON格式结果
        json_results = {
            scenario: [asdict(r) for r in results]
            for scenario, results in all_results.items()
        }
        json_path = self.output_dir / "test_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)

        logger.info(f"\n报告已保存到: {report_path}")
        logger.info(f"JSON结果已保存到: {json_path}")

        return report_text


def main():
    """主测试函数"""
    project_root = Path(__file__).parent.parent

    # 模型路径
    yolo_model_path = project_root / "runs/detect/runs/detect/bullseye_optimized/weights/best.pt"
    cifar_model_path = project_root / "experiments/runs/cifar100_resnet18/best.pt"

    # 测试图像目录
    test_image_dir = project_root / "data/raw"

    # 输出目录
    output_dir = project_root / "data/test_results"

    # 检查模型
    if not yolo_model_path.exists():
        logger.error(f"YOLO模型不存在: {yolo_model_path}")
        return

    if not cifar_model_path.exists():
        logger.error(f"CIFAR-100模型不存在: {cifar_model_path}")
        return

    logger.info("=" * 60)
    logger.info("开始实战场景测试")
    logger.info("=" * 60)

    # 创建测试器
    tester = ScenarioTester(
        yolo_model_path=yolo_model_path,
        cifar_model_path=cifar_model_path,
        output_dir=output_dir
    )

    all_results = {}

    # 1. 距离测试（使用第一张测试图像）
    test_images = list(test_image_dir.glob("*.png"))
    if test_images:
        sample_image = cv2.imread(str(test_images[0]))
        all_results['distance'] = tester.test_distance(sample_image, test_images[0].stem)

        # 2. 光照测试
        all_results['lighting'] = tester.test_lighting(sample_image, test_images[0].stem)

        # 3. 噪声测试
        all_results['noise'] = tester.test_noise(sample_image, test_images[0].stem)

    # 4. 批量测试
    all_results['batch'] = tester.batch_test(test_image_dir)

    # 生成报告
    report = tester.generate_report(all_results)
    print("\n" + report)


if __name__ == "__main__":
    main()
