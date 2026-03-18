#!/usr/bin/env python3
"""
YOLO 小目标检测优化训练脚本
综合优化 + YOLOv8m + 更高精度
"""

from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_optimized_config(
    output_path: Path,
    dataset_path: Path,
    model_size: str = "m",  # 升级到 m
    img_size: int = 960,     # 提高分辨率
    epochs: int = 150,        # 增加训练轮数
    batch_size: int = 12,     # 调整批次大小（m 模型更大）
):
    """
    创建优化的 YOLO 训练配置

    Args:
        output_path: 配置文件输出路径
        dataset_path: 数据集路径
        model_size: 模型大小 (m = medium)
        img_size: 输入图像尺寸（提高分辨率）
        epochs: 训练轮数
        batch_size: 批次大小
    """
    config = {
        # 模型配置（升级到 YOLOv8m）
        'model': f'yolov8{model_size}.pt',
        'data': str(dataset_path / 'dataset.yaml'),

        # 训练参数
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,

        # 优化器配置（AdamW + 学习率优化）
        'optimizer': 'AdamW',
        'lr0': 0.008,                # 降低初始学习率（更稳定）
        'lrf': 0.005,                # 降低最终学习率（更精细）
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # 数据增强（针对小目标优化）
        'hsv_h': 0.01,               # 色调增强
        'hsv_s': 0.6,                # 饱和度增强
        'hsv_v': 0.3,                # 明度增强
        'degrees': 0.0,              # 旋转角度（靶心不需要旋转）
        'translate': 0.1,            # 平移
        'scale': 0.7,                # 缩放范围增大（提高泛化）
        'shear': 0.0,                # 剪切
        'perspective': 0.0,          # 透视变换
        'flipud': 0.0,               # 上下翻转
        'fliplr': 0.5,               # 左右翻转
        'mosaic': 1.0,               # Mosaic 增强
        'mixup': 0.1,                # 添加 MixUp 增强（提高泛化）

        # 小目标检测优化
        'box': 8.0,                  # 提高 box loss gain
        'cls': 0.6,                  # 提高 cls loss gain
        'dfl': 1.8,                  # 提高 dfl loss gain

        # 训练设置
        'patience': 60,              # 增加早停耐心值
        'save': True,
        'save_period': 10,
        'cache': 'ram',
        'device': 0,
        'workers': 8,
        'project': 'runs/detect',
        'name': 'bullseye_optimized',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': True,
        'rect': False,
        'cos_lr': True,
        'label_smoothing': 0.0,      # 可尝试 0.01
        'dropout': 0.0,
        'val': True,
        'plots': True,
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"优化配置文件已保存: {output_path}")
    return config


def train_optimized_yolo(config_path: Path):
    """
    训练优化的 YOLO 模型

    Args:
        config_path: 配置文件路径
    """
    from ultralytics import YOLO

    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("开始优化训练 (YOLOv8m + 综合优化)")
    logger.info("=" * 60)
    logger.info(f"模型: {config['model']} (升级到 medium)")
    logger.info(f"数据集: {config['data']}")
    logger.info(f"分辨率: {config['imgsz']} (提升到 960)")
    logger.info(f"轮数: {config['epochs']} (增加轮数)")
    logger.info(f"批次大小: {config['batch']}")
    logger.info("=" * 60)

    # 加载模型
    model = YOLO(config['model'])

    # 训练
    results = model.train(
        data=config['data'],
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        device=config['device'],
        project=config['project'],
        name=config['name'],
        exist_ok=config['exist_ok'],
        pretrained=config['pretrained'],
        verbose=config['verbose'],
        seed=config['seed'],
        patience=config['patience'],
        save_period=config['save_period'],
        cache=config['cache'],
        plots=config['plots'],
        val=config['val'],
        single_cls=config['single_cls'],
        mixup=config['mixup'],  # 添加 MixUp
    )

    logger.info("=" * 60)
    logger.info("优化训练完成!")
    logger.info(f"最佳模型保存在: {results.save_dir / 'weights' / 'best.pt'}")
    logger.info("=" * 60)

    return results


def main():
    """主函数"""
    # 配置路径
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "yolo_dataset"
    config_path = project_root / "config" / "model" / "yolo_config_optimized.yaml"

    # 创建配置目录
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建优化配置
    logger.info("步骤 1: 创建优化配置...")
    config = create_optimized_config(
        output_path=config_path,
        dataset_path=dataset_path,
        model_size="m",      # 升级到 YOLOv8m
        img_size=960,        # 提高分辨率
        epochs=150,          # 增加训练轮数
        batch_size=12,       # 调整批次大小
    )

    # 打印配置
    logger.info("\n优化配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # 开始训练
    logger.info("\n步骤 2: 开始优化训练...")
    results = train_optimized_yolo(config_path)


if __name__ == "__main__":
    main()
