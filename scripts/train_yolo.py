#!/usr/bin/env python3
"""
YOLO 小目标检测训练脚本
针对小目标优化的配置
"""

from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_yolo_config(
    output_path: Path,
    dataset_path: Path,
    model_size: str = "s",  # n, s, m, l, x
    img_size: int = 640,     # 输入分辨率
    epochs: int = 100,
    batch_size: int = 16,
):
    """
    创建 YOLO 训练配置

    Args:
        output_path: 配置文件输出路径
        dataset_path: 数据集路径
        model_size: 模型大小
        img_size: 输入图像尺寸
        epochs: 训练轮数
        batch_size: 批次大小
    """
    config = {
        # 模型配置
        'model': f'yolov8{model_size}.pt',
        'data': str(dataset_path / 'dataset.yaml'),

        # 训练参数
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,

        # 小目标优化配置
        'optimizer': 'AdamW',       # 优化器
        'lr0': 0.01,                # 初始学习率
        'lrf': 0.01,                # 最终学习率 (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # 数据增强
        'hsv_h': 0.015,             # 色调增强
        'hsv_s': 0.7,               # 饱和度增强
        'hsv_v': 0.4,               # 明度增强
        'degrees': 0.0,             # 旋转角度（靶心不需要旋转）
        'translate': 0.1,           # 平移
        'scale': 0.5,               # 缩放
        'shear': 0.0,               # 剪切
        'perspective': 0.0,         # 透视变换
        'flipud': 0.0,              # 上下翻转
        'fliplr': 0.5,              # 左右翻转
        'mosaic': 1.0,              # Mosaic 增强
        'mixup': 0.0,               # MixUp 增强

        # 小目标检测优化
        'box': 7.5,                 # box loss gain
        'cls': 0.5,                 # cls loss gain
        'dfl': 1.5,                 # dfl loss gain

        # 训练设置
        'patience': 50,             # 早停耐心值
        'save': True,               # 保存检查点
        'save_period': 10,          # 每 10 轮保存一次
        'cache': 'ram',             # 缓存数据到内存
        'device': 0,                # GPU 设备
        'workers': 8,               # 数据加载线程数
        'project': 'runs/detect',   # 项目目录
        'name': 'bullseye_train',   # 实验名称
        'exist_ok': True,           # 覆盖已存在的实验
        'pretrained': True,         # 使用预训练权重
        'verbose': True,            # 详细输出
        'seed': 42,                 # 随机种子
        'deterministic': True,      # 确定性训练
        'single_cls': True,         # 单类别训练
        'rect': False,              # 矩形训练
        'cos_lr': True,             # 余弦学习率调度
        'label_smoothing': 0.0,     # 标签平滑
        'dropout': 0.0,             # Dropout
        'val': True,                # 验证
        'plots': True,              # 保存训练曲线图
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"配置文件已保存: {output_path}")
    return config


def train_yolo(config_path: Path):
    """
    训练 YOLO 模型

    Args:
        config_path: 配置文件路径
    """
    from ultralytics import YOLO

    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("开始 YOLO 训练")
    logger.info("=" * 60)
    logger.info(f"模型: {config['model']}")
    logger.info(f"数据集: {config['data']}")
    logger.info(f"分辨率: {config['imgsz']}")
    logger.info(f"轮数: {config['epochs']}")
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
    )

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"最佳模型保存在: {results.save_dir / 'weights' / 'best.pt'}")
    logger.info("=" * 60)

    return results


def main():
    """主函数"""
    # 配置路径
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "yolo_dataset"
    config_path = project_root / "config" / "model" / "yolo_config.yaml"

    # 创建配置目录
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建训练配置
    logger.info("步骤 1: 创建训练配置...")
    config = create_yolo_config(
        output_path=config_path,
        dataset_path=dataset_path,
        model_size="s",      # 使用 YOLOv8s (small)
        img_size=640,        # 输入分辨率 640x640
        epochs=100,          # 训练 100 轮
        batch_size=16,       # 批次大小 16
    )

    # 打印配置
    logger.info("\n训练配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # 开始训练
    logger.info("\n步骤 2: 开始训练...")
    results = train_yolo(config_path)


if __name__ == "__main__":
    main()
