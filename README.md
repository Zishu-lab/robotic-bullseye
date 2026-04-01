# 机器人靶心高精度识别与定位系统

## 项目简介

**目标赛事**: 瑞抗机器人开发者大赛 / 中国机器人及人工智能大赛

**核心挑战**: 解决远距离（小目标）识别率低、环境干扰大、识别距离受限的问题。

**技术路径**: YOLO 目标检测 + EfficientNet-B3 分类 + 形态学特征预处理

---

## 项目目录结构

```
robotic-bullseye/
├── src/
│   ├── models/
│   │   └── integrated_pipeline.py  # 集成推理管道 (YOLO + 分类器)
│   ├── preprocessing/
│   │   ├── ring_extractor.py    # 圆环特征提取 (用于训练数据增强/调试)
│   │   └── center_crop.py       # 中心裁剪
│   └── services/
│       └── detection_service.py  # 检测服务封装 (单例模式)
├── config/
│   ├── camera_config.py        # 摄像头配置
│   └── model_config.yaml       # 模型路径配置
├── scripts/
│   ├── train_cifar100_optimized.py  # EfficientNet-B3 训练 (80.18%)
│   └── ...
├── experiments/runs/
│   └── cifar100_optimized/     # 新模型 (80.18%)
├── app.py                     # Web 界面
└── requirements.txt
```

---

## 核心模块说明

### 1. 集成推理管道 (`src/models/integrated_pipeline.py`)
- 连接 YOLO 检测 + CIFAR-100 分类
- 自动检测模型类型 (ResNet / EfficientNet)
- 支持图片和视频流处理

### 2. 检测服务 (`src/services/detection_service.py`)
- 单例模式，统一管理模型和推理
- 配置文件驱动 (`config/model_config.yaml`)
- 异步视频流处理

### 3. 圆环特征提取 (`src/preprocessing/ring_extractor.py`)
- **用途**: 训练数据增强、调试分析
- 提取黑色外圈和灰色圆环特征
- CLAHE 对比度增强

---

## 快速开始

### 环境依赖

- Python >= 3.8
- PyTorch >= 1.10
- YOLOv8 ultralytics >= 8.0.0
- opencv-python >= 4.8.0
- torchvision >= 0.15.0

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Zishu-lab/robotic-bullseye.git

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 启动 Web 界面

```bash
python app.py
# 访问 http://localhost:5000
```

---

## 模型性能

| 模型 | 准确率 | 用途 |
|------|--------|------|
| EfficientNet-B3 | **80.18%** | CIFAR-100 分类 (当前) |
| ResNet-50 | 68.53% | CIFAR-100 分类 (旧版) |
| YOLOv8m | 99.4% mAP50 | 靶心检测 |

---

## 配置文件

模型路径配置 (`config/model_config.yaml`):

```yaml
models:
  yolo:
    path: runs/detect/runs/detect/bullseye_train/weights/best.pt
    confidence_threshold: 0.3
  classifier:
    path: experiments/runs/cifar100_optimized/best.pt

device:
  type: auto  # auto, cuda, cpu
```

---

## 开发文档

详细开发计划请查看: [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md)

---

## 许可证

MIT License
