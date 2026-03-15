 # 机器人靶心高精度识别与定位系统

  ## 📌 项目简介

  **目标赛事**：瑞抗机器人开发者大赛 /
  中国机器人及人工智能大赛

  **核心挑战**：解决远距离（小目标）识别率低、环境
  干扰大、识别距离受限的问题。

  **技术路径**：YOLO 目标检测 + 形态学特征预处理 +
  空间几何定位

  ---

  ## 📁 项目目录结构

  robotic-bullseye/
  ├── src/                    # 核心代码
  │   ├── data/              # 数据处理工具
  │   ├── preprocessing/      # 特征预处理算法
  │   ├── models/            # YOLO 模型封装
  │   └── localization/       # 定位算法
  ├── config/                 # 配置文件
  ├── data/                   # 数据集目录
  ├── experiments/            # 训练输出
  └── scripts/               # 可执行脚本

  ---

  ## 🚀 快速开始

  ### 环境依赖

  - Python >= 3.8
  - PyTorch >= 1.10
  - YOLOv8/v10 ultralytics>=8.0.0  
  - opencv-python>=4.8.0   pillow>=10.0.0
  - numpy>=1.24.0  scipy>=1.10.0
  -  matplotlib>=3.7.0  tensorboard>=2.14.0 
  -  tqdm>=4.65.0 
  ### 安装步骤

  ```bash
  # 1. 克隆仓库
  git clone <>

  # 2. 创建虚拟环境
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  # 或 venv\Scripts\activate  # Windows

  # 3. 安装依赖
  pip install -r requirements.txt

  ---
  📖 开发文档

  详细开发计划请查看：./DEVELOPMENT_PLAN.md

  ---
  📝 许可证

  MIT License
