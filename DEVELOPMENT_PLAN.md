# 机器人靶心高精度识别与定位系统开发计划 (Competition Edition)

## 0. 项目背景
- **目标赛事**：瑞抗机器人开发者大赛 / 中国机器人及人工智能大赛
- **核心挑战**：解决远距离（小目标）识别率低、环境干扰大、识别距离受限的问题。
- **技术路径**：YOLO 目标检测 + 特征预处理（黑/灰圆环形态学提取）+ 空间几何定位。

## 1. 任务拆解与里程碑 (Atomic Tasks)

### 阶段一：数据工程与环境搭建 [分支: `feat/data-prep`]
- [ ] **Task 1.1**: 目录结构标准化（符合 Git 规范，区分 `assets/`, `src/`, `config/`）。
- [ ] **Task 1.2**: 原始图集（Atlas）的类别检查与增强（针对远距离样本进行超分辨率或锐化处理）。
- [ ] **Task 1.3**: 验证环境依赖（PyTorch/NVIDIA TensorRT 加速需求）。
- **Commit Pattern**: `feat: setup project structure and dataset inspection`

### 阶段二：特征预处理算法开发 [分支: `feat/pre-processing`]
- [ ] **Task 2.1**: 编写基于颜色空间（HSV/LAB）的黑色/灰色圆环提取脚本。
- [ ] **Task 2.2**: 引入形态学操作（闭运算/孔洞填充）以分离靶标边缘。
- [ ] **Task 2.3**: 验证预处理对“远距离小目标”对比度的提升效果。
- **Commit Pattern**: `feat: implement concentric ring feature extraction`

### 阶段三：模型训练与策略优化 [分支: `feat/yolo-train`]
- [ ] **Task 3.1**: 配置 YOLOv8/v10 小目标检测层（P2 层/增大输入分辨率）。
- [ ] **Task 3.2**: 融合“环形特征”作为模型辅助输入或注意力机制引导。
- [ ] **Task 3.3**: 针对复杂背景进行负样本（Hard Negative Mining）训练。
- **Commit Pattern**: `feat: optimize YOLO model for small target detection`

### 阶段四：定位算法与实战测试 [分支: `feat/localization`]
- [ ] **Task 4.1**: 基于识别出的像素坐标进行 PnP 或质心解算，实现靶心定位。
- [ ] **Task 4.2**: 性能压测（FPS/延迟）以满足机器人实时打击要求。
- **Commit Pattern**: `fix: improve localization accuracy and latency`

## 2. 交互协议 (Mentor Mode)
1. **启发式提问**：每一步请先询问我的思路。
2. **代码规范**：所有 Python 代码需符合 PEP8 规范。
3. **知识点讲解**：涉及到“形态学算子”或“YOLO 损失函数”时，需提供简要原理解析。
