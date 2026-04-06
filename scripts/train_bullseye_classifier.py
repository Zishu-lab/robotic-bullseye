#!/usr/bin/env python3
"""
靶心内容分类训练脚本
使用裁剪的靶心图片训练分类模型
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import json
import logging
from tqdm import tqdm
from PIL import Image
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BullseyeDataset(Dataset):
    """靶心分类数据集"""

    def __init__(self, image_dir: Path, labels_file: Path, transform=None, augment_factor=3):
        """
        初始化数据集

        Args:
            image_dir: 图片目录
            labels_file: 标签文件路径
            transform: 数据增强
            augment_factor: 数据增强倍数
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.augment_factor = augment_factor

        # 加载标签
        with open(labels_file, 'r') as f:
            data = json.load(f)

        self.classes = data['classes']
        self.class_to_idx = data['class_to_idx']
        self.labels = data['labels']

        # 构建样本列表
        self.samples = []
        for name, info in self.labels.items():
            img_path = self.image_dir / f"{name}.png"
            if img_path.exists():
                self.samples.append((str(img_path), info['class_idx']))

        logger.info(f"加载了 {len(self.samples)} 个样本，{len(self.classes)} 个类别")

    def __len__(self):
        return len(self.samples) * self.augment_factor

    def __getitem__(self, idx):
        # 映射到原始样本
        actual_idx = idx % len(self.samples)
        img_path, label = self.samples[actual_idx]

        # 加载图片
        image = Image.open(img_path).convert('RGB')

        # 数据增强
        if self.transform:
            image = self.transform(image)

        return image, label


class BullseyeClassifier:
    """靶心分类训练器"""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化训练器

        Args:
            data_dir: 数据集目录
            batch_size: 批次大小
            num_workers: 数据加载线程数
            device: 训练设备
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)

        logger.info(f"使用设备: {self.device}")

        # 加载类别信息
        labels_file = self.data_dir / "classification_labels.json"
        with open(labels_file, 'r') as f:
            data = json.load(f)
        self.classes = data['classes']
        self.num_classes = len(self.classes)

        logger.info(f"类别数: {self.num_classes}")
        logger.info(f"类别: {self.classes[:10]}...")

    def get_transforms(self, img_size=224):
        """获取数据增强策略"""
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return train_transform, val_transform

    def prepare_data(self, img_size=224, val_split=0.2):
        """准备数据集"""
        train_transform, val_transform = self.get_transforms(img_size)

        # 创建完整数据集
        full_dataset = BullseyeDataset(
            image_dir=self.data_dir / "cropped",
            labels_file=self.data_dir / "classification_labels.json",
            transform=None,
            augment_factor=1
        )

        # 划分训练集和验证集
        total_size = len(full_dataset.samples)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        indices = list(range(total_size))
        random.seed(42)
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # 创建训练集（带增强）
        train_dataset = BullseyeDataset(
            image_dir=self.data_dir / "cropped",
            labels_file=self.data_dir / "classification_labels.json",
            transform=train_transform,
            augment_factor=5  # 训练时增强5倍
        )
        train_dataset.samples = [full_dataset.samples[i] for i in train_indices]

        # 创建验证集（无增强）
        val_dataset = BullseyeDataset(
            image_dir=self.data_dir / "cropped",
            labels_file=self.data_dir / "classification_labels.json",
            transform=val_transform,
            augment_factor=1
        )
        val_dataset.samples = [full_dataset.samples[i] for i in val_indices]

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        logger.info(f"训练集: {len(train_dataset)} 张图片")
        logger.info(f"验证集: {len(val_dataset)} 张图片")

        return train_loader, val_loader

    def create_model(self):
        """创建 ResNet-50 分类模型"""
        logger.info("创建 ResNet-50 模型...")

        # 加载预训练模型
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1')

        # 修改最后一层
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数量: {trainable_params:,}")

        return model

    def train(
        self,
        epochs: int = 50,
        learning_rate: float = 0.001,
        img_size: int = 224,
        save_dir: Path = None,
    ):
        """训练模型"""
        if save_dir is None:
            save_dir = self.data_dir.parent / "models" / "classifier"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 准备数据
        train_loader, val_loader = self.prepare_data(img_size)

        # 创建模型
        model = self.create_model()

        # 优化器和调度器
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 训练循环
        best_acc = 0.0

        logger.info("=" * 60)
        logger.info("开始训练靶心分类模型...")
        logger.info("=" * 60)

        for epoch in range(1, epochs + 1):
            # 训练
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.1f}%'
                })

            train_loss = train_loss / train_total
            train_acc = 100. * train_correct / train_total

            # 验证
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_loss / val_total
            val_acc = 100. * val_correct / val_total

            # 更新学习率
            scheduler.step()

            # 打印结果
            logger.info(
                f'Epoch {epoch}/{epochs}: '
                f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | '
                f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}% | '
                f'LR: {scheduler.get_last_lr()[0]:.6f}'
            )

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classes': self.classes,
                    'acc': val_acc,
                }, save_dir / 'best_classifier.pt')
                logger.info(f'✅ 保存最佳模型 (准确率: {val_acc:.1f}%)')

        # 保存最终模型
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'classes': self.classes,
            'acc': val_acc,
        }, save_dir / 'final_classifier.pt')

        logger.info("=" * 60)
        logger.info(f"训练完成! 最佳验证准确率: {best_acc:.1f}%")
        logger.info(f"模型保存在: {save_dir}")
        logger.info("=" * 60)

        return model


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "yolo_dataset"

    # 创建训练器
    trainer = BullseyeClassifier(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4,
    )

    # 开始训练
    trainer.train(
        epochs=50,
        learning_rate=0.001,
        img_size=224,
        save_dir=project_root / "models" / "classifier",
    )


if __name__ == "__main__":
    main()
