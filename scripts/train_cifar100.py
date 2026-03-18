#!/usr/bin/env python3
"""
CIFAR-100 分类模型训练脚本
使用 ResNet-18 + 迁移学习
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CIFAR100Trainer:
    """CIFAR-100 分类模型训练器"""

    def __init__(
        self,
        data_root: Path = Path("./data/cifar100"),
        batch_size: int = 128,
        num_workers: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化训练器

        Args:
            data_root: 数据集根目录
            batch_size: 批次大小
            num_workers: 数据加载线程数
            device: 训练设备
        """
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)

        logger.info(f"使用设备: {self.device}")

    def get_data_transforms(self):
        """
        获取数据增强策略

        Returns:
            (train_transform, test_transform)
        """
        # CIFAR-100 的均值和标准差
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        # 强力增强策略
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),  # Cutout
        ])

        # 测试时只做标准化
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        return train_transform, test_transform

    def prepare_data(self):
        """
        准备 CIFAR-100 数据集

        Returns:
            (train_loader, test_loader, classes)
        """
        train_transform, test_transform = self.get_data_transforms()

        # 下载训练集
        logger.info("准备 CIFAR-100 训练集...")
        train_set = torchvision.datasets.CIFAR100(
            root=self.data_root,
            train=True,
            download=True,
            transform=train_transform
        )

        # 下载测试集
        logger.info("准备 CIFAR-100 测试集...")
        test_set = torchvision.datasets.CIFAR100(
            root=self.data_root,
            train=False,
            download=True,
            transform=test_transform
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        classes = train_set.classes

        logger.info(f"训练集: {len(train_set)} 张图片")
        logger.info(f"测试集: {len(test_set)} 张图片")
        logger.info(f"类别数: {len(classes)}")

        return train_loader, test_loader, classes

    def create_model(self, num_classes: int = 100):
        """
        创建 ResNet-18 模型（迁移学习）

        Args:
            num_classes: 类别数量

        Returns:
            模型和优化器
        """
        logger.info("创建 ResNet-18 模型...")

        # 加载预训练模型
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

        # 修改最后一层以适应 CIFAR-100
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        # 如果有多个 GPU，使用 DataParallel
        if torch.cuda.device_count() > 1:
            logger.info(f"使用 {torch.cuda.device_count()} 个 GPU")
            model = nn.DataParallel(model)

        model = model.to(self.device)

        logger.info("模型创建成功")
        logger.info(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        return model

    def create_optimizer(self, model, learning_rate: float = 0.001):
        """
        创建优化器和学习率调度器

        Args:
            model: 模型
            learning_rate: 学习率

        Returns:
            (optimizer, scheduler)
        """
        # 只优化最后一层和部分骨干网络
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )

        # 余弦退火学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=30  # 30 轮后学习率降到最低
        )

        return optimizer, scheduler

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """
        训练一个 epoch

        Args:
            model: 模型
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            epoch: 当前轮数

        Returns:
            平均损失
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def evaluate(self, model, test_loader, criterion):
        """
        评估模型

        Args:
            model: 模型
            test_loader: 测试数据加载器
            criterion: 损失函数

        Returns:
            (损失, 准确率)
        """
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss = test_loss / total
        test_acc = 100. * correct / total

        return test_loss, test_acc

    def train(
        self,
        epochs: int = 30,
        learning_rate: float = 0.001,
        save_dir: Path = Path("./runs/classify"),
    ):
        """
        训练模型

        Args:
            epochs: 训练轮数
            learning_rate: 学习率
            save_dir: 模型保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 准备数据
        train_loader, test_loader, classes = self.prepare_data()

        # 创建模型
        model = self.create_model(num_classes=len(classes))

        # 创建优化器
        optimizer, scheduler = self.create_optimizer(model, learning_rate)

        # 损失函数
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_acc = 0
        logger.info("=" * 60)
        logger.info("开始训练...")
        logger.info("=" * 60)

        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )

            # 评估
            test_loss, test_acc = self.evaluate(model, test_loader, criterion)

            # 更新学习率
            scheduler.step()

            # 打印结果
            logger.info(
                f'Epoch {epoch}/{epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%'
            )

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': test_acc,
                }, save_dir / 'best.pt')
                logger.info(f'✅ 保存最佳模型 (准确率: {test_acc:.2f}%)')

            # 定期保存检查点
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': test_acc,
                }, save_dir / f'checkpoint_epoch_{epoch}.pt')

        logger.info("=" * 60)
        logger.info(f"训练完成! 最佳测试准确率: {best_acc:.2f}%")
        logger.info(f"模型保存在: {save_dir}")
        logger.info("=" * 60)


def main():
    """主函数"""
    # 配置路径
    project_root = Path(__file__).parent.parent
    save_dir = project_root / "experiments" / "runs" / "cifar100_resnet18"

    # 创建训练器
    trainer = CIFAR100Trainer(
        batch_size=128,
        num_workers=8,
    )

    # 开始训练
    trainer.train(
        epochs=30,              # 30 轮训练
        learning_rate=0.001,    # 初始学习率
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
