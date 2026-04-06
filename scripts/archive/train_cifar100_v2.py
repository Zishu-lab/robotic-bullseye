#!/usr/bin/env python3
"""
CIFAR-100 改进版V2训练脚本
修复问题:
- EMA评估逻辑修复
- 适度数据增强
- ResNet-50 (更适合从头训练)
- 更稳定的学习率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
import copy


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mixup_data(x, y, alpha=0.4, device='cuda'):
    """MixUp数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, beta=1.0, device='cuda'):
    """CutMix数据增强"""
    lam = np.random.beta(beta, beta)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x_copy = x.clone()
    x_copy[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return x_copy, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing损失函数"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        with torch.no_grad():
            target_one_hot = torch.zeros_like(pred)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
            target_one_hot = target_one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        return torch.mean(torch.sum(-target_one_hot * F.log_softmax(pred, dim=1), dim=1))


class ResNet50Improved(nn.Module):
    """改进的ResNet-50模型"""
    def __init__(self, num_classes=100, dropout_rate=0.2):
        super().__init__()
        # 从头训练，不使用预训练权重
        self.backbone = torchvision.models.resnet50(weights=None)

        num_features = self.backbone.fc.in_features

        # 改进的分类头
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class CIFAR100TrainerV2:
    """CIFAR-100 改进版V2训练器"""

    def __init__(
        self,
        data_root: Path = Path("./data/cifar100"),
        batch_size: int = 128,
        num_workers: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_mixup: bool = True,
        use_cutmix: bool = True,
        mixup_alpha: float = 0.4,
        cutmix_beta: float = 1.0,
        label_smoothing: float = 0.1,
        dropout_rate: float = 0.2,
    ):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_beta = cutmix_beta
        self.label_smoothing = label_smoothing
        self.dropout_rate = dropout_rate

        logger.info(f"使用设备: {self.device}")
        logger.info(f"MixUp: {use_mixup} (alpha={mixup_alpha}), CutMix: {use_cutmix}")
        logger.info(f"Label Smoothing: {label_smoothing}, Dropout: {dropout_rate}")

    def get_data_transforms(self):
        """获取数据增强策略"""
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        # 适度的训练增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        return train_transform, test_transform

    def prepare_data(self):
        """准备数据集"""
        train_transform, test_transform = self.get_data_transforms()

        train_set = torchvision.datasets.CIFAR100(
            root=self.data_root, train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR100(
            root=self.data_root, train=False, download=True, transform=test_transform
        )

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        logger.info(f"训练集: {len(train_set)} 张图片")
        logger.info(f"测试集: {len(test_set)} 张图片")

        return train_loader, test_loader, train_set.classes

    def create_model(self, num_classes=100):
        """创建改进的ResNet-50模型"""
        logger.info("创建 ResNet-50 改进模型...")

        model = ResNet50Improved(
            num_classes=num_classes,
            dropout_rate=self.dropout_rate
        )

        if torch.cuda.device_count() > 1:
            logger.info(f"使用 {torch.cuda.device_count()} 个 GPU")
            model = nn.DataParallel(model)

        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数量: {trainable_params:,}")

        return model

    def create_optimizer(self, model, learning_rate=0.1, weight_decay=5e-4, total_epochs=200):
        """创建SGD优化器和学习率调度器"""
        # SGD with momentum - 对CIFAR更稳定
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )

        # Cosine Annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=1e-5
        )

        return optimizer, scheduler

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 随机选择数据增强方式 (50%概率使用mixup/cutmix)
            use_aug = np.random.rand() < 0.5

            if use_aug and self.use_mixup and self.use_cutmix:
                if np.random.rand() < 0.5:
                    images, labels_a, labels_b, lam = mixup_data(
                        images, labels, self.mixup_alpha, self.device
                    )
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    images, labels_a, labels_b, lam = cutmix_data(
                        images, labels, self.cutmix_beta, self.device
                    )
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def evaluate(self, model, test_loader, criterion):
        """评估模型"""
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
        epochs: int = 200,
        learning_rate: float = 0.1,
        weight_decay: float = 5e-4,
        save_dir: Path = Path("./runs/classify"),
        resume_from: Path = None,
    ):
        """训练模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 准备数据
        train_loader, test_loader, classes = self.prepare_data()

        # 创建模型
        model = self.create_model(num_classes=len(classes))

        # 创建优化器
        optimizer, scheduler = self.create_optimizer(
            model, learning_rate, weight_decay, total_epochs=epochs
        )

        # 损失函数 (测试时不使用label smoothing)
        criterion = LabelSmoothingCrossEntropy(smoothing=self.label_smoothing)
        criterion_eval = nn.CrossEntropyLoss()

        # 恢复训练
        start_epoch = 0
        best_acc = 0
        if resume_from and Path(resume_from).exists():
            logger.info(f"从检查点恢复: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint.get('acc', 0)
            for _ in range(start_epoch):
                scheduler.step()
            logger.info(f"从 Epoch {start_epoch} 继续训练, 之前最佳准确率: {best_acc:.2f}%")

        # 训练循环
        logger.info("=" * 60)
        logger.info("开始改进版V2训练 (ResNet-50 + SGD + Cosine LR)")
        logger.info("=" * 60)

        for epoch in range(start_epoch + 1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )

            # 评估 (使用普通CrossEntropy)
            test_loss, test_acc = self.evaluate(model, test_loader, criterion_eval)

            # 更新学习率
            scheduler.step()

            # 打印结果
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f'Epoch {epoch}/{epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | '
                f'LR: {current_lr:.6f}'
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
            if epoch % 25 == 0:
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

        return best_acc


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='CIFAR-100 改进版V2分类模型训练')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数 (默认: 200)')
    parser.add_argument('--lr', type=float, default=0.1, help='学习率 (默认: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='权重衰减 (默认: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小 (默认: 128)')
    parser.add_argument('--no-mixup', action='store_true', help='禁用MixUp')
    parser.add_argument('--no-cutmix', action='store_true', help='禁用CutMix')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    args = parser.parse_args()

    # 配置路径
    project_root = Path(__file__).parent.parent
    save_dir = project_root / "experiments" / "runs" / "cifar100_v2"

    # 创建训练器
    trainer = CIFAR100TrainerV2(
        batch_size=args.batch_size,
        num_workers=8,
        use_mixup=not args.no_mixup,
        use_cutmix=not args.no_cutmix,
        mixup_alpha=0.4,
        cutmix_beta=1.0,
        label_smoothing=0.1,
        dropout_rate=0.2,
    )

    # 开始训练
    trainer.train(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=save_dir,
        resume_from=Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
