#!/usr/bin/env python3
"""
CIFAR-100 全面优化版训练脚本
目标准确率: 80%+

优化策略:
- EfficientNet-B3 (更先进的架构)
- EMA (指数移动平均)
- Warmup + Cosine Annealing 学习率调度
- RandAugment + MixUp + CutMix 数据增强
- Label Smoothing
- TTA (测试时增强)
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
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============== EMA (指数移动平均) ==============
class EMA:
    """指数移动平均，用于稳定训练"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============== 数据增强函数 ==============
def mixup_data(x, y, alpha=0.4, device='cuda'):
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
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============== Label Smoothing ==============
class LabelSmoothingCrossEntropy(nn.Module):
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


# ============== Warmup + Cosine 学习率调度 ==============
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_lr=1e-6, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ============== EfficientNet 适配 CIFAR-100 ==============
class EfficientNetForCIFAR(nn.Module):
    """EfficientNet 适配 CIFAR-100 (32x32 小图像)"""
    def __init__(self, model_name='efficientnet_b0', num_classes=100, dropout_rate=0.2):
        super().__init__()

        # 加载预训练 EfficientNet
        self.model = torchvision.models.efficientnet_b3(weights='IMAGENET1K_V1')

        # 修改第一层以适应 32x32 小图像 (减小 stride)
        original_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            3, original_conv.out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # 修改分类头
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ============== TTA (测试时增强) ==============
def tta_predict(model, images, num_augments=3):
    """测试时增强预测"""
    model.eval()
    all_outputs = []

    with torch.no_grad():
        # 原始预测
        outputs = model(images)
        all_outputs.append(outputs)

        # 水平翻转
        flipped = torch.flip(images, dims=[3])
        outputs = model(flipped)
        all_outputs.append(outputs)

        # 多尺度
        if num_augments > 2:
            scaled = F.interpolate(images, size=40, mode='bilinear', align_corners=False)
            scaled = F.interpolate(scaled, size=32, mode='bilinear', align_corners=False)
            outputs = model(scaled)
            all_outputs.append(outputs)

    return torch.stack(all_outputs).mean(dim=0)


# ============== 训练器 ==============
class CIFAR100OptimizedTrainer:
    """CIFAR-100 全面优化训练器"""

    def __init__(
        self,
        data_root: Path = Path("./data/cifar100"),
        batch_size: int = 64,
        num_workers: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_mixup: bool = True,
        use_cutmix: bool = True,
        mixup_alpha: float = 0.4,
        cutmix_beta: float = 1.0,
        label_smoothing: float = 0.1,
        dropout_rate: float = 0.2,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        use_tta: bool = True,
        tta_augments: int = 3,
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
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_tta = use_tta
        self.tta_augments = tta_augments

        logger.info("=" * 60)
        logger.info("CIFAR-100 全面优化训练器")
        logger.info("=" * 60)
        logger.info(f"设备: {self.device}")
        logger.info(f"MixUp: {use_mixup}, CutMix: {use_cutmix}")
        logger.info(f"Label Smoothing: {label_smoothing}, Dropout: {dropout_rate}")
        logger.info(f"EMA: {use_ema}, TTA: {use_tta}")

    def get_data_transforms(self):
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        # RandAugment 数据增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=10),
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

        logger.info(f"训练集: {len(train_set)} 张, 测试集: {len(test_set)} 张")
        return train_loader, test_loader, train_set.classes

    def create_model(self, num_classes=100):
        logger.info("创建 EfficientNet-B3 模型...")
        model = EfficientNetForCIFAR(
            model_name='efficientnet_b3',
            num_classes=num_classes,
            dropout_rate=self.dropout_rate
        )

        if torch.cuda.device_count() > 1:
            logger.info(f"使用 {torch.cuda.device_count()} 个 GPU")
            model = nn.DataParallel(model)

        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"总参数量: {total_params:,}")
        return model

    def create_optimizer(self, model, learning_rate=0.001, weight_decay=0.05, warmup_epochs=10, total_epochs=300):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=warmup_epochs, total_epochs=total_epochs,
            warmup_lr=1e-6, min_lr=1e-6
        )
        return optimizer, scheduler

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            use_aug = np.random.rand() < 0.5

            if use_aug and self.use_mixup and self.use_cutmix:
                if np.random.rand() < 0.5:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha, self.device)
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    images, labels_a, labels_b, lam = cutmix_data(images, labels, self.cutmix_beta, self.device)
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

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        return running_loss / total, 100. * correct / total

    def evaluate(self, model, test_loader, criterion, use_tta=False):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing', leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                if use_tta:
                    outputs = tta_predict(model, images, self.tta_augments)
                else:
                    outputs = model(images)

                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return test_loss / total, 100. * correct / total

    def train(
        self,
        epochs: int = 300,
        learning_rate: float = 0.001,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        save_dir: Path = Path("./runs/classify"),
        resume_from: Path = None,
    ):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        train_loader, test_loader, classes = self.prepare_data()
        model = self.create_model(num_classes=len(classes))
        optimizer, scheduler = self.create_optimizer(model, learning_rate, weight_decay, warmup_epochs, epochs)

        criterion = LabelSmoothingCrossEntropy(smoothing=self.label_smoothing)
        criterion_eval = nn.CrossEntropyLoss()

        # EMA
        ema = EMA(model, decay=self.ema_decay) if self.use_ema else None

        # 恢复训练
        start_epoch = 0
        best_acc = 0
        best_ema_acc = 0

        if resume_from and Path(resume_from).exists():
            logger.info(f"从检查点恢复: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint.get('acc', 0)
            for _ in range(start_epoch):
                scheduler.step()
            logger.info(f"从 Epoch {start_epoch} 继续训练, 之前最佳: {best_acc:.2f}%")

        logger.info("=" * 60)
        logger.info(f"开始训练 (EfficientNet-B3 + EMA + TTA)")
        logger.info(f"总轮数: {epochs}, Warmup: {warmup_epochs}, LR: {learning_rate}")
        logger.info("=" * 60)

        for epoch in range(start_epoch + 1, epochs + 1):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            current_lr = scheduler.step()

            if ema:
                ema.update()

            test_loss, test_acc = self.evaluate(model, test_loader, criterion_eval, use_tta=False)

            ema_acc = 0
            if ema:
                ema.apply_shadow()
                _, ema_acc = self.evaluate(model, test_loader, criterion_eval, use_tta=True)
                ema.restore()

            log_msg = f'Epoch {epoch}/{epochs}: Train {train_acc:.2f}% | Test {test_acc:.2f}%'
            if ema:
                log_msg += f' | EMA+TTA {ema_acc:.2f}%'
            log_msg += f' | LR {current_lr:.6f}'
            logger.info(log_msg)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': test_acc,
                    'ema_acc': ema_acc,
                    'classes': classes,
                }, save_dir / 'best.pt')
                logger.info(f'✅ 保存最佳模型 ({test_acc:.2f}%)')

            if ema and ema_acc > best_ema_acc:
                best_ema_acc = ema_acc
                ema.apply_shadow()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'acc': ema_acc,
                    'classes': classes,
                    'use_tta': True,
                }, save_dir / 'best_ema.pt')
                ema.restore()
                logger.info(f'✅ 保存最佳EMA模型 ({ema_acc:.2f}%)')

            if epoch % 25 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': test_acc,
                }, save_dir / f'checkpoint_epoch_{epoch}.pt')

        logger.info("=" * 60)
        logger.info(f"训练完成! 最佳: {best_acc:.2f}%")
        if ema:
            logger.info(f"最佳EMA+TTA: {best_ema_acc:.2f}%")
        logger.info(f"模型保存: {save_dir}")
        logger.info("=" * 60)

        return best_acc, best_ema_acc


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CIFAR-100 全面优化训练')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--no-mixup', action='store_true')
    parser.add_argument('--no-cutmix', action='store_true')
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--no-tta', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    save_dir = project_root / "experiments" / "runs" / "cifar100_optimized"

    trainer = CIFAR100OptimizedTrainer(
        batch_size=args.batch_size,
        num_workers=8,
        use_mixup=not args.no_mixup,
        use_cutmix=not args.no_cutmix,
        label_smoothing=0.1,
        dropout_rate=0.2,
        use_ema=not args.no_ema,
        use_tta=not args.no_tta,
    )

    trainer.train(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup,
        save_dir=save_dir,
        resume_from=Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
