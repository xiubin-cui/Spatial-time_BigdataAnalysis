import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from pathlib import Path
import os
import shutil
import logging
from typing import List, Tuple
import argparse
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """训练配置类，集中管理超参数"""
    data_dir: str = "./data"
    target_root_dir: str = "./trash"
    model_save_path: str = "./models/resnet18_{lr}_source.pth"
    num_classes: int = 4
    learning_rate: float = 0.01
    num_epochs: int = 25
    batch_size: int = 32  # 减小批次大小
    num_workers: int = 0  # 禁用多进程加载
    pretrained: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args() -> Config:
    """解析命令行参数，覆盖默认配置"""
    parser = argparse.ArgumentParser(description="Train ResNet18 for image classification")
    parser.add_argument("--data-dir", type=str, default=Config.data_dir, help="Dataset root directory")
    parser.add_argument("--target-root-dir", type=str, default=Config.target_root_dir, help="Directory for error images")
    parser.add_argument("--num-classes", type=int, default=Config.num_classes, help="Number of classes")
    parser.add_argument("--lr", type=float, default=Config.learning_rate, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=Config.num_epochs, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=Config.num_workers, help="Number of data loader workers")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    args = parser.parse_args()
    return Config(
        data_dir=args.data_dir,
        target_root_dir=args.target_root_dir,
        model_save_path=Config.model_save_path.format(lr=args.lr),
        num_classes=args.num_classes,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrained=args.pretrained
    )

def setup_device() -> torch.device:
    """设置计算设备（GPU或CPU）"""
    device = torch.device(Config.device)
    logger.info(f"使用设备: {device}")
    return device

def create_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 减小分辨率
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        train_dataset = datasets.ImageFolder(root=f"{config.data_dir}/train", transform=transform)
        val_dataset = datasets.ImageFolder(root=f"{config.data_dir}/val", transform=transform)
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise ValueError(f"无法加载数据集: {e}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
        persistent_workers=False
    )
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    return train_loader, val_loader

def initialize_model(config: Config) -> nn.Module:
    """
    初始化 ResNet18 模型
    """
    try:
        model = models.resnet18(weights=None if not config.pretrained else "DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
        logger.info(f"初始化模型 ResNet18，类别数: {config.num_classes}, 预训练: {config.pretrained}")
        return model
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        raise RuntimeError(f"无法初始化模型: {e}")

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config
) -> List[str]:
    """
    训练模型并在每个 epoch 后进行验证
    """
    model.to(config.device)
    error_images = []
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_id, (inputs, labels) in enumerate(train_loader):
            try:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (batch_id + 1) % 50 == 0:
                    logger.info(
                        f"训练 Epoch: {epoch} [{(batch_id + 1) * len(inputs)}/{len(train_loader.dataset)} "
                        f"({100. * (batch_id + 1) / total_batches:.0f}%)] 损失: {loss.item():.6f}"
                    )
            except RuntimeError as e:
                logger.error(f"训练批次 {batch_id} 出错: {e}")
                continue

        epoch_loss = running_loss / total_batches
        logger.info(f"Epoch {epoch}/{config.num_epochs - 1}, 平均损失: {epoch_loss:.4f}")
        error_images = test_model(model, val_loader, criterion, config.device)

    return error_images

def test_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> List[str]:
    """
    验证模型性能并记录预测错误的图像
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    error_images = []

    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(val_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)

                for i in range(len(preds)):
                    if preds[i] != labels[i]:
                        image_idx = i + batch_id * val_loader.batch_size
                        if image_idx < len(val_loader.dataset.imgs):
                            error_images.append(val_loader.dataset.imgs[image_idx][0])
            except RuntimeError as e:
                logger.error(f"验证批次 {batch_id} 出错: {e}")
                continue

    accuracy = correct / total if total > 0 else 0
    avg_loss = test_loss / len(val_loader) if len(val_loader) > 0 else 0
    logger.info(
        f"验证集: 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total} ({100 * accuracy:.0f}%)"
    )
    return error_images

def move_error_images(error_images: List[str], target_root_dir: str) -> None:
    """
    移动预测错误的图像到目标目录
    """
    target_root = Path(target_root_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    for image_path in error_images:
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.warning(f"图像 {image_path} 不存在，跳过")
                continue
            category = image_path.parent.name
            file_name = image_path.name
            target_dir = target_root / category
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(target_dir / file_name))
            logger.info(f"移动图像 {image_path} 到 {target_dir / file_name}")
        except (shutil.Error, OSError) as e:
            logger.error(f"移动图像 {image_path} 失败: {e}")
    logger.info("所有错误预测图像已处理")

def main():
    """主函数，执行模型训练、验证和错误图像处理"""
    try:
        config = parse_args()
        logger.info(f"训练配置: {config}")

        device = setup_device()
        train_loader, val_loader = create_data_loaders(config)
        model = initialize_model(config)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

        error_images = train_model(model, train_loader, val_loader, criterion, optimizer, config)

        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        torch.save(model.state_dict(), config.model_save_path)
        logger.info(f"模型已保存至 {config.model_save_path}")

        if error_images:
            move_error_images(error_images, config.target_root_dir)
        else:
            logger.info("没有发现预测错误的图像")

    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理 CUDA 缓存")

if __name__ == "__main__":
    main()