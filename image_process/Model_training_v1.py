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

# --- 配置日志 ---
# 配置日志，包括控制台输出和文件输出，记录时间、级别和消息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
)
logger = logging.getLogger(__name__)

# --- 配置类 ---
@dataclass
class Config:
    """
    训练配置类，集中管理超参数和路径信息。
    使用 dataclass 简化类的定义，提供默认值。
    """
    data_dir: str = "./data"  # 数据集根目录
    target_root_dir: str = "./trash"  # 存储错误分类图像的目录
    model_save_path: str = "./resnet18_{lr}_source.pth"  # 模型保存路径，包含学习率作为动态部分
    num_classes: int = 4  # 分类任务的类别数量
    learning_rate: float = 0.01  # 初始学习率
    num_epochs: int = 25  # 训练的总 epoch 数量
    batch_size: int = 64  # 每个批次处理的图像数量
    num_workers: int = 4  # 数据加载的并行工作进程数
    pretrained: bool = False  # 是否使用预训练模型权重
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测并使用 GPU 或 CPU

# --- 参数解析 ---
def parse_args() -> Config:
    """
    解析命令行参数，允许用户覆盖 Config 类中的默认配置。
    使用 argparse 模块定义命令行接口。
    """
    parser = argparse.ArgumentParser(description="训练 ResNet18 进行图像分类")
    parser.add_argument("--data-dir", type=str, default=Config.data_dir,
                        help="数据集的根目录 (默认: ./data)")
    parser.add_argument("--target-root-dir", type=str, default=Config.target_root_dir,
                        help="用于存放错误分类图像的目录 (默认: ./trash)")
    parser.add_argument("--num-classes", type=int, default=Config.num_classes,
                        help="分类任务的类别数量 (默认: 4)")
    parser.add_argument("--lr", type=float, default=Config.learning_rate,
                        help="学习率 (默认: 0.01)")
    parser.add_argument("--epochs", type=int, default=Config.num_epochs,
                        help="训练的 epoch 数量 (默认: 25)")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size,
                        help="每个批次的图像数量 (默认: 64)")
    parser.add_argument("--pretrained", action="store_true",
                        help="使用 ImageNet 预训练权重初始化模型")
    args = parser.parse_args()

    # 根据解析到的参数创建并返回 Config 实例
    return Config(
        data_dir=args.data_dir,
        target_root_dir=args.target_root_dir,
        # 动态格式化模型保存路径，将学习率嵌入文件名
        model_save_path=Config.model_save_path.format(lr=args.lr),
        num_classes=args.num_classes,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        pretrained=args.pretrained
    )

# --- 设备设置 ---
def setup_device() -> torch.device:
    """
    设置并返回 PyTorch 使用的计算设备（GPU 或 CPU）。
    """
    device = torch.device(Config.device)
    logger.info(f"正在使用设备: {device}")
    return device

# --- 数据加载器创建 ---
def create_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器。
    定义图像预处理步骤，包括缩放、转换为 Tensor 和标准化。
    使用 ImageFolder 自动从目录结构加载图像。
    """
    # 定义图像预处理转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 将图像大小调整为 128x128 像素
        transforms.ToTensor(),          # 将 PIL 图像或 numpy.ndarray 转换为 Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 标准化图像像素值
    ])

    try:
        # 加载训练集和验证集，ImageFolder 要求数据集按类别组织在子目录中
        train_dataset = datasets.ImageFolder(root=f"{config.data_dir}/train", transform=transform)
        val_dataset = datasets.ImageFolder(root=f"{config.data_dir}/val", transform=transform)
    except Exception as e:
        logger.error(f"加载数据集失败，请检查 '{config.data_dir}/train' 和 '{config.data_dir}/val' 路径及内容: {e}")
        raise ValueError(f"无法加载数据集: {e}")

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 训练集需要打乱顺序
        num_workers=config.num_workers, # 多进程加载数据，提高效率
        pin_memory=True if config.device == "cuda" else False # 如果使用 GPU，将数据加载到 CUDA 内存中
    )
    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False, # 验证集不需要打乱顺序
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    logger.info(f"训练集大小: {len(train_dataset)} 张图像, 验证集大小: {len(val_dataset)} 张图像")
    return train_loader, val_loader

# --- 模型初始化 ---
def initialize_model(config: Config) -> nn.Module:
    """
    初始化 ResNet18 模型。
    根据 `config.pretrained` 决定是否加载预训练权重。
    将 ResNet 的最后一层全连接层替换为适合当前分类任务的输出层。
    """
    try:
        # 根据 torchvision 版本调整加载预训练模型的方式
        # 对于旧版本，使用 pretrained=True/False
        # 对于新版本 (>=0.13.0)，使用 weights="DEFAULT" 或 weights=None
        # 鉴于报错信息，判断为旧版本，使用 pretrained=True
        model = models.resnet18(pretrained=config.pretrained) # 修正: 使用 pretrained 参数
        # 替换最后一层全连接层，以适应新的类别数
        num_ftrs = model.fc.in_features # 获取全连接层的输入特征数
        model.fc = nn.Linear(num_ftrs, config.num_classes) # 重新定义全连接层
        logger.info(f"初始化 ResNet18 模型，类别数: {config.num_classes}, 预训练: {config.pretrained}")
        return model
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        raise RuntimeError(f"无法初始化模型: {e}")

# --- 模型训练 ---
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config
) -> List[str]:
    """
    训练模型并在每个 epoch 后进行验证。
    """
    model.to(config.device) # 将模型移动到指定设备

    # 记录每个 epoch 验证时预测错误的图像路径
    all_error_images_paths = []

    for epoch in range(config.num_epochs):
        model.train() # 设置模型为训练模式
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_id, (inputs, labels) in enumerate(train_loader):
            try:
                inputs, labels = inputs.to(config.device), labels.to(config.device) # 将数据移动到指定设备
                optimizer.zero_grad() # 清零梯度
                outputs = model(inputs) # 前向传播
                loss = criterion(outputs, labels) # 计算损失
                loss.backward() # 反向传播
                optimizer.step() # 更新模型参数
                running_loss += loss.item()

                # 每 50 个批次打印一次训练信息，提供实时反馈
                if (batch_id + 1) % 50 == 0:
                    logger.info(
                        f"Epoch: {epoch+1}/{config.num_epochs} | "
                        f"Batch: {batch_id+1}/{total_batches} | "
                        f"Loss: {loss.item():.6f}"
                    )
            except RuntimeError as e:
                logger.error(f"训练批次 {batch_id+1} 发生运行时错误: {e}", exc_info=True)
                # 记录 problematic batch 可以考虑跳过或更复杂的错误处理
                continue
            except Exception as e:
                logger.error(f"训练批次 {batch_id+1} 发生未知错误: {e}", exc_info=True)
                continue


        epoch_loss = running_loss / total_batches
        logger.info(f"--- Epoch {epoch+1}/{config.num_epochs} 训练完成, 平均损失: {epoch_loss:.4f} ---")

        # 每个 epoch 结束后进行验证
        epoch_error_images = test_model(model, val_loader, criterion, config.device)
        if epoch_error_images:
            all_error_images_paths.extend(epoch_error_images) # 收集所有 epoch 中预测错误的图像

    return all_error_images_paths

# --- 模型测试/验证 ---
def test_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> List[str]:
    """
    验证模型在验证集上的性能，并记录预测错误的图像路径。
    """
    model.eval() # 设置模型为评估模式，禁用 dropout 和 BatchNorm 等
    correct = 0
    total = 0
    test_loss = 0.0
    error_images_current_epoch = []

    with torch.no_grad(): # 在评估模式下，不计算梯度，节省内存和计算
        for batch_id, (inputs, labels) in enumerate(val_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1) # 获取预测类别
                correct += torch.sum(preds == labels).item() # 统计正确预测的数量
                total += labels.size(0) # 统计总样本数

                # 记录预测错误的图像路径
                for i in range(len(preds)):
                    if preds[i] != labels[i]:
                        # 获取原始图像路径。val_loader.dataset 是 ImageFolder 类型，其 imgs 属性包含 (image_path, class_id) 元组
                        # 这里的 image_idx 需要考虑到当前 batch 在整个数据集中的偏移量
                        global_image_idx = batch_id * val_loader.batch_size + i
                        if global_image_idx < len(val_loader.dataset.imgs): # 确保索引不越界
                            error_images_current_epoch.append(val_loader.dataset.imgs[global_image_idx][0])
            except RuntimeError as e:
                logger.error(f"验证批次 {batch_id+1} 发生运行时错误: {e}", exc_info=True)
                continue
            except Exception as e:
                logger.error(f"验证批次 {batch_id+1} 发生未知错误: {e}", exc_info=True)
                continue

    accuracy = correct / total if total > 0 else 0
    avg_loss = test_loss / len(val_loader) if len(val_loader) > 0 else 0
    logger.info(
        f"验证结果: 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total} ({100 * accuracy:.2f}%)"
    )
    return error_images_current_epoch

# --- 移动错误图像 ---
def move_error_images(error_images_paths: List[str], target_root_dir: str) -> None:
    """
    将预测错误的图像移动到目标目录，并按照原始类别创建子目录。
    """
    # 使用 pathlib 模块处理路径，更面向对象且跨平台
    target_root = Path(target_root_dir)
    target_root.mkdir(parents=True, exist_ok=True) # 创建目标根目录，如果不存在

    # 使用 set 来存储已经处理过的图像路径，避免重复移动，提高效率和健壮性
    processed_images = set()

    for image_path_str in error_images_paths:
        if image_path_str in processed_images:
            continue # 如果该图像已被处理，则跳过

        try:
            image_path = Path(image_path_str)
            if not image_path.exists():
                logger.warning(f"图像 '{image_path}' 不存在，跳过移动。")
                continue

            # 从图像路径中提取原始类别名称（父目录名）和文件名
            category = image_path.parent.name
            file_name = image_path.name

            # 构建目标子目录和目标文件路径
            target_dir = target_root / category
            target_dir.mkdir(parents=True, exist_ok=True) # 为当前类别创建子目录

            destination_path = target_dir / file_name
            # 如果目标文件已存在，为了避免错误，可以考虑重命名或覆盖
            # 这里选择直接移动，如果目标存在且是文件，shutil.move 会覆盖
            shutil.move(str(image_path), str(destination_path))
            logger.info(f"已移动图像: '{image_path}' 到 '{destination_path}'")
            processed_images.add(image_path_str) # 标记为已处理

        except FileNotFoundError:
            logger.error(f"移动图像 '{image_path_str}' 失败: 源文件不存在。")
        except shutil.Error as e:
            logger.error(f"移动图像 '{image_path_str}' 失败 (shutil 错误): {e}", exc_info=True)
        except OSError as e:
            logger.error(f"移动图像 '{image_path_str}' 失败 (操作系统错误): {e}", exc_info=True)
        except Exception as e:
            logger.error(f"移动图像 '{image_path_str}' 发生未知错误: {e}", exc_info=True)

    logger.info("所有预测错误的图像处理完成。")


# --- 主函数 ---
def main():
    """
    主函数，协调模型训练、验证、保存和错误图像处理的整个流程。
    """
    try:
        # 1. 加载配置
        config = parse_args()
        logger.info(f"当前训练配置: {config}")

        # 2. 设置设备
        device = setup_device()

        # 3. 创建数据加载器
        train_loader, val_loader = create_data_loaders(config)

        # 4. 初始化模型
        model = initialize_model(config)

        # 5. 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss() # 交叉熵损失函数适用于多分类问题
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9) # SGD 优化器

        # 6. 训练和验证模型
        # train_model 返回在验证阶段所有 epoch 中所有预测错误的图像路径列表
        error_images_collected = train_model(model, train_loader, val_loader, criterion, optimizer, config)

        # 7. 保存模型
        # 确保模型保存路径的父目录存在
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        torch.save(model.state_dict(), config.model_save_path) # 只保存模型参数
        logger.info(f"模型参数已保存至: {config.model_save_path}")

        # 8. 移动错误预测的图像
        # 由于 test_model 在每个 epoch 都会运行并返回错误图像，
        # 我们收集了所有 epoch 的错误图像。这里使用 set 来去重。
        unique_error_images = list(set(error_images_collected))
        if unique_error_images:
            logger.info(f"共发现 {len(unique_error_images)} 张预测错误的图像，正在移动...")
            move_error_images(unique_error_images, config.target_root_dir)
        else:
            logger.info("本轮训练和验证中，没有发现预测错误的图像。")

    except Exception as e:
        logger.error(f"程序执行过程中发生严重错误: {e}", exc_info=True)
        # 抛出异常以便上层调用者可以捕获
        raise
    finally:
        # 确保在程序结束时清理 CUDA 缓存，无论是否发生错误
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理 CUDA 缓存。")

# --- 程序入口点 ---
if __name__ == "__main__":
    main()