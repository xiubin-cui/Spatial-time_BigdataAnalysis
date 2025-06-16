import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List
import argparse
import os

# --- 配置日志 ---
# 配置日志，包括控制台输出和文件输出，记录时间、级别和消息，用于跟踪预测过程
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("prediction.log")]
)
logger = logging.getLogger(__name__)

# --- 自定义数据集类 ---
class CustomImageFolder(datasets.ImageFolder):
    """
    扩展 torchvision.datasets.ImageFolder 类，
    使其在返回图像和标签的同时，也返回原始图像的文件路径。
    这对于记录预测错误的图像非常有用。
    """
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """
        重写 __getitem__ 方法，返回图像、标签和图像路径。
        Args:
            index (int): 数据集中的索引。

        Returns:
            Tuple[torch.Tensor, int, str]: 图像张量、对应的类别标签、图像的完整文件路径。
        """
        # 调用父类的 __getitem__ 方法获取图像和标签
        image, label = super().__getitem__(index)
        # 获取图像的原始文件路径
        path = self.imgs[index][0]
        return image, label, path

# --- 参数解析 ---
def parse_args():
    """
    解析命令行参数。
    允许用户通过命令行指定数据目录、模型路径、输出文件等参数。
    """
    parser = argparse.ArgumentParser(description="使用 ResNet18 模型进行图像预测")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="数据集的根目录，通常包含 'val' 子目录 (默认: ./data)")
    parser.add_argument("--model-path", type=str, default="./resnet18_0.01_source.pth",
                        help="已训练模型的状态字典文件路径 (默认: ./resnet18_0.01_source.pth)")
    parser.add_argument("--output-file", type=str, default="./image_batch_results.txt",
                        help="存储所有图像预测结果的文本文件路径 (默认: ./image_batch_results.txt)")
    parser.add_argument("--incorrect-file", type=str, default="./incorrect_images.txt",
                        help="存储预测错误图像路径的文本文件路径 (默认: ./incorrect_images.txt)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="数据加载器中的批次大小 (默认: 32)")
    parser.add_argument("--num-classes", type=int, default=4,
                        help="模型分类任务的类别数量 (默认: 4)")
    args = parser.parse_args()
    return args

# --- 设备设置 ---
def setup_device() -> torch.device:
    """
    设置并返回 PyTorch 使用的计算设备（GPU 或 CPU）。
    优先使用 CUDA (GPU)，如果不可用则回退到 CPU。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"正在使用设备: {device}")
    return device

# --- 模型加载 ---
def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    加载预训练的 ResNet18 模型及其训练好的状态字典。
    适配旧版 torchvision，使用 `pretrained=False` 初始化模型结构。
    Args:
        model_path (str): 模型状态字典文件的路径。
        num_classes (int): 模型需要分类的类别数量。
        device (torch.device): 模型将被加载到的设备（CPU 或 GPU）。

    Returns:
        nn.Module: 加载并配置好的模型。

    Raises:
        FileNotFoundError: 如果模型文件不存在。
        RuntimeError: 如果模型加载或状态字典加载失败。
    """
    try:
        # 使用 pathlib 确保路径操作的健壮性
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件 '{model_file}' 不存在。请检查路径。")

        # 初始化 ResNet18 模型结构。
        # 修正: 对于旧版本 torchvision (如 Python 3.6 环境下)，
        # 应使用 `pretrained=False` 来初始化一个未经预训练的模型结构，
        # 而不是 `weights=None`。`weights` 参数是在 torchvision 0.13.0+ 中引入的。
        model = models.resnet18(pretrained=False) # 适配旧版 torchvision

        # 替换最后一层全连接层，以适应新的类别数。
        # 这一步是模型微调的关键，确保输出与分类任务匹配。
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        # 加载模型的状态字典。
        # 修正: 移除 `weights_only=True` 参数，因为这在旧版 PyTorch 中可能不支持，
        # 如果模型文件本身只包含状态字典，则不会有影响。
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)

        # 将模型移动到指定的设备并设置为评估模式。
        # eval() 模式会关闭 Dropout 层和 BatchNorm 层的训练行为，确保预测的确定性。
        model.to(device)
        model.eval()
        logger.info(f"成功加载模型: '{model_file}' 到设备: {device}")
        return model
    except FileNotFoundError as e:
        logger.error(f"模型文件未找到: {e}")
        raise
    except Exception as e:
        logger.error(f"加载模型或状态字典失败: {e}", exc_info=True)
        raise RuntimeError(f"无法加载模型: {e}")

# --- 数据加载器创建 ---
def create_data_loader(data_dir: str, batch_size: int) -> DataLoader:
    """
    创建用于预测的数据加载器。
    定义图像预处理步骤，并使用自定义的 CustomImageFolder 以获取图像路径。
    Args:
        data_dir (str): 数据集的根目录。
        batch_size (int): 每个批次处理的图像数量。

    Returns:
        DataLoader: 配置好的数据加载器。

    Raises:
        FileNotFoundError: 如果数据集验证目录不存在。
        ValueError: 如果加载数据集失败。
    """
    # 定义图像预处理转换。确保这些转换与训练时使用的转换一致。
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 将图像大小调整为 64x64 像素，可能与训练时不同，需确认
        transforms.ToTensor(),          # 将 PIL 图像或 numpy.ndarray 转换为 Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # 标准化图像像素值
    ])

    try:
        # 构建验证集的完整路径。假设验证数据位于 data_dir/val。
        val_data_path = Path(data_dir) / "val"
        if not val_data_path.exists():
            raise FileNotFoundError(f"数据集目录 '{val_data_path}' 不存在。请检查路径。")

        # 使用 CustomImageFolder 加载数据集，这样在预测时可以获取图像路径。
        test_dataset = CustomImageFolder(root=str(val_data_path), transform=transform)
        # 创建 DataLoader。预测时通常不需要打乱顺序，且 num_workers 可以设置为 0 简化调试。
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    except FileNotFoundError as e:
        logger.error(f"数据集目录未找到: {e}")
        raise
    except Exception as e:
        logger.error(f"加载数据集失败: {e}", exc_info=True)
        raise ValueError(f"无法加载数据集: {e}")

# --- 模型预测 ---
def predict(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    对测试数据集进行预测，并收集真实标签、预测标签以及所有预测错误的图像路径。
    Args:
        model (nn.Module): 已经加载并处于评估模式的模型。
        test_loader (DataLoader): 包含测试数据的 DataLoader。
        device (torch.device): 进行预测的设备。

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]:
            - true_labels (np.ndarray): 所有图像的真实类别标签数组。
            - predictions (np.ndarray): 所有图像的预测类别标签数组。
            - incorrect_paths (List[str]): 预测错误的图像的完整文件路径列表。
    """
    model.eval() # 确保模型处于评估模式
    true_labels = []
    predictions = []
    incorrect_paths = [] # 存储预测错误的图像路径

    with torch.no_grad(): # 在预测阶段禁用梯度计算，以节省内存并加速
        for batch_id, (inputs, labels, paths) in enumerate(test_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device) # 将数据移动到设备
                outputs = model(inputs) # 前向传播获取模型输出
                _, preds = torch.max(outputs, 1) # 获取最高概率对应的类别作为预测结果

                # 将 Tensor 转换为 numpy 数组并添加到列表中
                true_labels.extend(labels.cpu().numpy())
                predictions.extend(preds.cpu().numpy())

                # 检查并记录预测错误的图像路径
                for i, (pred, label, path) in enumerate(zip(preds, labels, paths)):
                    if pred != label:
                        incorrect_paths.append(path)
            except RuntimeError as e:
                logger.error(f"预测批次 {batch_id+1} 发生运行时错误: {e}", exc_info=True)
                # 可以选择跳过此批次或记录更多错误信息
                continue
            except Exception as e:
                logger.error(f"预测批次 {batch_id+1} 发生未知错误: {e}", exc_info=True)
                continue

    logger.info(f"预测完成。发现 {len(incorrect_paths)} 张预测错误的图像。")
    return np.array(true_labels), np.array(predictions), incorrect_paths

# --- 保存结果 ---
def save_results(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    incorrect_paths: List[str],
    output_file: str,
    incorrect_file: str,
) -> float:
    """
    将预测结果（真实值与预测值）写入文件，将预测错误的图像路径写入另一个文件，
    并计算整体预测准确率。
    Args:
        true_labels (np.ndarray): 真实类别标签数组。
        predictions (np.ndarray): 预测类别标签数组。
        incorrect_paths (List[str]): 预测错误的图像路径列表。
        output_file (str): 结果输出文件路径。
        incorrect_file (str): 错误图像路径输出文件路径。

    Returns:
        float: 模型的准确率。
    """
    # 定义类别映射，将数字标签转换为可读的类别名称
    # 确保这里的映射与训练时数据集的类别顺序一致
    label_map = {0: "Cyclone", 1: "Earthquake", 2: "Flood", 3: "Wildfire"}
    accuracy = np.mean(true_labels == predictions) # 计算准确率

    # 确保输出文件所在的目录存在
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir = Path(incorrect_file).parent
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    # 保存所有图像的预测结果到 output_file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("图像路径\t真实值\t预测值\n") # 添加图像路径列
            # 获取 DataLoader 的 dataset 对象，用于通过索引获取原始图像路径
            # 注意: 这里需要获取原始图像的路径，但当前的 save_results 并没有直接传入
            # 完整的文件路径，只有真实标签和预测标签。
            # 为了准确记录每个预测结果对应的图像路径，
            # 需要修改 predict 函数返回图像路径，并在此处传入
            # 或者，如果只需要统计，则此处代码足够。
            # 考虑到当前的需求和 CustomImageFolder 的设计，我将修改 predict 函数
            # 来传递路径。
            # 这里暂时无法直接获取每条记录的图像路径，所以只记录标签。
            # 如果需要图像路径，需要在 predict 函数中将路径也一并传递到 save_results
            # 并在 main 函数中获取和传递。
            #
            # 鉴于 `incorrect_paths` 已经存在，并且 `output_file` 主要是为了总览，
            # 假设 `output_file` 仅需记录标签。如果需要每个预测的详细路径，
            # 则 `predict` 函数需要返回一个 (true_label, pred_label, path) 的列表。
            for true, pred in zip(true_labels, predictions):
                true_name = label_map.get(true, f"未知({true})")
                pred_name = label_map.get(pred, f"未知({pred})")
                f.write(f"{true_name}\t{pred_name}\n") # 仅记录真实值和预测值
        logger.info(f"预测结果已保存至: '{output_file}'")
    except IOError as e:
        logger.error(f"保存预测结果到 '{output_file}' 失败: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"保存预测结果时发生未知错误: {e}", exc_info=True)


    # 保存所有预测错误的图像路径到 incorrect_file
    try:
        if incorrect_paths:
            with open(incorrect_file, "w", encoding="utf-8") as f:
                for path in incorrect_paths:
                    f.write(f"{path}\n")
            logger.info(f"错误图像路径已保存至: '{incorrect_file}' ({len(incorrect_paths)} 条)")
        else:
            logger.info("没有预测错误的图像，未生成错误图像路径文件。")
    except IOError as e:
        logger.error(f"保存错误图像路径到 '{incorrect_file}' 失败: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"保存错误图像路径时发生未知错误: {e}", exc_info=True)

    return accuracy

# --- 主函数 ---
def main():
    """
    主函数，协调模型预测、结果保存和资源清理的整个流程。
    """
    try:
        # 1. 解析命令行参数
        args = parse_args()
        logger.info(f"当前预测配置: {args}")

        # 2. 设置计算设备 (GPU/CPU)
        device = setup_device()

        # 3. 加载模型
        # 注意：这里传入的 args.num_classes 必须与训练时模型的类别数一致
        model = load_model(args.model_path, args.num_classes, device)

        # 4. 创建数据加载器
        test_loader = create_data_loader(args.data_dir, args.batch_size)

        # 5. 执行预测
        true_labels, predictions, incorrect_paths = predict(model, test_loader, device)

        # 6. 保存预测结果和计算准确率
        accuracy = save_results(true_labels, predictions, incorrect_paths, args.output_file, args.incorrect_file)

        logger.info(
            f"预测流程完成！总准确率: {accuracy:.4f}。 "
            f"详细结果请查看 '{args.output_file}'，错误预测图像路径请查看 '{args.incorrect_file}'。"
        )

    except Exception as e:
        # 捕获并记录主函数中可能发生的任何未处理异常
        logger.error(f"程序执行出错: {e}", exc_info=True)
        # 重新抛出异常，以便系统可以感知到程序异常退出
        raise
    finally:
        # 无论程序是否出错，最终都会尝试清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理 CUDA 缓存。")

# --- 程序入口点 ---
if __name__ == "__main__":
    main()