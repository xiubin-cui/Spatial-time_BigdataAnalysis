import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, List


class CustomImageFolder(datasets.ImageFolder):
    """扩展 ImageFolder 类以获取图像路径"""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """
        获取图像、标签和路径

        Args:
            index (int): 数据索引

        Returns:
            Tuple[torch.Tensor, int, str]: 图像张量、标签和路径
        """
        image, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return image, label, path


def setup_device() -> torch.device:
    """设置计算设备（GPU或CPU）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    加载预训练模型

    Args:
        model_path (str): 模型文件路径
        device (torch.device): 计算设备

    Returns:
        nn.Module: 加载的模型
    """
    try:
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")


def create_data_loader(data_dir: str, batch_size: int = 64) -> DataLoader:
    """
    创建测试数据加载器

    Args:
        data_dir (str): 数据集根目录路径
        batch_size (int): 批次大小，默认为64

    Returns:
        DataLoader: 测试数据加载器
    """
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    try:
        test_dataset = CustomImageFolder(root=f"{data_dir}/val", transform=transform)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        raise ValueError(f"加载数据集失败: {e}")


def predict(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    对测试数据集进行预测并记录错误预测的图像路径

    Args:
        model: 神经网络模型
        test_loader: 测试数据加载器
        device: 计算设备

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: 真实标签、预测标签和错误图像路径列表
    """
    model.eval()
    true_labels = []
    predictions = []
    incorrect_ids = []

    with torch.no_grad():
        for inputs, labels, paths in test_loader:
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                true_labels.extend(labels.cpu().numpy())
                predictions.extend(preds.cpu().numpy())

                # 记录错误预测的图像路径
                for i, (pred, label, path) in enumerate(zip(preds, labels, paths)):
                    if pred != label:
                        incorrect_ids.append(path)
            except Exception as e:
                print(f"预测批次出错: {e}")
                continue

    return np.array(true_labels), np.array(predictions), incorrect_ids


def save_results(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    incorrect_ids: List[str],
    output_file: str = "image_batch_results.txt",
    incorrect_file: str = "incorrect_images.txt",
) -> float:
    """
    保存预测结果和错误图像路径，并计算准确率

    Args:
        true_labels: 真实标签数组
        predictions: 预测标签数组
        incorrect_ids: 错误预测的图像路径列表
        output_file: 预测结果输出文件路径
        incorrect_file: 错误图像路径输出文件路径

    Returns:
        float: 预测准确率
    """
    label_map = {0: "Cyclone", 1: "Earthquake", 2: "Flood", 3: "Wildfire"}
    accuracy = np.mean(true_labels == predictions)

    # 保存预测结果
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("真实值\t预测值\n")
            for true, pred in zip(true_labels, predictions):
                true_name = label_map.get(true, "未知")
                pred_name = label_map.get(pred, "未知")
                f.write(f"{true_name}\t{pred_name}\n")
    except Exception as e:
        print(f"保存预测结果失败: {e}")

    # 保存错误图像路径
    try:
        with open(incorrect_file, "w", encoding="utf-8") as f:
            for path in incorrect_ids:
                f.write(f"{path}\n")
    except Exception as e:
        print(f"保存错误图像路径失败: {e}")

    return accuracy


def main():
    """主函数，执行模型预测和结果保存"""
    try:
        # 配置参数
        data_dir = "./data"  # BUG
        model_path = "./base_model_18_0.1_source.pth"  # BUG
        output_file = "./result/image_batch_results.txt"  # BUG
        incorrect_file = "./result/incorrect_images.txt"  # BUG

        # 初始化
        device = setup_device()
        model = load_model(model_path, device)
        test_loader = create_data_loader(data_dir)

        # 进行预测
        true_labels, predictions, incorrect_ids = predict(model, test_loader, device)

        # 保存结果并计算准确率
        accuracy = save_results(
            true_labels, predictions, incorrect_ids, output_file, incorrect_file
        )
        print(
            f"预测完成，结果已写入 {output_file}，错误图像路径已写入 {incorrect_file}"
        )
        print(f"测试集准确率: {accuracy:.4f}")

    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
