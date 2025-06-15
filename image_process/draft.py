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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("prediction.log")]
)
logger = logging.getLogger(__name__)

class CustomImageFolder(datasets.ImageFolder):
    """扩展 ImageFolder 类以获取图像路径"""
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        image, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return image, label, path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Predict with ResNet18 model")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--model-path", type=str, default="./resnet18_0.01_source.pth", help="Model file path")
    parser.add_argument("--output-file", type=str, default="./image_batch_results.txt", help="Output file for predictions")
    parser.add_argument("--incorrect-file", type=str, default="./incorrect_images.txt", help="Output file for incorrect image paths")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for data loader")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes")
    args = parser.parse_args()
    return args

def setup_device() -> torch.device:
    """设置计算设备（GPU或CPU）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    return device

def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    加载预训练模型
    """
    try:
        # 检查模型文件是否存在
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

        # 初始化 ResNet18 模型
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # 加载 state_dict
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise RuntimeError(f"无法加载模型: {e}")

def create_data_loader(data_dir: str, batch_size: int) -> DataLoader:
    """
    创建测试数据加载器
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 减小分辨率以降低内存占用
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    try:
        # 检查数据集目录是否存在
        data_dir = Path(data_dir) / "val"
        if not data_dir.exists():
            raise FileNotFoundError(f"数据集目录 {data_dir} 不存在")

        test_dataset = CustomImageFolder(root=str(data_dir), transform=transform)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise ValueError(f"无法加载数据集: {e}")

def predict(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    对测试数据集进行预测并记录错误预测的图像路径
    """
    model.eval()
    true_labels = []
    predictions = []
    incorrect_ids = []

    with torch.no_grad():
        for batch_id, (inputs, labels, paths) in enumerate(test_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                true_labels.extend(labels.cpu().numpy())
                predictions.extend(preds.cpu().numpy())

                for i, (pred, label, path) in enumerate(zip(preds, labels, paths)):
                    if pred != label:
                        incorrect_ids.append(path)
            except RuntimeError as e:
                logger.error(f"预测批次 {batch_id} 出错: {e}")
                continue

    logger.info(f"预测完成，错误预测图像数: {len(incorrect_ids)}")
    return np.array(true_labels), np.array(predictions), incorrect_ids

def save_results(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    incorrect_ids: List[str],
    output_file: str,
    incorrect_file: str,
) -> float:
    """
    保存预测结果和错误图像路径，并计算准确率
    """
    label_map = {0: "Cyclone", 1: "Earthquake", 2: "Flood", 3: "Wildfire"}
    accuracy = np.mean(true_labels == predictions)

    # 确保输出目录存在
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存预测结果
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("真实值\t预测值\n")
            for true, pred in zip(true_labels, predictions):
                true_name = label_map.get(true, "未知")
                pred_name = label_map.get(pred, "未知")
                f.write(f"{true_name}\t{pred_name}\n")
        logger.info(f"预测结果已保存至: {output_file}")
    except Exception as e:
        logger.error(f"保存预测结果失败: {e}")

    # 保存错误图像路径
    try:
        with open(incorrect_file, "w", encoding="utf-8") as f:
            for path in incorrect_ids:
                f.write(f"{path}\n")
        logger.info(f"错误图像路径已保存至: {incorrect_file}")
    except Exception as e:
        logger.error(f"保存错误图像路径失败: {e}")

    return accuracy

def main():
    """主函数，执行模型预测和结果保存"""
    try:
        args = parse_args()
        logger.info(f"配置: {args}")

        device = setup_device()
        model = load_model(args.model_path, args.num_classes, device)
        test_loader = create_data_loader(args.data_dir, args.batch_size)

        true_labels, predictions, incorrect_ids = predict(model, test_loader, device)
        accuracy = save_results(true_labels, predictions, incorrect_ids, args.output_file, args.incorrect_file)

        logger.info(
            f"预测完成，结果已写入 {args.output_file}，错误图像路径已写入 {{args.incorrect_file}}，准确率: {accuracy:.4f}"
        )

    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理 CUDA 缓存")

if __name__ == "__main__":
    main()