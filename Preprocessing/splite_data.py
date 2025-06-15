import os
import shutil
import random
from typing import Tuple
from pathlib import Path


def split_data(
    source_dir: str, dest_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> None:
    """
    将数据集按指定比例分割为训练集、验证集和测试集，并复制到目标目录。

    Args:
        source_dir (str): 源数据集目录路径
        dest_dir (str): 目标数据集目录路径
        train_ratio (float): 训练集比例，默认为0.7
        val_ratio (float): 验证集比例，默认为0.15

    Raises:
        ValueError: 如果比例之和不等于1或输入路径无效
        OSError: 如果文件操作失败
    """
    # 验证输入比例
    test_ratio = 1.0 - train_ratio - val_ratio
    if not 0 <= train_ratio <= 1 or not 0 <= val_ratio <= 1 or not 0 <= test_ratio <= 1:
        raise ValueError("训练集、验证集和测试集比例必须在0到1之间")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("训练集、验证集和测试集比例之和必须为1")

    # 转换为Path对象，确保路径规范化
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # 验证源目录存在
    if not source_path.is_dir():
        raise ValueError(f"源目录 {source_dir} 不存在或不是目录")

    # 创建目标目录
    for split in ["train", "val", "test"]:
        (dest_path / split).mkdir(parents=True, exist_ok=True)

    def copy_files(files: list, src_dir: Path, dest_dir: Path) -> None:
        """复制文件到指定目录"""
        for file in files:
            try:
                shutil.copy(src_dir / file, dest_dir / file)
            except (IOError, OSError) as e:
                print(f"复制文件 {file} 失败: {e}")

    # 遍历源目录中的每个类别
    for category_path in source_path.iterdir():
        if not category_path.is_dir():
            continue

        category = category_path.name
        print(f"处理类别: {category}")

        # 获取并打乱文件列表
        files = list(category_path.iterdir())
        random.shuffle(files)

        # 计算分割点
        total_files = len(files)
        train_end = int(train_ratio * total_files)
        val_end = train_end + int(val_ratio * total_files)

        # 分割文件列表
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # 创建目标类别目录并复制文件
        for split, file_list in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            split_dir = dest_path / split / category
            split_dir.mkdir(parents=True, exist_ok=True)
            copy_files([f.name for f in file_list], category_path, split_dir)


def main():
    """主函数，设置参数并执行数据分割"""
    try:
        # 定义路径和分割比例
        source_directory = "../data/Cyclone_Wildfire_Flood_Earthquake_Database"  # BUG
        destination_directory = "./data"  # BUG
        train_ratio = 0.7
        val_ratio = 0.15

        # 执行数据分割
        split_data(source_directory, destination_directory, train_ratio, val_ratio)
        print("数据集分割完成")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
