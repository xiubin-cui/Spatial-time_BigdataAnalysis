import os
import shutil
import random
from pathlib import Path
from typing import Optional


def ensure_directory_exists(directory: Path) -> None:
    """
    确保指定目录存在，如果不存在则创建。

    参数:
        directory (Path): 目录路径。

    抛出:
        OSError: 如果目录创建失败。
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"创建目录 {directory} 失败: {e}")
        raise


def split_data(
    source_dir: str,
    dest_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: Optional[float] = None,
) -> None:
    """
    将数据集按指定比例分割为训练集、验证集和测试集，并复制到目标目录。

    参数:
        source_dir (str): 源数据集目录路径。
        dest_dir (str): 目标数据集目录路径。
        train_ratio (float): 训练集比例（0到1之间）。
        val_ratio (float): 验证集比例（0到1之间）。
        test_ratio (float, optional): 测试集比例。如果为 None，则计算为 1 - train_ratio - val_ratio。

    抛出:
        ValueError: 如果比例无效或总和不为1。
        OSError: 如果文件操作失败。
    """
    # 验证比例
    if not 0 <= train_ratio <= 1 or not 0 <= val_ratio <= 1:
        raise ValueError("训练集和验证集比例必须在0到1之间")
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio
    if (
        not 0 <= test_ratio <= 1
        or abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6
    ):
        raise ValueError("训练集、验证集和测试集比例之和必须为1")

    # 转换为 Path 对象
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # 验证源目录是否存在
    if not source_path.is_dir():
        raise ValueError(f"源目录 {source_dir} 不存在或不是目录")

    # 创建目标目录
    for split in ["train", "val", "test"]:
        ensure_directory_exists(dest_path / split)

    def copy_files(files: list, src_dir: Path, dest_dir: Path) -> None:
        """
        复制文件到指定目录。

        参数:
            files (list): 文件名列表。
            src_dir (Path): 源目录路径。
            dest_dir (Path): 目标目录路径。
        """
        for file in files:
            try:
                shutil.copy(src_dir / file, dest_dir / file)
                print(f"复制文件至: {dest_dir / file}")
            except (IOError, OSError) as e:
                print(f"复制文件 {src_dir / file} 失败: {e}")

    # 遍历源目录中的每个类别
    for category_path in source_path.iterdir():
        if not category_path.is_dir():
            print(f"跳过非目录项: {category_path}")
            continue

        category = category_path.name
        print(f"处理类别: {category}")

        # 获取并随机打乱文件列表
        files = [f.name for f in category_path.iterdir() if f.is_file()]
        if not files:
            print(f"类别 {category} 中未找到文件")
            continue
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
            ensure_directory_exists(split_dir)
            copy_files(file_list, category_path, split_dir)


def main() -> None:
    """
    主函数，设置参数并执行数据分割。
    """
    try:
        # 定义路径和分割比例
        source_directory = "./data/Cyclone_Wildfire_Flood_Earthquake_Database"
        destination_directory = "./data"
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        # 执行数据分割
        split_data(
            source_directory, destination_directory, train_ratio, val_ratio, test_ratio
        )
        print("数据集分割完成")
    except Exception as e:
        print(f"程序执行失败: {e}")


if __name__ == "__main__":
    main()
