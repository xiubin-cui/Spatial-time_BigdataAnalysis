import os
import shutil
import random
from typing import Tuple

def ensure_directory_exists(directory: str) -> None:
    """
    确保指定目录存在，如果不存在则创建

    参数:
        directory (str): 目录路径
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        print(f"创建目录 {directory} 失败: {e}")
        raise

def split_data(
    source_dir: str,
    dest_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float = None
) -> None:
    """
    将数据集按指定比例分割为训练集、验证集和测试集，并复制到目标目录

    参数:
        source_dir (str): 源数据集目录路径
        dest_dir (str): 目标目录路径
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float, optional): 测试集比例，默认为 1 - train_ratio - val_ratio

    返回:
        None
    """
    try:
        # 验证比例
        if not 0 <= train_ratio <= 1 or not 0 <= val_ratio <= 1:
            raise ValueError("训练集和验证集比例必须在 [0, 1] 范围内")
        if test_ratio is None:
            test_ratio = 1 - train_ratio - val_ratio
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("训练集、验证集和测试集比例之和必须为 1")

        # 创建目标目录
        train_dir = os.path.join(dest_dir, "train")
        val_dir = os.path.join(dest_dir, "val")
        test_dir = os.path.join(dest_dir, "test")
        ensure_directory_exists(dest_dir)
        ensure_directory_exists(train_dir)
        ensure_directory_exists(val_dir)
        ensure_directory_exists(test_dir)

        # 遍历源目录中的类别
        for category in os.listdir(source_dir):
            category_dir = os.path.join(source_dir, category)
            if not os.path.isdir(category_dir):
                print(f"跳过非目录项: {category_dir}")
                continue

            print(f"处理类别: {category}")

            # 获取并随机打乱文件列表
            files = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]
            if not files:
                print(f"类别 {category} 中未找到文件")
                continue
            random.shuffle(files)

            # 计算分割索引
            total_files = len(files)
            train_end = int(train_ratio * total_files)
            val_end = train_end + int(val_ratio * total_files)

            # 分割文件列表
            train_files = files[:train_end]
            val_files = files[train_end:val_end]
            test_files = files[val_end:]

            # 创建类别子目录
            train_category_dir = os.path.join(train_dir, category)
            val_category_dir = os.path.join(val_dir, category)
            test_category_dir = os.path.join(test_dir, category)
            ensure_directory_exists(train_category_dir)
            ensure_directory_exists(val_category_dir)
            ensure_directory_exists(test_category_dir)

            # 复制文件到对应目录
            def copy_files(file_list: list, src_dir: str, dst_dir: str) -> None:
                for file in file_list:
                    src_path = os.path.join(src_dir, file)
                    dst_path = os.path.join(dst_dir, file)
                    try:
                        shutil.copy(src_path, dst_path)
                        print(f"复制文件至: {dst_path}")
                    except (shutil.Error, OSError) as e:
                        print(f"复制文件 {src_path} 失败: {e}")

            copy_files(train_files, category_dir, train_category_dir)
            copy_files(val_files, category_dir, val_category_dir)
            copy_files(test_files, category_dir, test_category_dir)

    except Exception as e:
        print(f"数据集分割失败: {e}")
        raise

def main() -> None:
    """
    主函数：执行数据集分割
    """
    # 配置路径和分割比例
    source_directory = "./Cyclone_Wildfire_Flood_Earthquake_Database"
    destination_directory = "./data_source2"
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    try:
        split_data(source_directory, destination_directory, train_ratio, val_ratio, test_ratio)
        print("数据集分割完成")
    except Exception as e:
        print(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()