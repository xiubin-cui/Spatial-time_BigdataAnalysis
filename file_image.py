import os
import shutil
import random


def split_data(source_dir, dest_dir, train_ratio, val_ratio):
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    test_dir = os.path.join(dest_dir, "test")

    # 确保train, val, test目录存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for category in os.listdir(source_dir):
        print(category)
        category_dir = os.path.join(source_dir, category)
        print(category_dir)
        if not os.path.isdir(category_dir):
            continue

        # 获取当前类别的所有文件
        files = os.listdir(category_dir)
        random.shuffle(files)

        # 计算分割索引
        train_end = int(train_ratio * len(files))
        val_end = train_end + int(val_ratio * len(files))

        # 分割数据集
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # 创建类别目录
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        # 复制文件到train目录
        for file in train_files:
            print(os.path.join(train_dir, category, file))
            shutil.copy(
                os.path.join(category_dir, file),
                os.path.join(train_dir, category, file),
            )

        # 复制文件到val目录
        for file in val_files:
            shutil.copy(
                os.path.join(category_dir, file), os.path.join(val_dir, category, file)
            )

        # 复制文件到test目录
        for file in test_files:
            shutil.copy(
                os.path.join(category_dir, file), os.path.join(test_dir, category, file)
            )


# 定义源目录和目标目录
source_directory = "./image_handle_class"  # 替换为实际路径
destination_directory = "./data"  # 替换为实际路径

# 设置数据集分割比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 分割数据集
split_data(source_directory, destination_directory, train_ratio, val_ratio)
