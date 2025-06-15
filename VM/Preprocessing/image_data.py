import os
import random
import time
import numpy as np
import cv2
from pyspark.sql import SparkSession
from tqdm import tqdm

def initialize_spark(app_name="ImageProcessing"):
    """
    初始化 SparkSession，配置分区和超时参数
    参数: app_name (str): Spark 应用程序名称，默认为 "ImageProcessing"
    返回: SparkSession: 初始化后的 Spark 会话对象
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
        .getOrCreate()
    )

def color_shifting(image):
    """
    对图像应用颜色抖动（Color Shifting）
    参数: image (np.ndarray): 输入图像，RGB 格式
    返回: np.ndarray: 应用颜色抖动后的图像
    """
    try:
        alpha = np.random.normal(0, 0.01, 3)
        image_flat = image.reshape(-1, 3).astype(np.float32)
        cov = np.cov(image_flat.T)
        p, lambdas, _ = np.linalg.svd(cov)
        delta = np.dot(p, alpha * lambdas)
        image_flat += delta
        return np.clip(image_flat, 0, 255).astype(np.uint8).reshape(image.shape)
    except Exception as e:
        print(f"颜色抖动处理失败: {e}")
        return image

def preprocess_image(image_content, channel_means, target_size=(224, 224)):
    """
    预处理图像，包括调整尺寸、归一化、颜色抖动和光照校正

    参数:
        image_content (bytes): 图像的二进制数据
        channel_means (np.ndarray): 通道均值，用于归一化
        target_size (tuple): 目标图像尺寸，默认为 (224, 224)

    返回:
        list: 包含处理后的图像列表
    """
    try:
        # 解码图像
        image_array = np.frombuffer(image_content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            print("图像解码失败")
            return None

        # 调整尺寸
        short_edge = min(image.shape[:2])
        scale = random.uniform(256, 480) / short_edge
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size)

        # 归一化
        image = image - channel_means

        # 颜色抖动
        image = color_shifting(image)

        # 光照校正（直方图均衡）
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

        # 裁剪到目标尺寸
        x = random.randint(0, image.shape[1] - target_size[1])
        y = random.randint(0, image.shape[0] - target_size[0])
        image = image[y:y + target_size[0], x:x + target_size[1]]

        return [image]
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None

def save_images(images, output_dir, base_filename):
    """
    保存处理后的图像到指定目录

    参数:
        images (list): 处理后的图像列表
        output_dir (str): 输出目录路径
        base_filename (str): 图像基础文件名
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        for idx, image in enumerate(images):
            output_path = os.path.join(output_dir, f"{base_filename}_{idx}.jpg")
            cv2.imwrite(output_path, image)
            print(f"图像保存至: {output_path}")
    except Exception as e:
        print(f"保存图像失败: {e}")

def process_image_batch(spark, hdfs_path, output_base_dir, batch_size=100):
    """
    批量处理 HDFS 中的图像文件

    参数:
        spark (SparkSession): Spark 会话对象
        hdfs_path (str): HDFS 或本地输入路径
        output_base_dir (str): 输出基础目录
        batch_size (int): 每批处理的图像数量，默认为 100
    """
    try:
        # 读取图像文件
        image_df = spark.read.format("binaryFile").load(hdfs_path)
        total_images = image_df.count()
        channel_means = np.array([123.68, 116.78, 103.94])

        # 使用 tqdm 显示进度
        with tqdm(total=total_images, desc="处理图像") as pbar:
            for batch_start in range(0, total_images, batch_size):
                batch_df = image_df.limit(batch_start + batch_size).offset(batch_start).collect()
                for row in batch_df:
                    file_path = row.path
                    filename = os.path.basename(file_path).split('.')[0]
                    output_dir = os.path.join(output_base_dir, filename)

                    # 检查是否已处理
                    if os.path.exists(output_dir):
                        print(f"跳过 {filename}，已处理")
                        continue

                    processed_images = preprocess_image(row.content, channel_means)
                    if processed_images:
                        save_images(processed_images, output_dir, filename)

                pbar.update(min(batch_size, total_images - batch_start))
                time.sleep(1)  # 防止过载
    except Exception as e:
        print(f"批处理失败: {e}")

def main():
    """
    主函数：初始化 Spark 并批量处理图像
    """
    spark = initialize_spark()
    hdfs_path = "/home/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Earthquake"
    output_base_dir = "/home/lhr/big_data/Earthquake"
    process_image_batch(spark, hdfs_path, output_base_dir)
    spark.stop()

if __name__ == "__main__":
    main()