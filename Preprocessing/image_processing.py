# 图像数据预处理（如归一化、调整大小、去噪）。
import os
import random
import numpy as np
import cv2
import pywt
from pyspark.sql import SparkSession
from pathlib import Path
from typing import List, Optional

def create_spark_session(app_name: str = "ImageProcessing") -> SparkSession:
    """
    创建并配置 SparkSession。

    Args:
        app_name (str): Spark 应用程序名称

    Returns:
        SparkSession: 配置好的 SparkSession 对象
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
        .getOrCreate()
    )

def color_shifting(image: np.ndarray) -> np.ndarray:
    """
    对图像应用颜色抖动。

    Args:
        image (np.ndarray): 输入图像

    Returns:
        np.ndarray: 应用颜色抖动后的图像
    """
    try:
        alpha = np.random.normal(0, 0.01, 3)
        image_flat = image.reshape(-1, 3).astype(np.float32)
        cov = np.cov(image_flat.T)
        p, lambdas, _ = np.linalg.svd(cov)
        delta = np.dot(p, alpha * lambdas)
        image_flat = np.clip(image_flat + delta, 0, 255).astype(np.uint8)
        return image_flat.reshape(image.shape)
    except Exception as e:
        print(f"颜色抖动处理失败: {e}")
        return image

def wavelet_denoise(image: np.ndarray, wavelet: str = "haar", level: int = 1) -> np.ndarray:
    """
    对图像应用小波去噪。

    Args:
        image (np.ndarray): 输入图像
        wavelet (str): 小波类型，默认为 'haar'
        level (int): 小波分解级别，默认为 1

    Returns:
        np.ndarray: 去噪后的图像
    """
    try:
        coeffs = pywt.wavedec2(image, wavelet, level=level)
        coeffs_H = list(coeffs)
        coeffs_H[0] *= 0
        return np.uint8(pywt.waverec2(coeffs_H, wavelet))
    except Exception as e:
        print(f"小波去噪失败: {e}")
        return image

def preprocess_image(image_content: bytes, channel_means: np.ndarray) -> Optional[List[np.ndarray]]:
    """
    预处理图像，包括调整大小、归一化、颜色抖动、光照校正、滤波和去噪。

    Args:
        image_content (bytes): 图像的二进制数据
        channel_means (np.ndarray): 图像通道均值

    Returns:
        Optional[List[np.ndarray]]: 处理后的图像列表，若失败返回 None
    """
    try:
        # 解码图像
        image_array = np.frombuffer(image_content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            print("无法解码图像")
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

        # 光照校正
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

        # 高斯滤波
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # 腐蚀和膨胀
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.dilate(image, kernel, iterations=1)

        # 小波去噪
        image = wavelet_denoise(image)

        return [image]
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None

def save_images(images: List[np.ndarray], output_dir: Path, base_filename: str) -> None:
    """
    保存处理后的图像到指定目录。

    Args:
        images (List[np.ndarray]): 要保存的图像列表
        output_dir (Path): 输出目录
        base_filename (str): 文件基础名称
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        output_path = output_dir / f"{base_filename}.jpg"
        try:
            cv2.imwrite(str(output_path), image)
        except Exception as e:
            print(f"保存图像 {output_path} 失败: {e}")

def main():
    """
    主函数，执行图像处理和保存流程。
    """
    try:
        # 初始化 SparkSession
        spark = create_spark_session()

        # 配置路径和参数
        hdfs_path = r"D:\source\python\torch_big_data\Cyclone_Wildfire_Flood_Earthquake_Database\Wildfire"
        output_base_dir = Path("./Wildfire")
        channel_means = np.array([123.68, 116.78, 103.94])
        batch_size = 100

        # 读取图像数据
        image_df = spark.read.format("binaryFile").load(hdfs_path)
        file_count = image_df.count()

        # 批量处理
        for batch_start in range(0, file_count, batch_size):
            batch_df = image_df.limit(batch_start + batch_size).collect()[batch_start:]
            for row in batch_df:
                file_path = row.path
                filename = Path(file_path).stem
                print(f"处理文件: {filename}")

                output_path = output_base_dir / f"{filename}.jpg"
                if output_path.exists():
                    print(f"跳过 {filename}，文件已存在")
                    continue

                processed_images = preprocess_image(row.content, channel_means)
                if processed_images:
                    save_images(processed_images, output_base_dir, filename)

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
    finally:
        spark.stop()
        print("SparkSession 已停止")

if __name__ == "__main__":
    main()