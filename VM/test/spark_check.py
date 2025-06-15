from pyspark.sql import SparkSession
import cv2
import numpy as np
import os
import random
import pywt

def initialize_spark(app_name="ImageProcessing"):
    """
    初始化 SparkSession

    参数:
        app_name (str): Spark 应用程序名称，默认为 "ImageProcessing"

    返回:
        SparkSession: 初始化后的 Spark 会话对象
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def color_shifting(image):
    """
    对图像应用颜色抖动（Color Shifting）

    参数:
        image (np.ndarray): 输入图像，RGB 格式

    返回:
        np.ndarray: 应用颜色抖动后的图像
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

def wavelet_denoise(image, wavelet="haar", level=1):
    """
    对图像应用小波去噪

    参数:
        image (np.ndarray): 输入图像，RGB 格式
        wavelet (str): 小波类型，默认为 "haar"
        level (int): 去噪级别，默认为 1

    返回:
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

def preprocess_image(image_content, channel_means, target_size=(224, 224), num_crops=5):
    """
    预处理图像，包括调整尺寸、翻转、裁剪、归一化、颜色抖动、光照校正、滤波和去噪

    参数:
        image_content (bytes): 图像的二进制数据
        channel_means (np.ndarray): 通道均值，用于归一化
        target_size (tuple): 裁剪目标尺寸，默认为 (224, 224)
        num_crops (int): 每张图像的裁剪数量，默认为 5

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

        # 水平翻转并裁剪
        images = [image, cv2.flip(image, 1)]
        cropped_images = []
        for img in images:
            for _ in range(num_crops):
                x = random.randint(0, img.shape[1] - target_size[1])
                y = random.randint(0, img.shape[0] - target_size[0])
                cropped_images.append(img[y:y + target_size[0], x:x + target_size[1]])

        # 归一化
        cropped_images = [img - channel_means for img in cropped_images]

        # 颜色抖动
        cropped_images = [color_shifting(img) for img in cropped_images]

        # 光照校正（直方图均衡）
        for i in range(len(cropped_images)):
            image_yuv = cv2.cvtColor(cropped_images[i], cv2.COLOR_RGB2YUV)
            image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
            cropped_images[i] = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

        # 高斯滤波
        cropped_images = [cv2.GaussianBlur(img, (5, 5), 0) for img in cropped_images]

        # 腐蚀和膨胀
        kernel = np.ones((5, 5), np.uint8)
        cropped_images = [
            cv2.dilate(cv2.erode(img, kernel, iterations=1), kernel, iterations=1)
            for img in cropped_images
        ]

        # 小波去噪
        cropped_images = [wavelet_denoise(img) for img in cropped_images]

        return cropped_images
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

def process_image(spark, hdfs_path, output_base_dir, channel_means):
    """
    从 HDFS 读取并处理图像，保存到本地

    参数:
        spark (SparkSession): Spark 会话对象
        hdfs_path (str): HDFS 图像文件路径
        output_base_dir (str): 输出基础目录
        channel_means (np.ndarray): 通道均值，用于归一化
    """
    try:
        # 读取图像
        image_df = spark.read.format("binaryFile").load(hdfs_path)
        if image_df.count() == 0:
            print(f"未找到图像: {hdfs_path}")
            return

        # 处理并保存
        for row in image_df.collect():
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
    except Exception as e:
        print(f"处理图像失败: {e}")

def main():
    """
    主函数：从 HDFS 读取、预处理并保存图像
    """
    # 初始化 Spark
    spark = initialize_spark()

    # 定义路径和参数
    hdfs_path = "hdfs://hadoop101:9000/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/0.jpg"
    output_base_dir = "/home/lhr/big_data/Cyclone"
    channel_means = np.array([123.68, 116.78, 103.94])

    try:
        # 处理图像
        process_image(spark, hdfs_path, output_base_dir, channel_means)
    except Exception as e:
        print(f"程序执行失败: {e}")
    finally:
        # 停止 SparkSession
        spark.stop()

if __name__ == "__main__":
    main()