from pyspark.sql import SparkSession
import cv2
import numpy as np
import os
import random
import pywt

# 创建 SparkSession
spark = SparkSession.builder.appName("ImageProcessing").getOrCreate()

# 读取 HDFS 中的图片文件
hdfs_dir = "hdfs://hadoop101:9000/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/"  # 修改为你的 HDFS 路径
image_df = spark.read.format("binaryFile").load(hdfs_dir + "*")

# 计算通道平均值（对于单张图片，可以使用通用的值）
channel_means = np.array([123.68, 116.78, 103.94])  # 用 ImageNet 数据集的平均值作为示例


# 定义颜色抖动（Color Shifting）函数
def color_shifting(image):
    alpha = np.random.normal(0, 0.01, 3)  # 随机生成α
    image_flat = image.reshape(-1, 3).astype(np.float32)  # 展平为二维数组
    cov = np.cov(image_flat.T)  # 计算协方差矩阵
    p, lambdas, _ = np.linalg.svd(cov)  # 计算特征向量和特征值
    delta = np.dot(p, alpha * lambdas)  # 计算偏移量

    # 添加颜色抖动
    image_flat += delta

    # 确保像素值在有效范围内
    image_flat = np.clip(image_flat, 0, 255).astype(np.uint8)

    return image_flat.reshape(image.shape)  # 恢复为原始形状


# 定义小波去噪函数
def wavelet_denoise(image, wavelet="haar", level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    image = pywt.waverec2(coeffs_H, wavelet)
    return np.uint8(image)


# 定义图片预处理函数
def preprocess_image(image_content, channel_means):
    try:
        # 转换二进制数据为 numpy 数组
        image_array = np.frombuffer(image_content, dtype=np.uint8)
        # 使用 OpenCV 解码图片
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return None

        # 调整尺寸(Rescaling)
        short_edge = min(image.shape[:2])
        scale = random.uniform(256, 480) / short_edge
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size)

        # 水平翻转并裁剪(Cropping)
        images = [image, cv2.flip(image, 1)]
        cropped_images = []
        for img in images:
            num_crops = 3  # 修改为你需要的裁剪数量
            for _ in range(num_crops):  # 随机裁剪 num_crops 张图片
                x = random.randint(0, img.shape[1] - 224)
                y = random.randint(0, img.shape[0] - 224)
                cropped_images.append(img[y : y + 224, x : x + 224])

        # 归一化(Normalizing)
        for i in range(len(cropped_images)):
            cropped_images[i] = cropped_images[i] - channel_means

        # 应用颜色抖动
        for i in range(len(cropped_images)):
            cropped_images[i] = color_shifting(cropped_images[i])

        # 光照校正（直方图均衡）
        for i in range(len(cropped_images)):
            image_yuv = cv2.cvtColor(cropped_images[i], cv2.COLOR_RGB2YUV)
            image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
            cropped_images[i] = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

        # 平滑滤波器（高斯滤波器）
        for i in range(len(cropped_images)):
            cropped_images[i] = cv2.GaussianBlur(cropped_images[i], (5, 5), 0)

        # 腐蚀和膨胀
        kernel = np.ones((5, 5), np.uint8)
        for i in range(len(cropped_images)):
            cropped_images[i] = cv2.erode(cropped_images[i], kernel, iterations=1)
            cropped_images[i] = cv2.dilate(cropped_images[i], kernel, iterations=1)

        # # 小波去噪
        # for i in range(len(cropped_images)):
        #     cropped_images[i] = wavelet_denoise(cropped_images[i])

        return cropped_images
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


# 定义保存图片的函数
def save_images(images, output_dir, base_filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{base_filename}_{idx}.jpg")
        cv2.imwrite(output_path, image)


# 处理并保存图片
output_base_dir = "/home/lhr/big_data/Cyclone"  # 修改为本地保存路径

# 批量处理文件
batch_size = 100  # 设置每批处理的文件数量
file_count = image_df.count()

for batch_start in range(0, file_count, batch_size):
    batch_df = image_df.limit(batch_start + batch_size).collect()[batch_start:]
    for row in batch_df:
        file_path = row.path
        image_content = row.content
        filename = os.path.basename(file_path).split(".")[0]
        print(filename)
        output_dir = os.path.join(output_base_dir, filename)

        # 检查文件是否已经存在
        if os.path.exists(output_dir):
            print(f"Skipping {filename}, already processed.")
            continue
        processed_images = preprocess_image(image_content, channel_means)
        if processed_images is not None:
            # output_dir = os.path.join(output_base_dir, filename)
            save_images(processed_images, output_dir, filename)

# 停止 SparkSessionq
spark.stop()
#


# import os
# import random
# import numpy as np
# import cv2
# import pywt
# from pyspark.sql import SparkSession
# import time
# from tqdm import tqdm
#
# spark = SparkSession.builder \
#     .appName("ImageProcessing") \
#     .config("spark.sql.shuffle.partitions", "8") \
#     .config("spark.network.timeout", "600s") \
#     .config("spark.executor.heartbeatInterval", "60s") \
#     .getOrCreate()
#
# # 读取 HDFS 中的图片文件
# hdfs_path = "hdfs://hadoop101:9000/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Earthquake"  # 修改为你的 HDFS 路径
# # hdfs_path = "/home/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Earthquake"
# image_df = spark.read.format("binaryFile").load(hdfs_path)
#
# # 计算通道平均值（对于单张图片，可以使用通用的值）
# channel_means = np.array([123.68, 116.78, 103.94])  # 用 ImageNet 数据集的平均值作为示例
#
#
# # 定义颜色抖动（Color Shifting）函数
# def color_shifting(image):
#     alpha = np.random.normal(0, 0.01, 3)  # 随机生成α
#     image_flat = image.reshape(-1, 3).astype(np.float32)  # 展平为二维数组
#     cov = np.cov(image_flat.T)  # 计算协方差矩阵
#     p, lambdas, _ = np.linalg.svd(cov)  # 计算特征向量和特征值
#     delta = np.dot(p, alpha * lambdas)  # 计算偏移量
#
#     # 添加颜色抖动
#     image_flat += delta
#
#     # 确保像素值在有效范围内
#     image_flat = np.clip(image_flat, 0, 255).astype(np.uint8)
#
#     return image_flat.reshape(image.shape)  # 恢复为原始形状
#
#
# # 定义小波去噪函数
# def wavelet_denoise(image, wavelet='haar', level=1):
#     coeffs = pywt.wavedec2(image, wavelet, level=level)
#     coeffs_H = list(coeffs)
#     coeffs_H[0] *= 0
#     image = pywt.waverec2(coeffs_H, wavelet)
#     return np.uint8(image)
#
#
# # 定义图片预处理函数
# def preprocess_image(image_content, channel_means):
#     # 转换二进制数据为 numpy 数组
#     image_array = np.frombuffer(image_content, dtype=np.uint8)
#     # 使用 OpenCV 解码图片
#     image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#     if image is None:
#         return None
#
#     # 调整尺寸(Rescaling)
#     short_edge = min(image.shape[:2])
#     scale = random.uniform(256, 480) / short_edge
#     new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
#     image = cv2.resize(image, new_size)
#
#     # # 水平翻转并裁剪(Cropping)
#     # images = [image, cv2.flip(image, 1)]
#     # cropped_images = []
#     # for img in images:
#     #     num_crops = 5  # 修改为你需要的裁剪数量
#     #     for _ in range(num_crops):  # 随机裁剪 num_crops 张图片
#     #         x = random.randint(0, img.shape[1] - 224)
#     #         y = random.randint(0, img.shape[0] - 224)
#     #         cropped_images.append(img[y:y + 224, x:x + 224])
#
#     # 归一化(Normalizing)
#     for i in range(len(cropped_images)):
#         cropped_images[i] = cropped_images[i] - channel_means
#
#     # 应用颜色抖动
#     for i in range(len(cropped_images)):
#         cropped_images[i] = color_shifting(cropped_images[i])
#
#     # 光照校正（直方图均衡）
#     for i in range(len(cropped_images)):
#         image_yuv = cv2.cvtColor(cropped_images[i], cv2.COLOR_RGB2YUV)
#         image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
#         cropped_images[i] = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
#
#     # 平滑滤波器（高斯滤波器）
#     for i in range(len(cropped_images)):
#         cropped_images[i] = cv2.GaussianBlur(cropped_images[i], (5, 5), 0)
#
#     # 腐蚀和膨胀
#     kernel = np.ones((5, 5), np.uint8)
#     for i in range(len(cropped_images)):
#         cropped_images[i] = cv2.erode(cropped_images[i], kernel, iterations=1)
#         cropped_images[i] = cv2.dilate(cropped_images[i], kernel, iterations=1)
#
#     # # 小波去噪
#     # for i in range(len(cropped_images)):
#     #     cropped_images[i] = wavelet_denoise(cropped_images[i])
#
#     return cropped_images
#
#
# # 定义保存图片的函数
# def save_images(images, output_dir, base_filename):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     for idx, image in enumerate(images):
#         output_path = os.path.join(output_dir, f"{base_filename}_{idx}.jpg")
#         cv2.imwrite(output_path, image)
#
#
# # 处理并保存图片
#
# batch_size = 100  # 减少批处理大小
# total_images = image_df.count()
# batch_start = 0
#
# # 使用 tqdm 实现动态进度条
# with tqdm(total=total_images, desc="Processing Images") as pbar:
#     while batch_start < total_images:
#         output_dir = "/home/lhr/big_data/Earthquake"
#         try:
#             batch_df = image_df.limit(batch_start + batch_size).collect()[batch_start:]
#             for row in batch_df:
#                 file_path = row.path
#                 filename = os.path.basename(file_path).split('.')[0]
#                 # print(filename)
#
#                 # 检查文件是否已经存在
#                 if os.path.exists(os.path.join(output_dir, filename)):
#                     # print(f"Skipping {filename}, already processed.")
#                     continue
#                 image_content = row.content
#                 processed_images = preprocess_image(image_content, channel_means)
#                 if processed_images is not None:
#                     output_dir_with_filename = os.path.join(output_dir, filename)
#                     save_images(processed_images, output_dir_with_filename, filename)
#
#             batch_start += batch_size
#             pbar.update(batch_size)  # 更新进度条
#             time.sleep(1)  # 每次批处理后休眠 1 秒
#
#         except Exception as e:
#             print(f"Error processing batch starting at {batch_start}: {e}")
#             time.sleep(5)  # 重试前休眠 5 秒
#
#
# # 停止 SparkSession
# spark.stop()


import os
import random
import numpy as np
import cv2
import pywt
from pyspark.sql import SparkSession
import time
from tqdm import tqdm

# 创建 SparkSession
spark = (
    SparkSession.builder.appName("ImageProcessing")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.network.timeout", "600s")
    .config("spark.executor.heartbeatInterval", "60s")
    .getOrCreate()
)

# 读取 HDFS 或本地文件中的图片文件
hdfs_path = "/home/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Earthquake"  # 修改为你的本地路径
image_df = spark.read.format("binaryFile").load(hdfs_path)

# 计算通道平均值（对于单张图片，可以使用通用的值）
channel_means = np.array([123.68, 116.78, 103.94])  # 用 ImageNet 数据集的平均值作为示例


# 定义颜色抖动（Color Shifting）函数
def color_shifting(image):
    alpha = np.random.normal(0, 0.01, 3)  # 随机生成α
    image_flat = image.reshape(-1, 3).astype(np.float32)  # 展平为二维数组
    cov = np.cov(image_flat.T)  # 计算协方差矩阵
    p, lambdas, _ = np.linalg.svd(cov)  # 计算特征向量和特征值
    delta = np.dot(p, alpha * lambdas)  # 计算偏移量

    # 添加颜色抖动
    image_flat += delta

    # 确保像素值在有效范围内
    image_flat = np.clip(image_flat, 0, 255).astype(np.uint8)

    return image_flat.reshape(image.shape)  # 恢复为原始形状


# 定义小波去噪函数
def wavelet_denoise(image, wavelet="haar", level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    image = pywt.waverec2(coeffs_H, wavelet)
    return np.uint8(image)


# 定义图片预处理函数
def preprocess_image(image_content, channel_means):
    # 转换二进制数据为 numpy 数组
    image_array = np.frombuffer(image_content, dtype=np.uint8)
    # 使用 OpenCV 解码图片
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # 调整尺寸(Rescaling)
    short_edge = min(image.shape[:2])
    scale = random.uniform(256, 480) / short_edge
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    image = cv2.resize(image, new_size)

    # 归一化(Normalizing)
    image = image - channel_means

    # 应用颜色抖动
    image = color_shifting(image)

    # 光照校正（直方图均衡）
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    # # 平滑滤波器（高斯滤波器）
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    #
    # # 腐蚀和膨胀
    # kernel = np.ones((5, 5), np.uint8)
    # image = cv2.erode(image, kernel, iterations=1)
    # image = cv2.dilate(image, kernel, iterations=1)

    # 小波去噪
    # image = wavelet_denoise(image)

    return [image]  # 返回列表以保持与后续代码一致


# 定义保存图片的函数
def save_images(images, output_dir, base_filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{base_filename}_{idx}.jpg")
        cv2.imwrite(output_path, image)


# 处理并保存图片
batch_size = 100  # 减少批处理大小
total_images = image_df.count()
batch_start = 0

for batch_start in range(0, file_count, batch_size):
    batch_df = image_df.limit(batch_start + batch_size).collect()[batch_start:]
    for row in batch_df:
        file_path = row.path
        image_content = row.content
        filename = os.path.basename(file_path).split(".")[0]
        print(filename)
        output_dir = os.path.join(output_base_dir, filename)

        # 检查文件是否已经存在
        if os.path.exists(output_dir):
            print(f"Skipping {filename}, already processed.")
            continue
        processed_images = preprocess_image(image_content, channel_means)
        if processed_images is not None:
            # output_dir = os.path.join(output_base_dir, filename)
            save_images(processed_images, output_dir, filename)

# 停止 SparkSession
spark.stop()
