#
# from pyspark.sql import SparkSession
# import cv2
# import numpy as np
# import os
#
# # 创建 SparkSession
# spark = SparkSession.builder.appName("ImageProcessing").getOrCreate()
#
# # 读取 HDFS 中的图片文件
# hdfs_path = "hdfs://hadoop101:9000/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/0.jpg"  # 修改为你的 HDFS 路径
# image_rdd = spark.read.format("binaryFile").load(hdfs_path).rdd
#
# # 定义图片预处理函数
# def preprocess_image(image_content):
#     # 转换二进制数据为 numpy 数组
#     image_array = np.frombuffer(image_content, dtype=np.uint8)
#     # 使用 OpenCV 解码图片
#     image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#     # 进行图片预处理，例如调整大小
#     processed_image = cv2.resize(image, (224, 224))  # 修改为所需大小
#     return processed_image
#
# # 定义保存图片的函数
# def save_image(image, output_dir, filename):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     output_path = os.path.join(output_dir, filename)
#     cv2.imwrite(output_path, image)
#
# # 处理并保存图片
# output_dir = "/home/lhr/big_data"  # 修改为本地保存路径
# for row in image_rdd.collect():
#     file_path = row.path
#     image_content = row.content
#     filename = os.path.basename(file_path)
#     processed_image = preprocess_image(image_content)
#     save_image(processed_image, output_dir, filename)
#
# # 停止 SparkSession
# spark.stop()


# from pyspark.sql import SparkSession
# import cv2
# import numpy as np
# import os
# import random
#
# # 创建 SparkSession
# spark = SparkSession.builder.appName("ImageProcessing").getOrCreate()
#
# # 读取 HDFS 中的图片文件
# hdfs_path = "hdfs://hadoop101:9000/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/0.jpg"  # 修改为你的 HDFS 路径
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
#     # 水平翻转并裁剪(Cropping)
#     images = [image, cv2.flip(image, 1)]
#     cropped_images = []
#     print(len(images))
#
#     for img in images:
#         num_crops = 5  # 修改为你需要的裁剪数量
#         for _ in range(num_crops):  # 随机裁剪 num_crops 张图片
#             x = random.randint(0, img.shape[1] - 224)
#             y = random.randint(0, img.shape[0] - 224)
#             cropped_images.append(img[y:y + 224, x:x + 224])
#
#     # 归一化(Normalizing)
#     for i in range(len(cropped_images)):
#         cropped_images[i] = cropped_images[i] - channel_means
#     #
#     # 应用颜色抖动
#     for i in range(len(cropped_images)):
#         cropped_images[i] = color_shifting(cropped_images[i])
#
#     return cropped_images
#
# # 定义保存图片的函数
# def save_images(images, output_dir, base_filename):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     for idx, image in enumerate(images):
#         output_path = os.path.join(output_dir, f"{base_filename}_{idx}.jpg")
#         cv2.imwrite(output_path, image)
#
# # 处理并保存图片
# output_dir = "/home/lhr/big_data/Cyclone"  # 修改为本地保存路径
# for row in image_df.collect():
#     file_path = row.path
#     image_content = row.content
#     filename = os.path.basename(file_path).split('.')[0]
#     processed_images = preprocess_image(image_content, channel_means)
#     if processed_images is not None:
#         save_images(processed_images, output_dir, filename)
#
# # 停止 SparkSession
# spark.stop()


# from pyspark.sql import SparkSession
# import cv2
# import numpy as np
# import os
# import random
# import pywt
# # 创建 SparkSession
# spark = SparkSession.builder.appName("ImageProcessing").getOrCreate()
#
# # 读取 HDFS 中的图片文件
# hdfs_path = "hdfs://hadoop101:9000/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/0.jpg"  # 修改为你的 HDFS 路径
# image_df = spark.read.format("binaryFile").load(hdfs_path)
#
# # 计算通道平均值（对于单张图片，可以使用通用的值）
# channel_means = np.array([123.68, 116.78, 103.94])  # 用 ImageNet 数据集的平均值作为示例
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
# # 定义小波去噪函数
# def wavelet_denoise(image, wavelet='haar', level=1):
#     coeffs = pywt.wavedec2(image, wavelet, level=level)
#     coeffs_H = list(coeffs)
#     coeffs_H[0] *= 0
#     image = pywt.waverec2(coeffs_H, wavelet)
#     return np.uint8(image)
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
#     # 水平翻转并裁剪(Cropping)
#     images = [image, cv2.flip(image, 1)]
#     cropped_images = []
#     for img in images:
#         num_crops = 5  # 修改为你需要的裁剪数量
#         for _ in range(num_crops):  # 随机裁剪 num_crops 张图片
#             x = random.randint(0, img.shape[1] - 224)
#             y = random.randint(0, img.shape[0] - 224)
#             cropped_images.append(img[y:y + 224, x:x + 224])
#
#     # 归一化(Normalizing)
#     for i in range(len(cropped_images)):
#         cropped_images[i] = cropped_images[i] - channel_means
#
#     # 应用颜色抖动
#     for i in range(len(cropped_images)):
#         cropped_images[i] = color_shifting(cropped_images[i])
#
#     # # 灰度变换（调整亮度和对比度）
#     # alpha = 1.5  # 对比度控制，取值大于1提高对比度
#     # beta = 50  # 亮度控制，取值大于0提高亮度
#     # for i in range(len(cropped_images)):
#     #     cropped_images[i] = cv2.convertScaleAbs(cropped_images[i], alpha=alpha, beta=beta)
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
#     #
#     # 腐蚀和膨胀
#     kernel = np.ones((5, 5), np.uint8)
#     for i in range(len(cropped_images)):
#         cropped_images[i] = cv2.erode(cropped_images[i], kernel, iterations=1)
#         cropped_images[i] = cv2.dilate(cropped_images[i], kernel, iterations=1)
#     #
#     # # Otsu’s二值化
#     # for i in range(len(cropped_images)):
#     #     gray_image = cv2.cvtColor(cropped_images[i], cv2.COLOR_RGB2GRAY)
#     #     _, cropped_images[i] = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # 小波去噪
#     for i in range(len(cropped_images)):
#         cropped_images[i] = wavelet_denoise(cropped_images[i])
#
#     return cropped_images
#
# # 定义保存图片的函数
# def save_images(images, output_dir, base_filename):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     for idx, image in enumerate(images):
#         output_path = os.path.join(output_dir, f"{base_filename}_{idx}.jpg")
#         cv2.imwrite(output_path, image)
#
# # 处理并保存图片
# output_dir = "/home/lhr/big_data/Cyclone"  # 修改为本地保存路径
# for row in image_df.collect():
#     file_path = row.path
#     image_content = row.content
#     filename = os.path.basename(file_path).split('.')[0]
#     processed_images = preprocess_image(image_content, channel_means)
#     if processed_images is not None:
#         save_images(processed_images, output_dir, filename)
#
# # 停止 SparkSession
# spark.stop()
