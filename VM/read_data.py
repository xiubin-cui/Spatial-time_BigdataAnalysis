# # from pyspark.sql import SparkSession  # SparkConf、SparkContext 和 SQLContext 都已经被封装在 SparkSession
# #
# # spark = SparkSession \
# #     .builder \
# #     .appName('my spark task') \
# #     .getOrCreate()
# from pyspark import SparkContext
# from PIL import Image
#
# # 创建 SparkContext
# sc = SparkContext(appName="ReadImages")
#
# # 定义读取图像的函数
# def read_image(path):
#     try:
#         image = Image.open(path)
#         return (path, image)
#     except:
#         return (path, None)
#
# # 定义图像文件目录
# image_dir = "hdfs://user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone"
#
# # 读取图像
# images_rdd = sc.binaryFiles(image_dir)
# images = images_rdd.map(lambda x: read_image(x[0]))
#
# # 显示读取到的图像
# for path, image in images.collect():
#     if image is not None:
#         print(f"Image path: {path}, Image size: {image.size}")
#     else:
#         print(f"Failed to read image at path: {path}")
#
# # 关闭 SparkContext
# sc.stop()


#
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, pandas_udf, PandasUDFType
# from PIL import Image
# import io
# import base64
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 初始化 SparkSession
# spark = SparkSession.builder \
#     .appName("HDFS Image Display") \
#     .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop101:9000") \
#     .getOrCreate()
#
# # 读取 HDFS 中的图片数据
# hdfs_path = 'hdfs://hadoop101:9000/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/*.jpg'
# images_df = spark.read.format("binaryFile").load(hdfs_path)
#
# # 定义 Pandas UDF 来处理图片数据
# @pandas_udf("string", PandasUDFType.SCALAR)
# def encode_image(image_content):
#     encoded_images = []
#     for image in image_content:
#         img = Image.open(io.BytesIO(image))
#         buffered = io.BytesIO()
#         img.save(buffered, format="JPEG")
#         encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
#         encoded_images.append(encoded_image)
#     return pd.Series(encoded_images)
#
# # 将图片内容编码为 Base64
# images_df = images_df.withColumn("image_base64", encode_image(col("content")))
#
# # 收集结果并展示图片
# image_data = images_df.select("image_base64").collect()
#
# for row in image_data:
#     image_base64 = row["image_base64"]
#     img_data = base64.b64decode(image_base64)
#     img = Image.open(io.BytesIO(img_data))
#
#     # 使用 matplotlib 显示图片
#     plt.imshow(img)
#     plt.axis('off')  # 关闭坐标轴
#     plt.show()
#
# # 停止 SparkSession
# spark.stop()


# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, pandas_udf, PandasUDFType
# from PIL import Image
# import io
# import base64
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 初始化 SparkSession 并增加 JVM 内存
# spark = SparkSession.builder \
#     .appName("HDFS Image Display") \
#     .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop101:9000") \
#     .config("spark.executor.memory", "4g") \
#     .config("spark.driver.memory", "4g") \
#     .config("spark.memory.offHeap.enabled", True) \
#     .config("spark.memory.offHeap.size", "4g") \
#     .getOrCreate()
#
# # 读取 HDFS 中的一张图片数据
# hdfs_path = 'hdfs://hadoop101:9000/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/*.jpg'
# images_df = spark.read.format("binaryFile").load(hdfs_path).limit(1)
#
#
# # 定义 Pandas UDF 来处理图片数据
# @pandas_udf("string", PandasUDFType.SCALAR)
# def encode_image(image_content):
#     encoded_images = []
#     for image in image_content:
#         img = Image.open(io.BytesIO(image))
#         buffered = io.BytesIO()
#         img.save(buffered, format="JPEG")
#         encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
#         encoded_images.append(encoded_image)
#     return pd.Series(encoded_images)
#
#
# # 将图片内容编码为 Base64
# images_df = images_df.withColumn("image_base64", encode_image(col("content")))
#
# # 收集结果并展示图片
# image_data = images_df.select("image_base64").collect()
#
# if image_data:
#     row = image_data[0]
#     image_base64 = row["image_base64"]
#     img_data = base64.b64decode(image_base64)
#     img = Image.open(io.BytesIO(img_data))
#
#     # 使用 matplotlib 显示图片
#     plt.imshow(img)
#     plt.axis('off')  # 关闭坐标轴
#     plt.show()
#
# # 停止 SparkSession
# spark.stop()

import cv2
from pyhdfs import HdfsClient
from numpy as np
if __name__ == '__main__':
    # 创建HDFS连接客户端
    client = HdfsClient(hosts="hadoop101", user_name="user")
    # 打开图片
    hdfs_img_path = "/image/demo.jpg"
    response = client.open(hdfs_img_path)
    # 将二进制流转化为图片
    mat = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
    # 展示
    cv2.show("demo_img", mat)
    cv2.waitKey(0)
