from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from PIL import Image
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from pyhdfs import HdfsClient
import cv2
import numpy as np

def initialize_spark(app_name="HDFSImageDisplay"):
    """
    初始化 SparkSession，配置 HDFS 和内存参数
    参数: app_name (str): Spark 应用程序名称，默认为 "HDFSImageDisplay"
    返回: SparkSession: 初始化后的 Spark 会话对象
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop101:9000")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.memory.offHeap.enabled", True)
        .config("spark.memory.offHeap.size", "4g")
        .getOrCreate()
    )

def initialize_hdfs_client(hosts="hadoop101:9870", user_name="user"):
    """
    初始化 HDFS 客户端连接

    参数:
        hosts (str): HDFS NameNode 地址，默认为 "hadoop101:9870"
        user_name (str): HDFS 用户名，默认为 "user"

    返回:
        pyhdfs.HdfsClient: 初始化后的 HDFS 客户端对象
    """
    try:
        return HdfsClient(hosts=hosts, user_name=user_name)
    except Exception as e:
        print(f"初始化 HDFS 客户端失败: {e}")
        raise

@pandas_udf("string", PandasUDFType.SCALAR)
def encode_image(image_content):
    """
    Pandas UDF：将图像内容编码为 Base64 字符串
    参数: image_content (pandas.Series): 包含图像二进制内容的 Series
    返回: pandas.Series: 包含 Base64 编码的图像字符串的 Series
    """
    encoded_images = []
    for image in image_content:
        try:
            img = Image.open(io.BytesIO(image))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            encoded_images.append(encoded_image)
        except Exception as e:
            print(f"图像编码失败: {e}")
            encoded_images.append(None)
    return pd.Series(encoded_images)

def display_image_pil(image_base64):
    """
    使用 Matplotlib 显示 Base64 编码的 PIL 图像
    参数: image_base64 (str): Base64 编码的图像字符串
    """
    try:
        img_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_data))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"显示图像失败: {e}")

def display_image_hdfs(client, hdfs_path):
    """
    从 HDFS 读取图像并使用 OpenCV 显示

    参数:
        client (pyhdfs.HdfsClient): HDFS 客户端对象
        hdfs_path (str): HDFS 上的图像文件路径
    """
    try:
        with client.open(hdfs_path) as response:
            mat = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
            if mat is None:
                print(f"无法解码图像: {hdfs_path}")
                return
            cv2.imshow("HDFS Image", mat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"从 HDFS 读取或显示图像失败: {e}")

def process_and_display_images(spark, hdfs_path, limit=1):
    """
    从 HDFS 读取图像并处理显示（使用 Spark 和 PIL）

    参数:
        spark (SparkSession): Spark 会话对象
        hdfs_path (str): HDFS 图像文件路径（支持通配符）
        limit (int): 处理的图像数量上限，默认为 1

    返回:
        bool: 处理成功返回 True，失败返回 False
    """
    try:
        # 读取图像
        images_df = spark.read.format("binaryFile").load(hdfs_path).limit(limit)
        if images_df.count() == 0:
            print("未找到图像文件")
            return False

        # 编码为 Base64
        images_df = images_df.withColumn("image_base64", encode_image(col("content")))
        
        # 收集并显示
        image_data = images_df.select("image_base64").collect()
        for row in image_data:
            if row["image_base64"]:
                display_image_pil(row["image_base64"])
        
        return True
    except Exception as e:
        print(f"处理图像失败: {e}")
        return False

def main():
    """
    主函数：从 HDFS 读取并显示图像（支持 Spark 和直接 HDFS 方式）
    """
    # 初始化 Spark 和 HDFS 客户端
    spark = initialize_spark()
    hdfs_client = initialize_hdfs_client()

    try:
        # 定义路径
        hdfs_spark_path = "hdfs://hadoop101:9000/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/*.jpg"
        hdfs_single_path = "/image/demo.jpg"

        # 使用 Spark 处理并显示图像
        print("处理 Cyclone 数据集中的图像...")
        process_and_display_images(spark, hdfs_spark_path, limit=1)

        # 使用 HDFS 客户端显示单张图像
        print("显示单张 HDFS 图像...")
        display_image_hdfs(hdfs_client, hdfs_single_path)

    except Exception as e:
        print(f"程序执行失败: {e}")
    finally:
        # 清理资源
        spark.stop()
        plt.close('all')
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()