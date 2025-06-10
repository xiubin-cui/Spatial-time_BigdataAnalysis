# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
#
#
# def preprocess_data(df):
#     # 数据清洗和处理空缺值
#     # 1. 显示数据的基本信息
#     print("Data Overview:")
#     df.printSchema()
#     df.describe().show()
#
#     # 2. 检查空缺值
#     print("Null values count:")
#     df.select([col(c).isNull().alias(c) for c in df.columns]).groupBy().sum().show()
#
#     # 3. 删除包含任何空缺值的行
#     cleaned_df = df.na.drop()
#
#     # 4. 显示处理后的数据
#     print("Cleaned Data Overview:")
#     cleaned_df.printSchema()
#     cleaned_df.describe().show()
#
#     return cleaned_df
#
#
# def read_and_preprocess_hdfs_csv(file_path):
#     # 创建 SparkSession
#     spark = SparkSession.builder \
#         .appName("ReadHDFSCsv") \
#         .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop101:9000") \
#         .getOrCreate()
#
#     # 读取 HDFS CSV 文件
#     try:
#         df = spark.read.csv(file_path, header=True, inferSchema=True, encoding="utf-8")
#         # 预处理数据
#         cleaned_df = preprocess_data(df)
#         # 显示处理后的数据
#         cleaned_df.show()
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         # 停止 SparkSession
#         spark.stop()
#
#
# if __name__ == '__main__':
#     # HDFS 上的 CSV 文件路径
#     csv_data=["中国地震台网地震目录","全球地震台网地震目录_2","地震灾情数据列表","强震动参数数据集_2"]
#     hdfs_csv_path = "hdfs://hadoop101:9000/user/lhr/big_data/中国地震台网地震目录.csv"
#
#     # 调用函数读取和预处理 CSV 文件
#     read_and_preprocess_hdfs_csv(hdfs_csv_path)

import os

output_dir = "/home/lhr/big_data/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
#
# def preprocess_data(df):
#     # 数据清洗和处理空缺值
#     print("Data Overview:")
#     df.printSchema()
#     df.describe().show()
#
#     print("Null values count:")
#     df.select([col(c).isNull().alias(c) for c in df.columns]).groupBy().sum().show()
#
#     # 删除包含任何空缺值的行
#     cleaned_df = df.na.drop()
#
#     print("Cleaned Data Overview:")
#     cleaned_df.printSchema()
#     cleaned_df.describe().show()
#
#     return cleaned_df
#
# def read_and_preprocess_hdfs_csv(file_path, spark):
#     # 读取 HDFS CSV 文件
#     try:
#         df = spark.read.csv(file_path, header=True, inferSchema=True, encoding="utf-8")
#         # 预处理数据
#         cleaned_df = preprocess_data(df)
#         return cleaned_df
#     except Exception as e:
#         print(f"An error occurred while processing {file_path}: {e}")
#         return None
#
# def main():
#     # 创建 SparkSession
#     spark = SparkSession.builder \
#         .appName("ReadAndProcessMultipleCSVs") \
#         .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop101:9000") \
#         .getOrCreate()
#
#     # CSV 文件列表
#     csv_data = [
#         "中国地震台网地震目录",
#         "全球地震台网地震目录_2",
#         "地震灾情数据列表",
#         "强震动参数数据集_2"
#     ]
#
#     # HDFS 路径和文件
#     base_hdfs_path = "hdfs://hadoop101:9000/user/lhr/big_data/"
#     base_local_path = "/home/lhr/big_data/"
#
#     for csv_file in csv_data:
#         hdfs_csv_path = f"{base_hdfs_path}{csv_file}.csv"
#         # 读取和预处理数据
#         df = read_and_preprocess_hdfs_csv(hdfs_csv_path, spark)
#         if df is not None:
#             # 将处理后的 DataFrame 写入本地
#             local_output_path = f"{base_local_path}processed_{csv_file}.csv"
#             df.write.csv(local_output_path, header=True, mode="overwrite")
#             print(f"Processed data saved to {local_output_path}")
#
#     # 停止 SparkSession
#     spark.stop()
#
# if __name__ == '__main__':
#     main()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os


def preprocess_data(df):
    # 数据清洗和处理空缺值
    print("Data Overview:")
    df.printSchema()
    df.describe().show()

    print("Null values count:")
    df.select([col(c).isNull().alias(c) for c in df.columns]).groupBy().sum().show()

    # 删除包含任何空缺值的行
    cleaned_df = df.na.drop()

    # print("Cleaned Data Overview:")
    # cleaned_df.printSchema()
    # cleaned_df.describe().show()

    return cleaned_df


def read_and_preprocess_hdfs_csv(file_path, spark):
    # 读取 HDFS CSV 文件
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True, encoding="utf-8")
        # 预处理数据
        cleaned_df = preprocess_data(df)
        return cleaned_df
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None


def save_as_single_csv(df, local_output_path):
    # 确保输出目录存在
    output_dir = os.path.dirname(local_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将 DataFrame 转换为 Pandas DataFrame
    pandas_df = df.toPandas()
    print(len(pandas_df))
    # 保存为单个 CSV 文件
    # pandas_df.to_csv(local_output_path, index=False)
    # print(f"Processed data saved to {local_output_path}")


def main():
    # 创建 SparkSession
    spark = (
        SparkSession.builder.appName("ReadAndProcessMultipleCSVs")
        .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop101:9000")
        .getOrCreate()
    )

    # CSV 文件列表
    csv_data = [
        "中国地震台网地震目录",
        "全球地震台网地震目录_2",
        "地震灾情数据列表",
        "强震动参数数据集_2",
    ]

    # HDFS 路径和文件
    base_hdfs_path = "hdfs://hadoop101:9000/user/lhr/big_data/"
    base_local_path = "hdfs://hadoop101:9000/user/lhr/big_data/"

    for csv_file in csv_data:
        hdfs_csv_path = f"{base_hdfs_path}{csv_file}.csv"
        # 读取和预处理数据
        df = read_and_preprocess_hdfs_csv(hdfs_csv_path, spark)
        if df is not None:
            local_output_path = f"{base_local_path}processed_{csv_file}.csv"
            print(local_output_path)
            save_as_single_csv(df, local_output_path)
            print(f"Processed data saved to {local_output_path}")

    # 停止 SparkSession
    spark.stop()


if __name__ == "__main__":
    main()
