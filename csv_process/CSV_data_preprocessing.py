from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os


def preprocess_data(df):
    """
    数据预处理函数：清洗数据并处理空缺值
    参数: df (pyspark.sql.DataFrame): 输入的 Spark DataFrame
    返回: pyspark.sql.DataFrame: 清洗后的 DataFrame
    """
    # 打印数据概览
    print("数据概览:")
    df.printSchema()
    df.describe().show()

    # 检查并打印空缺值统计
    print("空缺值统计:")
    df.select([col(c).isNull().alias(c) for c in df.columns]).groupBy().sum().show()

    # 删除包含空缺值的行
    cleaned_df = df.na.drop()

    return cleaned_df


def read_and_preprocess_hdfs_csv(file_path, spark):
    """
    读取并预处理 HDFS 上的 CSV 文件
    参数:
        file_path (str): HDFS 文件路径
        spark (SparkSession): Spark 会话对象
    返回:
        pyspark.sql.DataFrame: 预处理后的 DataFrame，或在出错时返回 None
    """
    try:
        # 读取 CSV 文件
        df = spark.read.csv(file_path, header=True, inferSchema=True, encoding="utf-8")
        # 预处理数据
        return preprocess_data(df)
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {e}")
        return None


def save_as_single_csv(df, local_output_path):
    """
    将 DataFrame 保存为单个 CSV 文件

    参数:
        df (pyspark.sql.DataFrame): 输入的 Spark DataFrame
        local_output_path (str): 本地输出路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(local_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 转换为 Pandas DataFrame 并保存为 CSV
    pandas_df = df.toPandas()
    print(f"保存 {len(pandas_df)} 条记录到 {local_output_path}")
    # pandas_df.to_csv(local_output_path, index=False)  # 注释掉实际保存操作


def main():
    """
    主函数：批量读取和处理 HDFS 上的 CSV 文件，并保存结果
    """
    # 创建 SparkSession
    spark = (
        SparkSession.builder.appName("ReadAndProcessMultipleCSVs")
        .config("spark.hadoop.fs.defaultFS", "hdfs://master:9000")  # BUG
        .getOrCreate()
    )

    # CSV 文件列表
    csv_files = [
        "CEN_Center_Earthquake_Catalog",
        "GSN_Earthquake_Catalog",
        "Earthquake_disaster_data_list",
        "Strong_Motion_Parameters_Dataset",
    ]

    # 定义 HDFS 和本地路径
    base_hdfs_path = "hdfs://master:9000/home/data/"  # BUG
    base_local_path = "./data"  # BUG

    # 批量处理 CSV 文件
    for csv_file in csv_files:
        hdfs_csv_path = f"{base_hdfs_path}{csv_file}.csv"
        # 读取和预处理数据
        df = read_and_preprocess_hdfs_csv(hdfs_csv_path, spark)
        if df is not None:
            local_output_path = f"{base_local_path}processed_{csv_file}.csv"
            save_as_single_csv(df, local_output_path)

    # 停止 SparkSession
    spark.stop()


if __name__ == "__main__":
    main()
