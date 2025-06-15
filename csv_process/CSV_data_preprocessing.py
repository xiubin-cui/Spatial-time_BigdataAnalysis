from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os
import logging

# Configure logging for better output than print statements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_data(df):
    """
    数据预处理函数：清洗数据并处理空缺值
    参数: df (pyspark.sql.DataFrame): 输入的 Spark DataFrame
    返回: pyspark.sql.DataFrame: 清洗后的 DataFrame
    """
    logging.info("数据概览:")
    df.printSchema()
    df.describe().show()

    logging.info("空缺值统计:")
    # Using cast("int") to convert boolean (True/False) to integer (1/0) for sum() calculation
    df.select([col(c).isNull().cast("int").alias(c) for c in df.columns]).groupBy().sum().show()

    original_count = df.count()
    cleaned_df = df.na.drop()
    cleaned_count = cleaned_df.count()
    logging.info(f"原始行数: {original_count}, 清洗后行数: {cleaned_df.count()}")
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
        # Pass all read options as keyword arguments directly to .csv() method
        df = spark.read.csv(
            file_path,
            header=True,
            inferSchema=True,
            encoding="utf-8"  # 'encoding' parameter typically handles character encoding
        )
        # Process the data
        return preprocess_data(df)
    except Exception as e:
        logging.error(f"处理文件 {file_path} 时发生错误: {e}", exc_info=True)
        return None


def save_dataframe_to_hdfs(df, hdfs_output_path):
    """
    将 DataFrame 保存为 HDFS 上的 CSV 文件。
    Spark DataFrame.write.csv automatically handles saving to HDFS if the path is an HDFS path.

    参数:
        df (pyspark.sql.DataFrame): 输入的 Spark DataFrame
        hdfs_output_path (str): HDFS 输出路径 (e.g., hdfs://master:9000/path/to/file.csv)
    """
    if df is None:
        logging.warning(f"尝试保存一个空的 DataFrame 到 HDFS: {hdfs_output_path}. 操作跳过。")
        return

    try:
        # For HDFS, you don't need to create directories manually like with local file system
        # Spark handles this. Use 'overwrite' mode for simplicity, or 'append' if needed.
        # Spark writes partitioned output (multiple files in a directory).
        # The output path should be a directory, not a single file name.
        df.write.csv(hdfs_output_path, header=True, mode="overwrite", encoding="utf-8")
        logging.info(f"成功保存处理后的数据到 HDFS: {hdfs_output_path}")
    except Exception as e:
        logging.error(f"保存 DataFrame 到 HDFS 失败: {e}", exc_info=True)


def main():
    """
    主函数：批量读取和处理 HDFS 上的 CSV 文件，并将结果保存到 HDFS。
    """
    # --- Configuration Parameters ---
    # BUG: Please ensure this is the correct address and port for your HDFS NameNode
    HDFS_NAMENODE = "hdfs://master:9000"
    # BUG: Please verify this HDFS path matches your actual HDFS data storage path
    HDFS_INPUT_BASE_PATH = "hdfs://master:9000/home/data/"
    # Define the HDFS output base path for processed files
    # The output will be a directory containing multiple part-files
    HDFS_OUTPUT_BASE_PATH = "hdfs://master:9000/home/data/" # As per your requirement

    # Create SparkSession
    spark = (
        SparkSession.builder.appName("ReadAndProcessMultipleCSVsToHDFS")
        .config("spark.hadoop.fs.defaultFS", HDFS_NAMENODE)
        .getOrCreate()
    )
    logging.info(f"SparkSession initialized, connected to HDFS: {HDFS_NAMENODE}")

    # CSV files list
    csv_files = [
        "CEN_Center_Earthquake_Catalog",
        "GSN_Earthquake_Catalog",
        "Earthquake_disaster_data_list",
        "Strong_Motion_Parameters_Dataset",
    ]

    # Batch process CSV files
    for csv_file in csv_files:
        hdfs_input_path = f"{HDFS_INPUT_BASE_PATH}{csv_file}.csv"
        logging.info(f"\n--- 正在处理文件: {hdfs_input_path} ---")

        # Read and preprocess data
        df = read_and_preprocess_hdfs_csv(hdfs_input_path, spark)
        
        if df is not None:
            # Define the HDFS output path for the processed file
            # Spark writes to a directory, not a single file name.
            # The structure will be: hdfs://master:9000/home/data/processed_CEN_Center_Earthquake_Catalog/part-XXXXX.csv
            hdfs_output_dir = f"{HDFS_OUTPUT_BASE_PATH}processed_{csv_file}.csv"
            save_dataframe_to_hdfs(df, hdfs_output_dir)
        else:
            logging.error(f"由于数据读取或预处理失败，跳过文件 {csv_file} 的保存操作。")

    # Stop SparkSession
    spark.stop()
    logging.info("\n所有文件处理完毕。SparkSession 已停止。")


if __name__ == "__main__":
    main()