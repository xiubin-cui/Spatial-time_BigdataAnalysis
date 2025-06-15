from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, Normalizer
from pyspark.ml.functions import vector_to_array
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_spark(app_name="BigDataFeatureProcessing", hdfs_namenode="hdfs://master:9000"):
    """
    初始化 SparkSession。
    参数:
        app_name (str): Spark 应用程序名称。
        hdfs_namenode (str): HDFS NameNode 地址。
    返回:
        SparkSession: 初始化后的 Spark 会话对象。
    """
    try:
        spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.hadoop.fs.defaultFS", hdfs_namenode)
            .getOrCreate()
        )
        logging.info(f"SparkSession '{app_name}' 初始化成功，连接到 HDFS: {hdfs_namenode}")
        return spark
    except Exception as e:
        logging.error(f"SparkSession 初始化失败: {e}", exc_info=True)
        raise # Re-raise to prevent further execution


def read_data(spark, file_path):
    """
    从 HDFS 读取 CSV 文件。
    参数:
        spark (SparkSession): Spark 会话对象。
        file_path (str): HDFS 文件路径。
    返回:
        pyspark.sql.DataFrame: 读取后的 DataFrame。
    """
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True, encoding="utf-8")
        logging.info(f"成功读取文件: {file_path}，行数: {df.count()}")
        logging.info("数据概览:")
        df.printSchema()
        df.describe().show()
        return df
    except Exception as e:
        logging.error(f"读取文件 {file_path} 失败: {e}", exc_info=True)
        return None


def clean_data(df, columns_to_check_for_nulls=None):
    """
    数据预处理函数：检查空缺值并删除包含空缺值的行。
    参数:
        df (pyspark.sql.DataFrame): 输入的 Spark DataFrame。
        columns_to_check_for_nulls (list, optional): 仅检查这些列中的空值并删除行。
                                                 如果为 None，则检查所有列。
    返回:
        pyspark.sql.DataFrame: 清洗后的 DataFrame。
    """
    if df is None:
        return None

    logging.info("空缺值统计:")
    df.select([col(c).isNull().cast("int").alias(c) for c in df.columns]).groupBy().sum().show()

    original_count = df.count()
    if columns_to_check_for_nulls:
        # Drop rows where any of the specified columns have nulls
        cleaned_df = df.na.drop(subset=columns_to_check_for_nulls)
        logging.info(f"仅在 {columns_to_check_for_nulls} 列中删除包含空缺值的行。")
    else:
        # Drop rows where any column has nulls
        cleaned_df = df.na.drop()
        logging.info("在所有列中删除包含空缺值的行。")

    cleaned_count = cleaned_df.count()
    logging.info(f"原始行数: {original_count}, 清洗后行数: {cleaned_count} (删除 {original_count - cleaned_count} 行空缺值)")
    return cleaned_df


def cast_feature_columns_to_double(df, feature_columns):
    """
    将指定特征列转换为 Double 类型。
    如果转换失败（例如，字符串“N/A”），则结果为 null。
    参数:
        df (pyspark.sql.DataFrame): 输入的 Spark DataFrame。
        feature_columns (list): 需要转换的特征列名称列表。
    返回:
        pyspark.sql.DataFrame: 转换后的 DataFrame。
    """
    if df is None:
        return None

    try:
        for column in feature_columns:
            if column in df.columns: # Check if column exists before casting
                df = df.withColumn(column, col(column).cast("double"))
            else:
                logging.warning(f"特征列 '{column}' 不存在于 DataFrame 中，跳过转换。")
        logging.info(f"成功将特征列 {feature_columns} 转换为 Double 类型。")
        return df
    except Exception as e:
        logging.error(f"转换特征列类型失败: {e}", exc_info=True)
        return None


def assemble_features(df, feature_columns, output_col_name="vector_compose_features"):
    """
    使用 VectorAssembler 将指定特征列组合成特征向量。
    设置 handleInvalid="keep" 以处理可能的 null 值，这些 null 值将被转换为 NaN。
    参数:
        df (pyspark.sql.DataFrame): 输入的 Spark DataFrame。
        feature_columns (list): 要组合的特征列名称列表。
        output_col_name (str): 组合后特征向量的列名。
    返回:
        pyspark.sql.DataFrame: 包含组合特征向量的 DataFrame。
    """
    if df is None:
        return None

    try:
        # Check if all feature_columns actually exist in the DataFrame
        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            logging.error(f"无法组装特征向量: 缺少以下列: {missing_cols}. DataFrame 列: {df.columns}")
            return None

        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol=output_col_name,
            handleInvalid="keep" # Key change here!
        )
        assembled_df = assembler.transform(df)
        logging.info("特征向量组装成功。")
        assembled_df.select(output_col_name, *feature_columns).show(truncate=False)
        return assembled_df
    except Exception as e:
        logging.error(f"特征向量组装失败: {e}", exc_info=True)
        return None


def standardize_features(df, input_col, output_col, feature_columns_for_display):
    """
    对特征向量进行 MinMax 标准化处理。
    参数:
        df (pyspark.sql.DataFrame): 包含组合特征向量的 DataFrame。
        input_col (str): 输入特征向量列名。
        output_col (str): 标准化后特征向量的列名。
        feature_columns_for_display (list): 原始特征列名称列表，用于展开和展示。
    返回:
        pyspark.sql.DataFrame: 标准化后的 DataFrame。
    """
    if df is None:
        return None

    try:
        min_max_scaler = MinMaxScaler(inputCol=input_col, outputCol=output_col)
        min_max_scaler_model = min_max_scaler.fit(df)
        scaled_df = min_max_scaler_model.transform(df)

        # Expand the scaled vector into individual columns for display/saving
        scaled_df = scaled_df.withColumn(f"{output_col}_array", vector_to_array(output_col))
        for i, feature_name in enumerate(feature_columns_for_display):
            scaled_df = scaled_df.withColumn(f"MinMaxScaler_{feature_name}", col(f"{output_col}_array")[i])

        logging.info("MinMax 标准化处理成功。")
        scaled_df.select(*[f"MinMaxScaler_{feature_name}" for feature_name in feature_columns_for_display]).show(truncate=False)
        return scaled_df
    except Exception as e:
        logging.error(f"MinMax 标准化处理失败: {e}", exc_info=True)
        return None


def normalize_features_l2_and_save(standardized_df, feature_columns, hdfs_output_path, file_name_base):
    """
    对标准化后的特征进行 L2 正则化处理并保存结果到 HDFS。
    参数:
        standardized_df (pyspark.sql.DataFrame): 标准化后的 DataFrame。
        feature_columns (list): 原始特征列名称列表。
        hdfs_output_path (str): HDFS 输出目录的基础路径。
        file_name_base (str): 原始文件名的基础部分，用于构建输出文件名。
    返回:
        pyspark.sql.DataFrame: 正则化后的 DataFrame。
    """
    if standardized_df is None:
        return None

    try:
        normalizer = Normalizer(inputCol="MinMaxScaler_features", outputCol="normalized_features", p=2.0)
        normalized_df = normalizer.transform(standardized_df)

        # 将特征向量转换为数组并拆分为单独列
        normalized_df = normalized_df.withColumn(
            "normalized_features_array", vector_to_array("normalized_features")
        )
        for i, feature_name in enumerate(feature_columns):
            normalized_df = normalized_df.withColumn(
                f"normalized_{feature_name}", col("normalized_features_array")[i]
            )

        logging.info("正则化后的数据:")
        normalized_df.select(
            *[f"normalized_{feature_name}" for feature_name in feature_columns]
        ).show(truncate=False)

        # 删除中间列
        columns_to_drop = [
            "vector_compose_features",
            "MinMaxScaler_features",
            "MinMaxScaler_features_array",
            "normalized_features",
            "normalized_features_array",
        ]
        # Only drop columns that actually exist in the DataFrame
        normalized_df_final = normalized_df.drop(*[c for c in columns_to_drop if c in normalized_df.columns])

        # --- FIX: Ensure .csv suffix and handle single file output ---
        output_hdfs_dir = f"{hdfs_output_path}processed_{file_name_base}_normalized_MinMaxScaler.csv"

        # Coalesce to 1 partition to save as a single CSV file.
        # Be cautious with very large datasets, as this can be a bottleneck.
        normalized_df_final.coalesce(1).write.csv(
            output_hdfs_dir, header=True, mode="overwrite", encoding="utf-8"
        )
        logging.info(f"正则化数据已保存至 HDFS 路径: {output_hdfs_dir}")
        return normalized_df_final
    except Exception as e:
        logging.error(f"正则化处理或保存失败: {e}", exc_info=True)
        return None


def process_csv_file_pipeline(spark, hdfs_input_full_path, hdfs_output_base_path, file_name_base, feature_columns):
    """
    处理单个 CSV 文件，包括读取、数据类型转换、特征组合、标准化和正则化，并保存结果。

    参数:
        spark (SparkSession): Spark 会话对象。
        hdfs_input_full_path (str): HDFS 输入文件的完整路径 (e.g., hdfs://.../file.csv)。
        hdfs_output_base_path (str): HDFS 输出目录的基础路径 (e.g., hdfs://.../data/).
        file_name_base (str): 原始文件名的基础部分 (e.g., "CEN_Center_Earthquake_Catalog")。
        feature_columns (list): 特征列名称列表。
    """
    logging.info(f"\n--- 开始处理文件: {hdfs_input_full_path} ---")

    # 1. 读取 CSV 文件
    df = read_data(spark, hdfs_input_full_path)
    if df is None:
        logging.error(f"由于读取失败，跳过文件 {hdfs_input_full_path} 的后续处理。")
        return

    # 2. 将特征列转换为 double 类型
    # This step is critical. If non-numeric values exist, they become null here.
    typed_df = cast_feature_columns_to_double(df, feature_columns)
    if typed_df is None:
        logging.error(f"文件 {hdfs_input_full_path} 特征列类型转换失败。")
        return

    # 3. 数据预处理 (清洗空缺值)
    # It's highly recommended to drop nulls *after* casting to double,
    # as casting non-numeric strings to double will result in nulls.
    cleaned_df = clean_data(typed_df, columns_to_check_for_nulls=feature_columns)
    if cleaned_df is None or cleaned_df.count() == 0:
        logging.warning(f"文件 {hdfs_input_full_path} 清洗后为空，跳过特征工程。")
        return

    # 4. 特征组合
    assembled_df = assemble_features(cleaned_df, feature_columns) # Pass cleaned_df here
    if assembled_df is None:
        logging.error(f"文件 {hdfs_input_full_path} 特征向量组装失败。")
        return

    # 5. 标准化
    standardized_df = standardize_features(assembled_df, "vector_compose_features", "MinMaxScaler_features", feature_columns)
    if standardized_df is None:
        logging.error(f"文件 {hdfs_input_full_path} 标准化失败。")
        return

    # 6. 正则化并保存到 HDFS
    normalize_features_l2_and_save(standardized_df, feature_columns, hdfs_output_base_path, file_name_base)


def main():
    """
    主函数：批量处理 HDFS 上的 CSV 文件，进行特征标准化和正则化，并将结果保存到 HDFS。
    """
    # --- Configuration Parameters ---
    HDFS_NAMENODE = "hdfs://master:9000" # BUG: 请检查你的 HDFS NameNode 地址和端口是否正确
    HDFS_INPUT_BASE_PATH = "hdfs://master:9000/home/data/" # BUG: 请验证你的 HDFS 数据实际存储路径
    HDFS_OUTPUT_BASE_PATH = "hdfs://master:9000/home/data/" # HDFS output path for processed files

    # Create SparkSession
    spark = initialize_spark(hdfs_namenode=HDFS_NAMENODE)
    if spark is None:
        logging.error("SparkSession 未成功初始化，程序退出。")
        return

    # CSV 文件及其对应的特征列
    csv_configs = [
        {
            "file": "CEN_Center_Earthquake_Catalog",
            "features": ["震源深度(Km)", "Ms7", "mL", "mb8", "mB9"],
        },
        {
            "file": "GSN_Earthquake_Catalog",
            "features": ["震源深度(Km)", "Ms7", "mL", "mb8", "mB9"],
        },
        {
            "file": "Strong_Motion_Parameters_Dataset",
            "features": [
                "震源深度", "震中距", "仪器烈度", "总峰值加速度PGA", "总峰值速度PGV",
                "参考Vs30", "东西分量PGA", "南北分量PGA", "竖向分量PGA",
                "东西分量PGV", "南北分量PGV", "竖向分量PGV",
            ],
        },
    ]

    # Batch process CSV files
    for config in csv_configs:
        hdfs_input_full_path = f"{HDFS_INPUT_BASE_PATH}{config['file']}.csv"
        process_csv_file_pipeline(
            spark, hdfs_input_full_path, HDFS_OUTPUT_BASE_PATH, config["file"], config["features"]
        )

    # Stop SparkSession
    spark.stop()
    logging.info("\n所有文件处理完毕。SparkSession 已停止。")


if __name__ == "__main__":
    main()