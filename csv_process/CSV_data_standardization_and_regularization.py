from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, Normalizer
from pyspark.ml.functions import vector_to_array
import os


def initialize_spark(app_name="ReadAndProcessMultipleCSVs"):
    """
    初始化 SparkSession
    参数: app_name (str): Spark 应用程序名称，默认为 "ReadAndProcessMultipleCSVs"
    返回: SparkSession: 初始化后的 Spark 会话对象
    """
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.hadoop.fs.defaultFS", "hdfs://master:9000") #BUG
        .getOrCreate()
    )


def vector_compose(df, feature_columns):
    """
    使用 VectorAssembler 将指定特征列组合成特征向量

    参数:
        df (pyspark.sql.DataFrame): 输入的 Spark DataFrame
        feature_columns (list): 要组合的特征列名称列表

    返回:
        pyspark.sql.DataFrame: 包含组合特征向量的 DataFrame
    """
    try:
        assembler = VectorAssembler(
            inputCols=feature_columns, outputCol="vector_compose_features"
        )
        assembled_df = assembler.transform(df)
        print("组装后的特征向量数据:")
        assembled_df.select("vector_compose_features", *feature_columns).show(
            truncate=False
        )
        return assembled_df
    except Exception as e:
        print(f"特征向量组装失败: {e}")
        return None


def standardize_features(assembled_df, feature_columns):
    """
    对特征向量进行 MinMax 标准化处理

    参数:
        assembled_df (pyspark.sql.DataFrame): 包含组合特征向量的 DataFrame
        feature_columns (list): 特征列名称列表

    返回:
        pyspark.sql.DataFrame: 标准化后的 DataFrame
    """
    try:
        min_max_scaler = MinMaxScaler(
            inputCol="vector_compose_features", outputCol="MinMaxScaler_features"
        )
        min_max_scaler_model = min_max_scaler.fit(assembled_df)
        standardized_df = min_max_scaler_model.transform(assembled_df)

        # 将特征向量转换为数组并拆分为单独列
        standardized_df = standardized_df.withColumn(
            "MinMaxScaler_features_array", vector_to_array("MinMaxScaler_features")
        )
        for i, feature_name in enumerate(feature_columns):
            standardized_df = standardized_df.withColumn(
                f"MinMaxScaler_{feature_name}", col("MinMaxScaler_features_array")[i]
            )

        print("标准化后的数据:")
        standardized_df.select(
            *[f"MinMaxScaler_{feature_name}" for feature_name in feature_columns]
        ).show(truncate=False)
        return standardized_df
    except Exception as e:
        print(f"标准化处理失败: {e}")
        return None


def normalize_features(standardized_df, feature_columns, output_path, csv_file):
    """
    对标准化后的特征进行 L2 正则化处理并保存结果

    参数:
        standardized_df (pyspark.sql.DataFrame): 标准化后的 DataFrame
        feature_columns (list): 特征列名称列表
        output_path (str): 输出文件路径
        csv_file (str): CSV 文件名

    返回:
        pyspark.sql.DataFrame: 正则化后的 DataFrame
    """
    try:
        normalizer = Normalizer(
            inputCol="MinMaxScaler_features", outputCol="normalized_features", p=2.0
        )
        normalized_df = normalizer.transform(standardized_df)

        # 将特征向量转换为数组并拆分为单独列
        normalized_df = normalized_df.withColumn(
            "normalized_features_array", vector_to_array("normalized_features")
        )
        for i, feature_name in enumerate(feature_columns):
            normalized_df = normalized_df.withColumn(
                f"normalized_{feature_name}", col("normalized_features_array")[i]
            )

        print("正则化后的数据:")
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
        normalized_df = normalized_df.drop(*columns_to_drop)

        # 保存到 HDFS
        output_file = f"{output_path}_normalized_MinMaxScaler.csv"
        normalized_df.write.csv(
            output_file, header=True, mode="overwrite", encoding="utf-8"
        )
        print(f"正则化数据已保存至: {output_file}")
        return normalized_df
    except Exception as e:
        print(f"正则化处理或保存失败: {e}")
        return None


def process_csv_file(spark, hdfs_path, output_path, csv_file, feature_columns):
    """
    处理单个 CSV 文件，包括读取、特征转换、标准化和正则化

    参数:
        spark (SparkSession): Spark 会话对象
        hdfs_path (str): HDFS 输入文件路径
        output_path (str): 输出文件路径
        csv_file (str): CSV 文件名
        feature_columns (list): 特征列名称列表
    """
    try:
        # 读取 CSV 文件
        df = spark.read.csv(hdfs_path, header=True, inferSchema=True, encoding="utf-8")

        # 将特征列转换为 double 类型
        for column in feature_columns:
            df = df.withColumn(column, col(column).cast("double"))

        # 特征组合
        assembled_df = vector_compose(df, feature_columns)
        if assembled_df is None:
            return

        # 标准化
        standardized_df = standardize_features(assembled_df, feature_columns)
        if standardized_df is None:
            return

        # 正则化并保存
        normalize_features(standardized_df, feature_columns, output_path, csv_file)
    except Exception as e:
        print(f"处理文件 {hdfs_path} 失败: {e}")


def main():
    """
    主函数：批量处理 HDFS 上的 CSV 文件，进行特征标准化和正则化
    """
    spark = initialize_spark()

    # 定义文件和路径
    base_hdfs_path = "hdfs://master:9000/home/data/" #BUG
    base_output_path = "./data" #BUG

    # CSV 文件和对应的特征列
    csv_configs = [
        {
            "file": "processed_CEN_Center_Earthquake_Catalog",
            "features": ["震源深度(Km)", "Ms7", "mL", "mb", "mB"],
        },
        {
            "file": "processed_GSN_Earthquake_Catalog",
            "features": ["震源深度(Km)", "Ms7", "mL", "mb", "mB"],
        },
        {
            "file": "processed_Strong_Motion_Parameters_Dataset",
            "features": [
                "震源深度",
                "震中距",
                "仪器烈度",
                "总峰值加速度PGA",
                "总峰值速度PGV",
                "参考Vs30",
                "东西分量PGA",
                "南北分量PGA",
                "竖向分量PGA",
                "东西分量PGV",
                "南北分量PGV",
                "竖向分量PGV",
            ],
        }
    ]

    # 批量处理 CSV 文件
    for config in csv_configs:
        hdfs_csv_path = f"{base_hdfs_path}{config['file']}.csv"
        output_path = f"{base_output_path}{config['file']}"
        process_csv_file(
            spark, hdfs_csv_path, output_path, config["file"], config["features"]
        )

    # 停止 SparkSession
    spark.stop()


if __name__ == "__main__":
    main()
