# from pyspark.sql import SparkSession
#
# # 创建 SparkSession
# spark = SparkSession.builder.appName("FilterRows").getOrCreate()
#
# # 读取本地 CSV 文件
# file_path = "/home/lhr/big_data/processed_地震灾情数据列表.csv"  # 替换为你的文件路径
# df = spark.read.csv(file_path, header=True, inferSchema=True)
#
# # 打印原始数据框架
# print("Original DataFrame:")
# df.show()
#
# # 过滤掉 "直接经济损（万元）" 列值为 0 的行
# filtered_df = df.filter(df["直接经济损（万元）"] != 0)
#
# # 打印过滤后的数据框架
# print("Filtered DataFrame:")
# filtered_df.show()
#
# # 将过滤后的数据保存回文件（可选）
# filtered_df.write.csv("hdfs://hadoop101:9000/user/lhr/big_data/processed_地震灾情数据列表_scend.csv",encoding="utf-8",header=True)
# # filtered_df=filtered_df.toPandas()
# # filtered_df.to_csv("/home/lhr/big_data/processed_地震灾情数据列表_scend.csv",encoding="utf-8", index=False)
# # 停止 SparkSession
# spark.stop()


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler, Normalizer
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

spark = (
    SparkSession.builder.appName("ReadAndProcessMultipleCSVs")
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop101:9000")
    .getOrCreate()
)


def vector_compose(df, vector):
    # 选择要标准化/正则化的特征列
    # 使用 VectorAssembler 将特征列组合成特征向量
    assembler = VectorAssembler(inputCols=vector, outputCol="vector_compose_features")
    assembled_df = assembler.transform(df)

    # 打印组装后的数据框架
    print("Assembled DataFrame:")
    assembled_df.select("vector_compose_features", *vector).show(truncate=False)
    assembled_df.show()
    return assembled_df


def Standardization(assembled_df, local_output_path, vector, csv_file):

    # 创建 MinMaxScaler 对象
    min_max_scaler = MinMaxScaler(
        inputCol="vector_compose_features", outputCol="MinMaxScaler_features"
    )

    # 训练 MinMaxScaler 模型并进行转换
    min_max_scaler_model = min_max_scaler.fit(assembled_df)
    MinMaxScaler_df = min_max_scaler_model.transform(assembled_df)
    print(MinMaxScaler_df["MinMaxScaler_features"])
    # 打印正则化后的数据框架
    print("MinMaxScaler DataFrame:")
    MinMaxScaler_df.select("MinMaxScaler_features").show(truncate=False)

    # 将特征向量转换为数组
    MinMaxScaler_df = MinMaxScaler_df.withColumn(
        "MinMaxScaler_features_array", vector_to_array("MinMaxScaler_features")
    )
    print("MinMaxScaler_features_array DataFrame:")
    MinMaxScaler_df.select("MinMaxScaler_features_array").show(truncate=False)

    # 将数组中的每个元素拆分为单独的列
    for i, feature_name in enumerate(vector):
        MinMaxScaler_df = MinMaxScaler_df.withColumn(
            f"MinMaxScaler_{feature_name}", col("MinMaxScaler_features_array")[i]
        )

    # 打印正则化后的数据框架
    print("MinMaxScaler DataFrame with Separate Columns:")
    MinMaxScaler_df.select(
        *[f"MinMaxScaler_{feature_name}" for feature_name in vector]
    ).show(truncate=False)

    MinMaxScaler_df.show()
    # 将正则化后的数据保存到 HDFS
    # MinMaxScaler_df.write.csv(f"{local_output_path}_MinMaxScaler.csv", header=True, mode="overwrite", encoding="utf-8")

    return MinMaxScaler_df


def Normalization(assembled_df, local_output_path, vector, csv_file):
    # 创建 Normalizer 对象
    normalizer = Normalizer(
        inputCol="MinMaxScaler_features", outputCol="normalized_features", p=2.0
    )

    # 使用 Normalizer 进行转换
    normalized_df = normalizer.transform(assembled_df)
    normalized_df.show()
    # 打印正则化后的数据框架
    print("Normalized DataFrame:")
    normalized_df.select("normalized_features").show(truncate=False)

    # 将特征向量转换为数组
    normalized_df = normalized_df.withColumn(
        "normalized_features_array", vector_to_array("normalized_features")
    )
    print("normalized_features_array DataFrame:")
    normalized_df.select("normalized_features_array").show(truncate=False)
    # 将数组中的每个元素拆分为单独的列
    for i, feature_name in enumerate(vector):
        normalized_df = normalized_df.withColumn(
            f"normalized_{feature_name}", col("normalized_features_array")[i]
        )

    # 打印正则化后的数据框架
    print("Normalized DataFrame with Separate Columns:")
    normalized_df.select(
        *[f"normalized_{feature_name}" for feature_name in vector]
    ).show(truncate=False)

    normalized_df.show()
    columns_to_drop = [
        "MinMaxScaler_features",
        "vector_compose_features",
        "normalized_features",
        "MinMaxScaler_features_array",
        "normalized_features_array",
    ]
    # 删除指定列
    normalized_df = normalized_df.drop(*columns_to_drop)
    # 保存至hdfs
    normalized_df.write.csv(
        f"{local_output_path}_normalized_MinMaxScaler.csv",
        header=True,
        mode="overwrite",
        encoding="utf-8",
    )
    # # 保存至本地
    # normalized_df = normalized_df.toPandas()
    # normalized_df.to_csv(f"/home/lhr/big_data/{csv_file}_normalized_MinMaxScaler.csv", encoding="utf-8", index=False)
    return normalized_df


if __name__ == "__main__":
    # HDFS 路径和文件
    base_hdfs_path = "hdfs://hadoop101:9000/user/lhr/big_data/"
    base_local_path = "hdfs://hadoop101:9000/user/lhr/big_data/"

    # CSV 文件列表
    csv_data = [
        # "processed_中国地震台网地震目录_1",
        "processed_全球地震台网地震目录_2_1",
        # "processed_地震灾情数据列表_scend",
        "processed_强震动参数数据集_2_1",
    ]
    for csv_file in csv_data:
        hdfs_csv_path = f"{base_hdfs_path}{csv_file}.csv"
        # 读取和预处理数据
        df = spark.read.csv(
            hdfs_csv_path, header=True, inferSchema=True, encoding="utf-8"
        )
        # 打印原始数据框架
        # print("Original DataFrame:")
        # df.show()

        local_output_path = f"{base_local_path}{csv_file}"
        if csv_file == "processed_中国地震台网地震目录_1":
            vector = ["震源深度(Km)", "Ms7", "mL", "mb7", "mB8"]
            for column in vector:
                df = df.withColumn(column, col(column).cast("double"))
            assembled_df = vector_compose(df, vector)
            MinMaxScaler_df = Standardization(
                assembled_df, local_output_path, vector, csv_file
            )
            normalized_df = Normalization(
                MinMaxScaler_df, local_output_path, vector, csv_file
            )

        elif csv_file == "processed_全球地震台网地震目录_2_1":
            vector = ["震源深度(Km)", "Ms7", "mL", "mb7", "mB8"]
            for column in vector:
                df = df.withColumn(column, col(column).cast("double"))
            assembled_df = vector_compose(df, vector)
            MinMaxScaler_df = Standardization(
                assembled_df, local_output_path, vector, csv_file
            )
            normalized_df = Normalization(
                MinMaxScaler_df, local_output_path, vector, csv_file
            )

        # elif csv_file == "processed_地震灾情数据列表_scend":
        #     vector = ["震源深度(Km)", "Ms7", "mL", "mb7", "mB8"]
        #     for column in vector:
        #         df = df.withColumn(column, col(column).cast("double"))
        #     assembled_df = vector_compose(df, vector)
        #     normalized_df = Normalization(assembled_df, local_output_path)

        elif csv_file == "processed_强震动参数数据集_2_1":
            vector = [
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
            ]
            for column in vector:
                df = df.withColumn(column, col(column).cast("double"))
            assembled_df = vector_compose(df, vector)
            MinMaxScaler_df = Standardization(
                assembled_df, local_output_path, vector, csv_file
            )
            normalized_df = Normalization(
                MinMaxScaler_df, local_output_path, vector, csv_file
            )

    spark.stop()
