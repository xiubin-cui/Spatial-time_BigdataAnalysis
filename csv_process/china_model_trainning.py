from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col


def initialize_spark(app_name="PCAAndRegressionExample"):
    """
    初始化 SparkSession

    参数:
        app_name (str): Spark 应用程序名称，默认为 "PCAAndRegressionExample"

    返回:
        SparkSession: 初始化后的 Spark 会话对象
    """
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_and_prepare_data(spark, hdfs_path, feature_columns, label_column="Ms"):
    """
    加载 HDFS 数据并进行特征组合和 PCA 降维

    参数:
        spark (SparkSession): Spark 会话对象
        hdfs_path (str): HDFS 数据文件路径
        feature_columns (list): 标准化特征列名称列表
        label_column (str): 目标标签列名，默认为 "Ms"

    返回:
        tuple: (训练集 DataFrame, 测试集 DataFrame)
    """
    try:
        # 读取数据
        df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
        print("数据结构:")
        df.printSchema()
        df.show(5, truncate=False)

        # 特征组合
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df_assembled = assembler.transform(df)
        print("特征组合结果:")
        df_assembled.select("features").show(5, truncate=False)

        # PCA 降维
        pca = PCA(k=4, inputCol="features", outputCol="pca_features")
        pca_model = pca.fit(df_assembled)
        df_pca = pca_model.transform(df_assembled)
        print("PCA 降维结果:")
        df_pca.select("pca_features").show(5, truncate=False)

        # 划分训练集和测试集
        return df_pca.randomSplit([0.8, 0.2], seed=42)
    except Exception as e:
        print(f"数据加载或处理失败: {e}")
        raise


def compute_accuracy(predictions, label_col, prediction_col, threshold=0.5):
    """
    计算预测准确率（基于阈值）

    参数:
        predictions (pyspark.sql.DataFrame): 包含预测结果的 DataFrame
        label_col (str): 标签列名
        prediction_col (str): 预测列名
        threshold (float): 误差阈值，默认为 0.5

    返回:
        float: 准确率
    """
    try:
        predictions = predictions.withColumn(
            "correct",
            (col(label_col) - col(prediction_col)).between(-threshold, threshold),
        )
        return predictions.filter(col("correct")).count() / predictions.count()
    except Exception as e:
        print(f"准确率计算失败: {e}")
        return 0.0


def train_and_evaluate_model(model, param_grid, train_data, test_data, label_col="Ms"):
    """
    训练并评估模型（使用交叉验证）

    参数:
        model: 回归模型实例
        param_grid: 超参数网格
        train_data (pyspark.sql.DataFrame): 训练集
        test_data (pyspark.sql.DataFrame): 测试集
        label_col (str): 标签列名，默认为 "Ms"

    返回:
        tuple: (RMSE, 准确率, 训练后的模型)
    """
    try:
        evaluator = RegressionEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="rmse"
        )
        crossval = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            seed=42,
        )
        cv_model = crossval.fit(train_data)
        predictions = cv_model.transform(test_data)
        rmse = evaluator.evaluate(predictions)
        accuracy = compute_accuracy(predictions, label_col, "prediction")
        return rmse, accuracy, cv_model
    except Exception as e:
        print(f"模型训练或评估失败: {e}")
        return float("inf"), 0.0, None


def main():
    """
    主函数：加载数据、训练并评估多种回归模型
    """
    # 初始化 Spark
    spark = initialize_spark()

    # 数据路径和特征列
    hdfs_path = "hdfs://master:9000/home/data/processed_CEN_Center_Earthquake_Catalog_normalized_MinMaxScaler.csv" #BUG
    feature_columns = [
        "normalized_震源深度(Km)",
        "normalized_Ms7",
        "normalized_mL",
        "normalized_mb",
        "normalized_mB"
    ]

    try:
        # 加载和准备数据
        training_data, test_data = load_and_prepare_data(
            spark, hdfs_path, feature_columns
        )

        # 定义模型和超参数网格
        models = [
            (
                LinearRegression(featuresCol="pca_features", labelCol="Ms"),
                ParamGridBuilder()
                .addGrid(LinearRegression.regParam, [0.01, 0.1, 0.5])
                .addGrid(LinearRegression.elasticNetParam, [0.0, 0.5, 1.0])
                .build(),
                "Linear Regression",
            ),
            (
                DecisionTreeRegressor(featuresCol="pca_features", labelCol="Ms"),
                ParamGridBuilder()
                .addGrid(DecisionTreeRegressor.maxDepth, [5, 10, 15])
                .addGrid(DecisionTreeRegressor.minInstancesPerNode, [1, 2, 4])
                .build(),
                "Decision Tree Regression",
            ),
            (
                RandomForestRegressor(featuresCol="pca_features", labelCol="Ms"),
                ParamGridBuilder()
                .addGrid(RandomForestRegressor.numTrees, [20, 50, 100])
                .addGrid(RandomForestRegressor.maxDepth, [5, 10, 15])
                .build(),
                "Random Forest Regression",
            ),
            (
                GBTRegressor(featuresCol="pca_features", labelCol="Ms"),
                ParamGridBuilder()
                .addGrid(GBTRegressor.maxIter, [20, 50, 100])
                .addGrid(GBTRegressor.maxDepth, [5, 10, 15])
                .build(),
                "GBT Regression",
            ),
        ]

        # 训练并评估模型
        for model, param_grid, name in models:
            rmse, accuracy, _ = train_and_evaluate_model(
                model, param_grid, training_data, test_data
            )
            print(f"{name} RMSE: {rmse:.4f}, Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"程序执行失败: {e}")
    finally:
        # 关闭 SparkSession
        spark.stop()


if __name__ == "__main__":
    main()
