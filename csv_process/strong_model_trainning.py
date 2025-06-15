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
    初始化 SparkSession，配置执行器和驱动器资源

    参数:
        app_name (str): Spark 应用程序名称，默认为 "PCAAndRegressionExample"

    返回:
        SparkSession: 初始化后的 Spark 会话对象
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.executor.instances", "4")
        .config("spark.executor.cores", "4")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )

def load_and_prepare_data(spark, hdfs_path, feature_columns, label_column="震级", pca_k=9):
    """
    加载 HDFS 数据并进行特征组合、PCA 降维和数据打乱

    参数:
        spark (SparkSession): Spark 会话对象
        hdfs_path (str): HDFS 数据文件路径
        feature_columns (list): 标准化特征列名称列表
        label_column (str): 目标标签列名，默认为 "震级"
        pca_k (int): PCA 主成分数量，默认为 9

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
        pca = PCA(k=pca_k, inputCol="features", outputCol="pca_features")
        pca_model = pca.fit(df_assembled)
        df_pca = pca_model.transform(df_assembled)
        print("PCA 降维结果:")
        df_pca.select("pca_features").show(5, truncate=False)

        # 数据打乱
        df_pca = df_pca.sample(withReplacement=False, fraction=1.0, seed=1234)

        # 划分训练集和测试集
        return df_pca.randomSplit([0.8, 0.2], seed=1234)
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
            "correct", (col(label_col) - col(prediction_col)).between(-threshold, threshold)
        )
        return predictions.filter(col("correct")).count() / predictions.count()
    except Exception as e:
        print(f"准确率计算失败: {e}")
        return 0.0

def train_and_evaluate_model(model, train_data, test_data, label_col="震级", num_folds=5):
    """
    训练并评估模型（使用交叉验证）

    参数:
        model: 回归模型实例
        train_data (pyspark.sql.DataFrame): 训练集
        test_data (pyspark.sql.DataFrame): 测试集
        label_col (str): 标签列名，默认为 "震级"
        num_folds (int): 交叉验证折数，默认为 5

    返回:
        tuple: (RMSE, 准确率, 训练后的模型)
    """
    try:
        evaluator = RegressionEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="rmse"
        )
        crossval = CrossValidator(
            estimator=model,
            estimatorParamMaps=ParamGridBuilder().build(),
            evaluator=evaluator,
            numFolds=num_folds,
            seed=42
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
    hdfs_path = "hdfs://master:9000/home/data/processed_Strong_Motion_Parameters_Dataset_normalized_MinMaxScaler.csv" #BUG
    feature_columns = [
        "normalized_震源深度",
        "normalized_震中距",
        "normalized_仪器烈度",
        "normalized_总峰值加速度PGA",
        "normalized_总峰值速度PGV",
        "normalized_参考Vs30",
        "normalized_东西分量PGA",
        "normalized_南北分量PGA",
        "normalized_竖向分量PGA",
        "normalized_东西分量PGV",
        "normalized_南北分量PGV",
        "normalized_竖向分量PGV"
    ]

    try:
        # 加载和准备数据
        training_data, test_data = load_and_prepare_data(spark, hdfs_path, feature_columns)

        # 定义模型（使用固定超参数，符合原代码逻辑）
        models = [
            (
                LinearRegression(
                    featuresCol="pca_features",
                    labelCol="震级(M)",
                    regParam=0.1,
                    elasticNetParam=0.5
                ),
                "Linear Regression"
            ),
            (
                DecisionTreeRegressor(
                    featuresCol="pca_features",
                    labelCol="震级(M)",
                    maxDepth=10,
                    minInstancesPerNode=2
                ),
                "Decision Tree Regression"
            ),
            (
                RandomForestRegressor(
                    featuresCol="pca_features",
                    labelCol="震级(M)",
                    numTrees=50,
                    maxDepth=10
                ),
                "Random Forest Regression"
            ),
            (
                GBTRegressor(
                    featuresCol="pca_features",
                    labelCol="震级(M)",
                    maxIter=50,
                    maxDepth=10
                ),
                "GBT Regression"
            )
        ]

        # 训练并评估模型
        for model, name in models:
            rmse, accuracy, _ = train_and_evaluate_model(
                model, training_data, test_data, num_folds=5
            )
            print(f"{name} RMSE: {rmse:.4f}, Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"程序执行失败: {e}")
    finally:
        # 关闭 SparkSession
        spark.stop()

if __name__ == "__main__":
    main()