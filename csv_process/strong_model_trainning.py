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
import logging

# Configure logging for better visibility in Spark logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_spark(app_name="PCAAndRegressionExample", hdfs_namenode="hdfs://master:9000"):
    """
    初始化 SparkSession，配置执行器和驱动器资源。

    参数:
        app_name (str): Spark 应用程序名称。
        hdfs_namenode (str): HDFS NameNode 地址。

    返回:
        SparkSession: 初始化后的 Spark 会话对象。
    """
    try:
        spark = (
            SparkSession.builder
            .appName(app_name)
            .config("spark.hadoop.fs.defaultFS", hdfs_namenode)
            .config("spark.executor.instances", "4") # Number of executor instances
            .config("spark.executor.cores", "4")     # Number of cores per executor
            .config("spark.executor.memory", "8g")   # Memory per executor
            .config("spark.driver.memory", "8g")     # Memory for the driver
            .getOrCreate()
        )
        logging.info(f"SparkSession '{app_name}' 初始化成功，连接到 HDFS: {hdfs_namenode}。资源配置完成。")
        return spark
    except Exception as e:
        logging.error(f"SparkSession 初始化失败: {e}", exc_info=True)
        raise # Re-raise the exception to stop execution if Spark fails to initialize


def load_and_prepare_data(spark, hdfs_path, feature_columns, label_column, pca_k=9):
    """
    加载 HDFS 数据并进行特征组合、PCA 降维、数据类型转换、空值处理和数据打乱。

    参数:
        spark (SparkSession): Spark 会话对象。
        hdfs_path (str): HDFS 数据文件（目录）路径。
        feature_columns (list): 标准化特征列名称列表。
        label_column (str): 目标标签列名。
        pca_k (int): PCA 主成分数量，默认为 9。

    返回:
        tuple: (训练集 DataFrame, 测试集 DataFrame)
    """
    try:
        # 读取数据 (指向目录，因为Spark写入时会创建目录)
        df = spark.read.csv(hdfs_path, header=True, inferSchema=True, encoding="utf-8")
        logging.info(f"成功读取数据来自: {hdfs_path}，总行数: {df.count()}")
        logging.info("数据结构:")
        df.printSchema()
        df.show(5, truncate=False)

        # 检查并确保标签列存在且为 Double 类型
        if label_column not in df.columns:
            logging.error(f"错误: 标签列 '{label_column}' 不存在于 DataFrame 中。可用的列: {df.columns}")
            raise ValueError(f"Label column '{label_column}' not found.")
        df = df.withColumn(label_column, col(label_column).cast("double"))
        logging.info(f"标签列 '{label_column}' 已确保为 Double 类型。")

        # 检查并确保所有特征列存在且为 Double 类型
        for col_name in feature_columns:
            if col_name not in df.columns:
                logging.error(f"错误: 特征列 '{col_name}' 不存在于 DataFrame 中。可用的列: {df.columns}")
                raise ValueError(f"Feature column '{col_name}' not found.")
            df = df.withColumn(col_name, col(col_name).cast("double"))
        logging.info("所有特征列已确保为 Double 类型。")

        # 空值处理：在特征组合和PCA之前，删除包含空值的行
        # 针对所有特征列和标签列进行空值检查和删除
        cols_to_check = feature_columns + [label_column]
        original_count = df.count()
        df_cleaned = df.na.drop(subset=cols_to_check)
        if df_cleaned.count() < original_count:
            logging.warning(f"由于特征或标签列存在空值，删除了 {original_count - df_cleaned.count()} 行数据。")
        else:
            logging.info("数据中未发现特征或标签列的空值。")
        df = df_cleaned # Use the cleaned DataFrame from now on

        if df.count() == 0:
            logging.error("清洗后的 DataFrame 为空，无法进行特征工程和模型训练。")
            return None, None

        # 特征组合
        # handleInvalid="skip" will drop rows if any nulls remain in the inputCols for VectorAssembler
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
        df_assembled = assembler.transform(df)
        logging.info("特征组合完成。")
        df_assembled.select("features", label_column).show(5, truncate=False)

        # PCA 降维
        actual_num_features = len(feature_columns)
        # Ensure pca_k does not exceed the actual number of features
        if pca_k > actual_num_features:
            logging.warning(f"请求的 PCA 主成分数量 (k={pca_k}) 大于实际特征数量 ({actual_num_features})。将 k 设置为实际特征数量。")
            pca_k = actual_num_features

        if pca_k > 0 and actual_num_features > 0:
            logging.info(f"执行 PCA 降维，从 {actual_num_features} 个特征降至 {pca_k} 个主成分。")
            pca = PCA(k=pca_k, inputCol="features", outputCol="pca_features")
            pca_model = pca.fit(df_assembled)
            df_pca = pca_model.transform(df_assembled)
            logging.info("PCA 降维完成。")
            df_pca.select("pca_features", label_column).show(5, truncate=False)
        else:
            logging.warning("PCA 主成分数量或特征数量为零，跳过 PCA 降维。使用原始组合特征。")
            # If PCA is skipped, rename 'features' to 'pca_features' for consistent downstream model input
            df_pca = df_assembled.withColumnRenamed("features", "pca_features")

        # 数据打乱 (通过 randomSplit 实现，无需额外 sample)
        # df_pca = df_pca.sample(withReplacement=False, fraction=1.0, seed=1234) # This is usually not necessary before randomSplit

        # 划分训练集和测试集
        train_data, test_data = df_pca.randomSplit([0.8, 0.2], seed=1234)
        logging.info(f"数据划分完成。训练集行数: {train_data.count()}, 测试集行数: {test_data.count()}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"数据加载或处理失败: {e}", exc_info=True)
        raise # Re-raise the exception to indicate a critical failure


def compute_accuracy(predictions, label_col, prediction_col, threshold=0.5):
    """
    计算预测准确率（基于阈值）。

    参数:
        predictions (pyspark.sql.DataFrame): 包含预测结果的 DataFrame。
        label_col (str): 标签列名。
        prediction_col (str): 预测列名。
        threshold (float): 误差阈值，默认为 0.5。

    返回:
        float: 准确率。
    """
    try:
        # 确保用于计算的列没有空值
        predictions_cleaned = predictions.na.drop(subset=[label_col, prediction_col])
        if predictions_cleaned.count() == 0:
            logging.warning("用于准确率计算的 DataFrame 在去除空值后为空。返回准确率 0.0。")
            return 0.0

        predictions_with_correct = predictions_cleaned.withColumn(
            "correct", (col(label_col) - col(prediction_col)).between(-threshold, threshold)
        )
        correct_count = predictions_with_correct.filter(col("correct")).count()
        total_count = predictions_with_correct.count()
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        logging.info(f"准确率计算完成: 正确预测数 {correct_count} / 总预测数 {total_count}")
        return accuracy
    except Exception as e:
        logging.error(f"准确率计算失败: {e}", exc_info=True)
        return 0.0

def train_and_evaluate_model(model, train_data, test_data, label_col, num_folds=5, param_grid=None):
    """
    训练并评估模型（支持交叉验证或直接训练）。

    参数:
        model: 回归模型实例。
        train_data (pyspark.sql.DataFrame): 训练集。
        test_data (pyspark.sql.DataFrame): 测试集。
        label_col (str): 标签列名。
        num_folds (int): 交叉验证折数，默认为 5。
        param_grid (ParamGridBuilder): 超参数网格，如果为 None 或空，则不使用交叉验证。

    返回:
        tuple: (RMSE, 准确率, 训练后的模型)
    """
    try:
        evaluator = RegressionEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="rmse"
        )

        best_model = None
        # Check if param_grid is provided and not empty
        if param_grid and param_grid.isEmpty() is False:
            logging.info(f"开始使用交叉验证训练模型: {model.__class__.__name__}")
            crossval = CrossValidator(
                estimator=model,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=num_folds,
                seed=42,
                parallelism=4 # Adjust based on your cluster's CPU resources
            )
            cv_model = crossval.fit(train_data)
            best_model = cv_model.bestModel
            predictions = cv_model.transform(test_data)
            logging.info(f"交叉验证完成。最佳模型参数: {best_model.extractParamMap()}")
        else:
            logging.info(f"开始直接训练模型: {model.__class__.__name__} (无交叉验证)")
            model_fit = model.fit(train_data)
            best_model = model_fit
            predictions = model_fit.transform(test_data)
            logging.info("模型直接训练完成。")

        rmse = evaluator.evaluate(predictions)
        accuracy = compute_accuracy(predictions, label_col, "prediction")

        logging.info(f"模型 {model.__class__.__name__} 评估结果 - RMSE: {rmse:.4f}, Accuracy: {accuracy:.4f}")
        return rmse, accuracy, best_model
    except Exception as e:
        logging.error(f"模型训练或评估失败: {e}", exc_info=True)
        return float("inf"), 0.0, None

def main():
    """
    主函数：加载数据、训练并评估多种回归模型。
    """
    # 初始化 Spark
    spark = initialize_spark()

    # 数据路径和特征列
    # BUG FIX: HDFS path should point to the directory, not a .csv file
    hdfs_input_dir = "hdfs://master:9000/home/data/processed_Strong_Motion_Parameters_Dataset_normalized_MinMaxScaler"

    # BUG FIX: Corrected feature_columns to match normalized names from previous step
    # Make sure these names are exact as they appear in your processed CSV files
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
    # BUG FIX: Label column name should be consistent ("震级(M)" from model definitions)
    label_column = "震级(M)"
    pca_k_value = 9 # Define PCA k value here

    try:
        # 加载和准备数据
        training_data, test_data = load_and_prepare_data(
            spark, hdfs_input_dir, feature_columns, label_column, pca_k=pca_k_value
        )

        if training_data is None or test_data is None or training_data.count() == 0 or test_data.count() == 0:
            logging.error("训练集或测试集为空，无法进行模型训练。请检查数据加载和清洗步骤。")
            return

        # Define models and their configurations (including optional ParamGridBuilders for CV)
        models_configs = [
            (
                LinearRegression(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(LinearRegression.regParam, [0.01, 0.1, 0.5])
                .addGrid(LinearRegression.elasticNetParam, [0.0, 0.5, 1.0])
                .build(),
                "Linear Regression"
            ),
            (
                DecisionTreeRegressor(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(DecisionTreeRegressor.maxDepth, [5, 10, 15])
                .addGrid(DecisionTreeRegressor.minInstancesPerNode, [1, 2, 4])
                .build(),
                "Decision Tree Regression"
            ),
            (
                RandomForestRegressor(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(RandomForestRegressor.numTrees, [20, 50, 100])
                .addGrid(RandomForestRegressor.maxDepth, [5, 10, 15])
                .build(),
                "Random Forest Regression"
            ),
            (
                GBTRegressor(featuresCol="pca_features", labelCol=label_column, maxIter=50, maxDepth=10),
                ParamGridBuilder().build(), # Empty grid means no hyperparameter tuning via CV
                "GBT Regression"
            )
        ]

        # Train and evaluate models
        for model, param_grid, name in models_configs:
            logging.info(f"\n--- 开始处理模型: {name} ---")
            # Pass param_grid to train_and_evaluate_model to control CV
            rmse, accuracy, _ = train_and_evaluate_model(
                model, training_data, test_data, label_col=label_column, num_folds=5, param_grid=param_grid
            )
            logging.info(f"总结 - {name} RMSE: {rmse:.4f}, Accuracy: {accuracy:.4f}")

    except Exception as e:
        logging.error(f"程序执行失败: {e}", exc_info=True)
    finally:
        # 关闭 SparkSession
        spark.stop()
        logging.info("所有模型训练和评估完毕。SparkSession 已停止。")

if __name__ == "__main__":
    main()