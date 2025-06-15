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

# Configure logging for better output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_spark(app_name="PCAAndRegressionExample", hdfs_namenode="hdfs://master:9000"):
    """
    初始化 SparkSession。

    参数:
        app_name (str): Spark 应用程序名称，默认为 "PCAAndRegressionExample"。
        hdfs_namenode (str): HDFS NameNode 地址。

    返回:
        SparkSession: 初始化后的 Spark 会话对象。
    """
    try:
        spark = SparkSession.builder.appName(app_name)\
            .config("spark.hadoop.fs.defaultFS", hdfs_namenode)\
            .getOrCreate()
        logging.info(f"SparkSession '{app_name}' 初始化成功，连接到 HDFS: {hdfs_namenode}")
        return spark
    except Exception as e:
        logging.error(f"SparkSession 初始化失败: {e}", exc_info=True)
        raise # Re-raise to prevent further execution


def load_and_prepare_data(spark, hdfs_path, feature_columns, label_column="Ms"):
    """
    加载 HDFS 数据并进行特征组合和 PCA 降维。

    参数:
        spark (SparkSession): Spark 会话对象。
        hdfs_path (str): HDFS 数据文件路径。
        feature_columns (list): 标准化特征列名称列表。
        label_column (str): 目标标签列名，默认为 "Ms"。

    返回:
        tuple: (训练集 DataFrame, 测试集 DataFrame)
    """
    try:
        # 读取数据
        # For CSV files, Spark will create a directory with part-files.
        # So the path here should be the directory where the CSV parts are.
        df = spark.read.csv(hdfs_path, header=True, inferSchema=True, encoding="utf-8")
        logging.info(f"成功读取数据来自: {hdfs_path}，总行数: {df.count()}")
        logging.info("数据结构:")
        df.printSchema()
        df.show(5, truncate=False)

        # Ensure label column exists and is double type
        if label_column not in df.columns:
            logging.error(f"错误: 标签列 '{label_column}' 不存在于 DataFrame 中。可用的列: {df.columns}")
            raise ValueError(f"Label column '{label_column}' not found.")
        df = df.withColumn(label_column, col(label_column).cast("double"))
        logging.info(f"标签列 '{label_column}' 已确保为 Double 类型。")

        # Ensure all feature columns exist and are double type
        for col_name in feature_columns:
            if col_name not in df.columns:
                logging.error(f"错误: 特征列 '{col_name}' 不存在于 DataFrame 中。可用的列: {df.columns}")
                raise ValueError(f"Feature column '{col_name}' not found.")
            df = df.withColumn(col_name, col(col_name).cast("double")) # Cast again to be safe

        # Handle potential nulls in feature columns before assembling
        # VectorAssembler by default has handleInvalid="error", so we should ensure no nulls in input.
        # Your previous processing pipeline should have already handled this.
        # If not, you might need df.na.drop(subset=feature_columns + [label_column]) here.
        # For safety and robustness, let's explicitly drop nulls in relevant columns.
        original_count = df.count()
        df_cleaned = df.na.drop(subset=feature_columns + [label_column])
        if df_cleaned.count() < original_count:
            logging.warning(f"由于特征或标签列存在空值，删除了 {original_count - df_cleaned.count()} 行数据。")
        df = df_cleaned

        # 特征组合
        # The input to VectorAssembler must not contain nulls if handleInvalid="error" (default).
        # We assume the 'normalized_MinMaxScaler' output from previous step is clean or handled by na.drop above.
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip") # "skip" will drop rows with nulls here if any still exist
        df_assembled = assembler.transform(df)
        logging.info("特征组合完成。")
        df_assembled.select("features", label_column).show(5, truncate=False)

        # PCA 降维
        # Determine 'k' based on available features
        num_features = len(feature_columns)
        pca_k = min(4, num_features) # Ensure k is not greater than the number of features
        if pca_k < num_features:
            logging.info(f"执行 PCA 降维，从 {num_features} 个特征降至 {pca_k} 个主成分。")
            pca = PCA(k=pca_k, inputCol="features", outputCol="pca_features")
            pca_model = pca.fit(df_assembled)
            df_pca = pca_model.transform(df_assembled)
            logging.info("PCA 降维完成。")
            df_pca.select("pca_features", label_column).show(5, truncate=False)
        else:
            logging.warning(f"特征数量 ({num_features}) 不足或等于 PCA 组件数 (k={pca_k})，跳过 PCA 降维。使用原始组合特征。")
            df_pca = df_assembled.withColumnRenamed("features", "pca_features") # Rename for consistency

        # 划分训练集和测试集
        train_data, test_data = df_pca.randomSplit([0.8, 0.2], seed=42)
        logging.info(f"数据划分完成。训练集行数: {train_data.count()}, 测试集行数: {test_data.count()}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"数据加载或处理失败: {e}", exc_info=True)
        raise

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
        # Filter out NaNs if any in prediction or label to avoid issues
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

def train_and_evaluate_model(model, param_grid, train_data, test_data, label_col="Ms", use_cv=True):
    """
    训练并评估模型（支持交叉验证或直接训练）。

    参数:
        model: 回归模型实例。
        param_grid: 超参数网格，若为空则不使用交叉验证。
        train_data (pyspark.sql.DataFrame): 训练集。
        test_data (pyspark.sql.DataFrame): 测试集。
        label_col (str): 标签列名，默认为 "Ms"。
        use_cv (bool): 是否使用交叉验证，默认为 True。

    返回:
        tuple: (RMSE, 准确率, 训练后的模型)
    """
    try:
        evaluator = RegressionEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="rmse"
        )
        
        best_model = None
        if use_cv and param_grid:
            logging.info(f"开始使用交叉验证训练模型: {model.__class__.__name__}")
            crossval = CrossValidator(
                estimator=model,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=3, # You can adjust the number of folds
                seed=42,
                parallelism=4 # Increase parallelism for CrossValidator if resources allow
            )
            cv_model = crossval.fit(train_data)
            best_model = cv_model.bestModel
            predictions = cv_model.transform(test_data)
            logging.info(f"交叉验证完成。最佳模型参数: {best_model.extractParamMap()}")
        else:
            logging.info(f"开始直接训练模型: {model.__class__.__name__}")
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
    # Initialize Spark
    spark = initialize_spark()

    # Data path and feature columns
    # BUG FIX: Corrected HDFS path to reflect the directory Spark saves to.
    # BUG FIX: Corrected feature_columns based on previous normalization step and column names.
    hdfs_input_dir = "hdfs://master:9000/home/data/processed_GSN_Earthquake_Catalog_normalized_MinMaxScaler"
    # Ensure these match the output of your normalization step (normalized_feature_name)
    feature_columns = [
        "normalized_震源深度(Km)",
        "normalized_Ms7",
        "normalized_mL",
        "normalized_mb8",  # Corrected from normalized_mb
        "normalized_mB9"   # Corrected from normalized_mB
    ]
    label_column = "Ms" # Assuming 'Ms' is the target label, ensure it's still available

    try:
        # Load and prepare data
        training_data, test_data = load_and_prepare_data(spark, hdfs_input_dir, feature_columns, label_column=label_column)

        if training_data.count() == 0 or test_data.count() == 0:
            logging.error("训练集或测试集为空，无法进行模型训练。请检查数据加载和清洗步骤。")
            return

        # Define models and hyperparameter grids
        # All models will use "pca_features" as input
        models_configs = [
            (
                LinearRegression(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(LinearRegression.regParam, [0.01, 0.1, 0.5])
                .addGrid(LinearRegression.elasticNetParam, [0.0, 0.5, 1.0])
                .build(),
                "Linear Regression",
                True
            ),
            (
                DecisionTreeRegressor(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(DecisionTreeRegressor.maxDepth, [5, 10, 15])
                .addGrid(DecisionTreeRegressor.minInstancesPerNode, [1, 2, 4])
                .build(),
                "Decision Tree Regression",
                True
            ),
            (
                RandomForestRegressor(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(RandomForestRegressor.numTrees, [20, 50, 100])
                .addGrid(RandomForestRegressor.maxDepth, [5, 10, 15])
                .build(),
                "Random Forest Regression",
                True
            ),
            (
                # For GBTRegressor, maxIter and maxDepth are important.
                # You might want to include them in a ParamGridBuilder for tuning.
                # For now, it's set to direct training as per your original code.
                GBTRegressor(featuresCol="pca_features", labelCol=label_column, maxIter=50, maxDepth=10),
                [],  # Empty param grid means no CrossValidator for GBTRegressor
                "GBT Regression",
                False # Set to False as param_grid is empty
            )
        ]

        # Train and evaluate models
        for model, param_grid, name, use_cv in models_configs:
            logging.info(f"\n--- 开始处理模型: {name} ---")
            rmse, accuracy, _ = train_and_evaluate_model(
                model, param_grid, training_data, test_data, label_col=label_column, use_cv=use_cv
            )
            logging.info(f"总结 - {name} RMSE: {rmse:.4f}, Accuracy: {accuracy:.4f}")

    except Exception as e:
        logging.error(f"程序执行失败: {e}", exc_info=True)
    finally:
        # Stop SparkSession
        spark.stop()
        logging.info("所有模型训练和评估完毕。SparkSession 已停止。")

if __name__ == "__main__":
    main()