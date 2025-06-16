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
from pyspark.sql.functions import col, isnan # 导入 isnan 用于检查 NaN 值
import logging

# 配置日志，以便在Spark日志中更好地查看
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 防止脚本多次运行时出现重复的handler
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def initialize_spark(app_name="PCAAndRegressionExample", hdfs_namenode="oss://cug-111.cn-beijing.oss-dls.aliyuncs.com"):
    """
    初始化一个SparkSession，并配置执行器和驱动程序资源。

    参数:
        app_name (str): Spark应用程序的名称。
        hdfs_namenode (str): HDFS NameNode地址。

    返回:
        SparkSession: 初始化后的Spark会话对象。
    """
    try:
        spark = (
            SparkSession.builder
            .appName(app_name)
            .config("spark.hadoop.fs.defaultFS", hdfs_namenode) # 配置HDFS NameNode
            .config("spark.executor.instances", "4") # 执行器实例的数量
            .config("spark.executor.cores", "4")      # 每个执行器的核心数量
            .config("spark.executor.memory", "8g")    # 每个执行器的内存
            .config("spark.driver.memory", "8g")      # 驱动程序的内存
            .config("spark.sql.shuffle.partitions", "200") # Shuffle分区数
            .getOrCreate()
        )
        logger.info(f"SparkSession '{app_name}' 成功初始化并连接到 HDFS: {hdfs_namenode}。资源已配置。")
        return spark
    except Exception as e:
        logger.error(f"初始化SparkSession失败: {e}", exc_info=True)
        raise # 如果Spark初始化失败，则重新抛出异常以停止执行


def load_and_prepare_data(spark, hdfs_path, feature_columns, label_column="Ms7", pca_k=4):
    """
    从HDFS加载数据，执行特征向量化、PCA降维、数据类型转换、处理缺失值，
    并将数据拆分为训练集和测试集。

    参数:
        spark (SparkSession): Spark会话对象。
        hdfs_path (str): 数据文件的HDFS路径。
        feature_columns (list): 特征列名称列表。
        label_column (str): 目标标签列的名称，默认为"Ms7"。
        pca_k (int): PCA的主成分数量，默认为4。

    返回:
        tuple: (训练DataFrame, 测试DataFrame)
    """
    try:
        # 读取数据 (指向目录，因为Spark写入会创建目录)
        df = spark.read.csv(hdfs_path, header=True, inferSchema=True, encoding="utf-8")
        logger.info(f"成功从 {hdfs_path} 读取数据，初始总行数: {df.count()}")
        logger.info("数据Schema:")
        df.printSchema()
        df.show(5, truncate=False)

        # 验证并转换标签列类型为Double
        if label_column not in df.columns:
            logger.error(f"错误: 标签列 '{label_column}' 不存在于DataFrame中。可用列: {df.columns}")
            raise ValueError(f"未找到标签列 '{label_column}'。")
        df = df.withColumn(label_column, col(label_column).cast("double"))
        logger.info(f"标签列 '{label_column}' 已确保为Double类型。")

        # 验证并转换所有特征列类型为Double
        for col_name in feature_columns:
            if col_name not in df.columns:
                logger.error(f"错误: 特征列 '{col_name}' 不存在于DataFrame中。可用列: {df.columns}")
                raise ValueError(f"未找到特征列 '{col_name}'。")
            df = df.withColumn(col_name, col(col_name).cast("double"))
        logger.info("所有特征列已确保为Double类型。")

        # --- 增强的缺失值处理 ---
        # 统计并记录特征和标签列中的缺失值数量
        logger.info("正在检查特征和标签列中的缺失值...")
        cols_to_check_for_nulls = feature_columns + [label_column]
        for col_name in cols_to_check_for_nulls:
            missing_count = df.filter(col(col_name).isNull() | isnan(col(col_name))).count()
            if missing_count > 0:
                logger.warning(f"列 '{col_name}' 包含 {missing_count} 个缺失 (null/NaN) 值。")
        
        original_count = df.count()
        # 删除在所有特征列或标签列中存在缺失值的行
        df_cleaned = df.na.drop(subset=cols_to_check_for_nulls)
        if df_cleaned.count() < original_count:
            dropped_rows = original_count - df_cleaned.count()
            logger.warning(f"由于特征或标签列中存在缺失值，已删除 {dropped_rows} 行。当前行数: {df_cleaned.count()}")
        else:
            logger.info("在指定特征或标签列中未发现缺失值。")
        df = df_cleaned # 从现在开始使用清理后的DataFrame

        # 检查清理后的DataFrame是否为空
        if df.count() == 0:
            logger.error("清理后的DataFrame为空，无法继续进行特征工程和模型训练。")
            return None, None

        # 特征向量化：将所有特征列合并为一个向量列
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
        df_assembled = assembler.transform(df)
        logger.info("特征向量化完成。")
        if df_assembled.count() < df.count():
            logger.warning(f"VectorAssembler 跳过了 {df.count() - df_assembled.count()} 行，因为其中包含无效（通常是null/NaN）特征值。")
        df_assembled.select("features", label_column).show(5, truncate=False)

        # PCA 降维
        actual_num_features = len(feature_columns)
        # 确保 pca_k 不超过实际特征数量，且至少为1（如果特征存在）
        if pca_k <= 0 or pca_k > actual_num_features:
            if actual_num_features > 0:
                logger.warning(f"请求的PCA主成分数量 (k={pca_k}) 无效。将 k 设置为实际特征数量 {actual_num_features}。")
                pca_k = actual_num_features
            else:
                logger.warning("特征数量为零，无法执行PCA。跳过PCA。")
                pca_k = 0 # 无法执行PCA

        if pca_k > 0:
            logger.info(f"正在执行PCA降维，从 {actual_num_features} 个特征降至 {pca_k} 个主成分。")
            # PCA类构造函数中没有 'seed' 参数
            pca = PCA(k=pca_k, inputCol="features", outputCol="pca_features")
            pca_model = pca.fit(df_assembled)
            df_pca = pca_model.transform(df_assembled)
            logger.info("PCA降维完成。")
            if hasattr(pca_model, 'explainedVariance'):
                logger.info(f"PCA解释方差比: {pca_model.explainedVariance.toArray()}")
            df_pca.select("pca_features", label_column).show(5, truncate=False)
        else:
            logger.warning("PCA主成分数量为零或特征数量为零，跳过PCA。使用原始向量化特征。")
            df_pca = df_assembled.withColumnRenamed("features", "pca_features")

        # 将数据随机拆分为训练集和测试集
        train_data, test_data = df_pca.randomSplit([0.8, 0.2], seed=1234)
        logger.info(f"数据拆分完成。训练集行数: {train_data.count()}, 测试集行数: {test_data.count()}")
        return train_data, test_data
    except Exception as e:
        logger.error(f"数据加载或处理失败: {e}", exc_info=True)
        raise


def compute_accuracy(predictions, label_col, prediction_col, threshold=0.5):
    """
    计算预测准确率（基于阈值）。准确率定义为预测值与真实值差的绝对值在给定阈值内的比例。

    参数:
        predictions (pyspark.sql.DataFrame): 包含预测结果的DataFrame。
        label_col (str): 标签列的名称。
        prediction_col (str): 预测列的名称。
        threshold (float): 误差阈值，默认为0.5。

    返回:
        float: 准确率。
    """
    try:
        # 确保用于计算的列（标签和预测）没有空值，以避免计算错误
        predictions_cleaned = predictions.na.drop(subset=[label_col, prediction_col])
        if predictions_cleaned.count() == 0:
            logger.warning("DataFrame用于准确率计算时，在删除空值后为空。返回准确率0.0。")
            return 0.0

        # 计算绝对差并检查其是否在预设的阈值范围内
        predictions_with_correct = predictions_cleaned.withColumn(
            "correct", (col(label_col) - col(prediction_col)).between(-threshold, threshold)
        )
        correct_count = predictions_with_correct.filter(col("correct")).count()
        total_count = predictions_with_correct.count()
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        logger.info(f"准确率计算完成: 正确预测数 {correct_count} / 总预测数 {total_count}")
        return accuracy
    except Exception as e:
        logger.error(f"准确率计算失败: {e}", exc_info=True)
        return 0.0


def train_and_evaluate_model(model, train_data, test_data, label_col, num_folds=3, param_grid=None):
    """
    训练和评估模型（支持交叉验证进行超参数调优，或直接训练）。

    参数:
        model: 回归模型实例 (pyspark.ml.regression 模型)。
        train_data (pyspark.sql.DataFrame): 训练数据集。
        test_data (pyspark.sql.DataFrame): 测试数据集。
        label_col (str): 标签列的名称。
        num_folds (int): 交叉验证的折叠数，默认为3。
        param_grid (list): ParamGridBuilder 构建的超参数网格列表。如果为None或空列表，则跳过交叉验证。

    返回:
        tuple: (RMSE, 准确率, 训练好的模型实例)
    """
    try:
        evaluator = RegressionEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="rmse"
        )

        best_model = None
        # 判断是否进行交叉验证：当param_grid存在且非空时进行
        if param_grid and len(param_grid) > 0:
            logger.info(f"开始使用交叉验证训练模型: {model.__class__.__name__}，进行 {num_folds} 折交叉验证。")
            crossval = CrossValidator(
                estimator=model,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=num_folds,
                seed=42, # 交叉验证的随机种子，用于结果的可复现性
                parallelism=4 # 并行度，可根据集群CPU核数调整，提高超参数搜索速度
            )
            cv_model = crossval.fit(train_data)
            best_model = cv_model.bestModel
            predictions = cv_model.transform(test_data)
            try:
                best_params = {param.name: best_model.getOrDefault(param) for param in best_model.extractParamMap()}
                logger.info(f"交叉验证完成。最佳模型参数: {best_params}")
            except Exception as e:
                logger.warning(f"无法提取最佳模型参数: {e}")
        else:
            logger.info(f"开始直接训练模型: {model.__class__.__name__} (不使用交叉验证)。")
            model_fit = model.fit(train_data)
            best_model = model_fit
            predictions = model_fit.transform(test_data)
            logger.info("模型直接训练完成。")

        rmse = evaluator.evaluate(predictions)
        accuracy = compute_accuracy(predictions, label_col, "prediction")

        logger.info(f"模型 {model.__class__.__name__} 评估结果 - RMSE: {rmse:.4f}, 准确率: {accuracy:.4f}")
        return rmse, accuracy, best_model
    except Exception as e:
        logger.error(f"模型训练或评估失败: {e}", exc_info=True)
        return float("inf"), 0.0, None


def main():
    """
    主函数：初始化Spark，加载数据，训练并评估多种回归模型。
    """
    # 初始化 SparkSession
    spark = initialize_spark()

    # 数据路径和特征列定义
    # 确保此HDFS路径正确且可访问。这是您处理后的数据输出目录。
    hdfs_path = "oss://cug-111.cn-beijing.oss-dls.aliyuncs.com/user/hadoop/input/processed_CEN_Center_Earthquake_Catalog_normalized_MinMaxScaler"
    
    # 特征列的列表
    feature_columns = [
        "normalized_震源深度(Km)",
        "normalized_mL", 
        "normalized_mb8", 
        "normalized_mB9"  
    ]
    label_column = "Ms7" # 目标标签列

    try:
        # 加载和准备数据：执行数据读取、清洗、特征工程（向量化、PCA）和数据集分割
        # pca_k 默认为 4
        training_data, test_data = load_and_prepare_data(
            spark, hdfs_path, feature_columns, label_column=label_column
        )

        # 检查训练集或测试集是否为空，如果是则终止程序，因为无法训练模型
        if training_data is None or test_data is None or training_data.count() == 0 or test_data.count() == 0:
            logger.error("训练集或测试集为空，无法继续进行模型训练。请检查数据加载和清洗步骤。")
            return

        # 定义要训练的回归模型及其超参数网格配置
        # 每个元组包含：(模型实例, ParamGridBuilder构建的超参数网格列表, 模型名称)
        models_configs = [
            (
                # 线性回归模型：构造函数没有seed参数
                LinearRegression(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(LinearRegression.regParam, [0.01, 0.1])
                .addGrid(LinearRegression.elasticNetParam, [0.0, 0.5])
                .build(),
                "Linear Regression",
            ),
            (
                # 决策树回归模型：构造函数没有seed参数
                DecisionTreeRegressor(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(DecisionTreeRegressor.maxDepth, [5, 10])
                .addGrid(DecisionTreeRegressor.minInstancesPerNode, [1, 4])
                .build(),
                "Decision Tree Regression",
            ),
            (
                # 随机森林回归模型：支持seed参数
                RandomForestRegressor(featuresCol="pca_features", labelCol=label_column, seed=42),
                ParamGridBuilder()
                .addGrid(RandomForestRegressor.numTrees, [20, 50])
                .addGrid(RandomForestRegressor.maxDepth, [5, 10])
                .build(),
                "Random Forest Regression",
            ),
            (
                # GBT回归模型：支持seed参数
                GBTRegressor(featuresCol="pca_features", labelCol=label_column, maxIter=50, maxDepth=10, seed=42),
                [],  # 提供空列表，表示不进行交叉验证，直接使用预设的maxIter和maxDepth
                "GBT Regression",
            ),
        ]

        # 遍历模型配置，依次训练和评估每个模型
        for model, param_grid, name in models_configs:
            logger.info(f"\n--- 开始处理模型: {name} ---")
            # 调用训练和评估函数，传入模型实例、训练数据、测试数据、标签列和超参数网格
            # num_folds 默认为 3
            rmse, accuracy, _ = train_and_evaluate_model(
                model, training_data, test_data, label_col=label_column, param_grid=param_grid
            )
            logger.info(f"总结 - {name} RMSE: {rmse:.4f}, 准确率: {accuracy:.4f}")

    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
    finally:
        # 确保SparkSession在程序结束时被正确关闭，释放集群资源
        if 'spark' in locals() and spark:
            spark.stop()
            logger.info("所有模型训练和评估完毕。SparkSession 已停止。")

# 2025-06-16 19:08:57,233 - __main__ - INFO - 准确率计算完成: 正确预测数 1870 / 总预测数 1978
# 2025-06-16 19:08:57,234 - __main__ - INFO - 模型 GBTRegressor 评估结果 - RMSE: 0.2622, 准确率: 0.9454
# 2025-06-16 19:08:57,234 - __main__ - INFO - 总结 - GBT Regression RMSE: 0.2622, 准确率: 0.9454

if __name__ == "__main__":
    main()