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


def load_and_prepare_data(spark, hdfs_path, feature_columns, label_column="Ms7"): # Changed label_column to Ms7
    """
    加载 HDFS 数据并进行特征组合和 PCA 降维

    参数:
        spark (SparkSession): Spark 会话对象
        hdfs_path (str): HDFS 数据文件路径
        feature_columns (list): 标准化特征列名称列表
        label_column (str): 目标标签列名，默认为 "Ms7"

    返回:
        tuple: (训练集 DataFrame, 测试集 DataFrame)
    """
    try:
        # 读取数据
        # The previous script outputs the processed file within a directory.
        # Spark's read.csv automatically handles reading from a directory of part files.
        df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
        print("数据结构:")
        df.printSchema()
        df.show(5, truncate=False)

        # Cast label column to double for regression, if it's not already
        if df.schema[label_column].dataType != 'double':
            df = df.withColumn(label_column, col(label_column).cast("double"))
            print(f"将标签列 '{label_column}' 转换为 Double 类型。")
            
        # Filter out rows where the label column is null, as regression models require non-null labels
        initial_count = df.count()
        df = df.na.drop(subset=[label_column])
        if df.count() < initial_count:
            print(f"删除了 {initial_count - df.count()} 行标签为空的数据。")


        # 特征组合
        # Ensure all feature_columns exist and are numeric before assembling
        for col_name in feature_columns:
            if col_name not in df.columns:
                raise ValueError(f"Feature column '{col_name}' not found in DataFrame.")
            if df.schema[col_name].dataType not in ['double', 'float', 'integer', 'long']:
                df = df.withColumn(col_name, col(col_name).cast("double"))
                print(f"将特征列 '{col_name}' 转换为 Double 类型。")
        
        # Handle potential NaNs introduced by casting non-numeric to double
        # VectorAssembler with handleInvalid="keep" converts nulls/NaNS to NaNs in the vector,
        # but models generally can't handle NaNs directly. It's better to clean before PCA.
        initial_feature_count = df.count()
        df = df.na.drop(subset=feature_columns)
        if df.count() < initial_feature_count:
            print(f"删除了 {initial_feature_count - df.count()} 行特征为空的数据。")
            if df.count() == 0:
                raise ValueError("DataFrame became empty after dropping rows with null features.")

        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="keep")
        df_assembled = assembler.transform(df)
        print("特征组合结果:")
        df_assembled.select("features", label_column).show(5, truncate=False)

        # PCA 降维
        # It's important that features column does not contain NaNs before PCA.
        # handleInvalid="keep" in VectorAssembler helps, but often it's best to impute or drop.
        # Given the previous script drops nulls from feature_columns, this should be fine.
        pca = PCA(k=4, inputCol="features", outputCol="pca_features")
        pca_model = pca.fit(df_assembled)
        df_pca = pca_model.transform(df_assembled)
        print("PCA 降维结果:")
        df_pca.select("pca_features", label_column).show(5, truncate=False)

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


def train_and_evaluate_model(model, param_grid, train_data, test_data, label_col="Ms7"): # Changed label_col to Ms7
    """
    训练并评估模型（使用交叉验证）

    参数:
        model: 回归模型实例
        param_grid: 超参数网格
        train_data (pyspark.sql.DataFrame): 训练集
        test_data (pyspark.sql.DataFrame): 测试集
        label_col (str): 标签列名，默认为 "Ms7"

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

    # Data path and feature columns
    # The previous script generated a *directory* ending in .csv (e.g., .../your_file.csv/)
    # and inside that directory were the part-00000...csv files.
    # Spark's read.csv expects the directory path.
    hdfs_path = "hdfs://master:9000/home/data/processed_CEN_Center_Earthquake_Catalog_normalized_MinMaxScaler.csv"
    
    # Corrected feature columns based on previous script's output
    # The 'Ms' column was a feature in the original script but is used as a label here.
    # Assuming 'Ms7' is the actual label you want to predict based on the first script.
    # The first script also had 'Ms7', 'mL', 'mb8', 'mB9'. Let's adapt those.
    feature_columns = [
        "normalized_震源深度(Km)",
        "normalized_mL", # Assuming mL is present
        "normalized_mb8", # Using mb8 as per original feature list
        "normalized_mB9"  # Using mB9 as per original feature list
    ]
    label_column = "Ms7" # Explicitly define the label column

    try:
        # Load and prepare data
        training_data, test_data = load_and_prepare_data(
            spark, hdfs_path, feature_columns, label_column=label_column
        )

        if training_data.count() == 0 or test_data.count() == 0:
            print("训练集或测试集为空，无法进行模型训练。请检查数据。")
            return

        # Define models and hyperparameter grids
        models = [
            (
                LinearRegression(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(LinearRegression.regParam, [0.01, 0.1, 0.5])
                .addGrid(LinearRegression.elasticNetParam, [0.0, 0.5, 1.0])
                .build(),
                "Linear Regression",
            ),
            (
                DecisionTreeRegressor(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(DecisionTreeRegressor.maxDepth, [5, 10, 15])
                .addGrid(DecisionTreeRegressor.minInstancesPerNode, [1, 2, 4])
                .build(),
                "Decision Tree Regression",
            ),
            (
                RandomForestRegressor(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(RandomForestRegressor.numTrees, [20, 50, 100])
                .addGrid(RandomForestRegressor.maxDepth, [5, 10, 15])
                .build(),
                "Random Forest Regression",
            ),
            (
                GBTRegressor(featuresCol="pca_features", labelCol=label_column),
                ParamGridBuilder()
                .addGrid(GBTRegressor.maxIter, [20, 50, 100])
                .addGrid(GBTRegressor.maxDepth, [5, 10, 15])
                .build(),
                "GBT Regression",
            ),
        ]

        # Train and evaluate models
        for model, param_grid, name in models:
            rmse, accuracy, _ = train_and_evaluate_model(
                model, param_grid, training_data, test_data, label_col=label_column
            )
            print(f"\n--- {name} Results ---")
            print(f"RMSE: {rmse:.4f}")
            print(f"Accuracy (within {0.5} threshold): {accuracy:.4f}")

    except Exception as e:
        print(f"程序执行失败: {e}")
    finally:
        # Close SparkSession
        spark.stop()


if __name__ == "__main__":
    main()