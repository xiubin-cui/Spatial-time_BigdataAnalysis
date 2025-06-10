# from pyspark.sql import SparkSession
# from pyspark.ml.feature import VectorAssembler, PCA
# from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.sql.functions import col
#
# # 创建SparkSession
# spark = SparkSession.builder \
#     .appName("PCAAndRegressionExample") \
#     .getOrCreate()
#
# # 从HDFS读取处理后的数据
# hdfs_path_processed = "hdfs://hadoop101:9000/user/lhr/big_data/processed_强震动参数数据集_2_1_normalized_MinMaxScaler.csv"
# df = spark.read.csv(hdfs_path_processed, header=True, inferSchema=True)
#
# # 查看数据结构
# df.printSchema()
# df.show(5)
#
# # 假设这些是你的标准化特征列
# scaled_feature_columns = ["normalized_震源深度", "normalized_震中距", "normalized_仪器烈度",
#                           "normalized_总峰值加速度PGA", "normalized_总峰值速度PGV", "normalized_参考Vs30",
#                           "normalized_东西分量PGA", "normalized_南北分量PGA", "normalized_竖向分量PGA",
#                           "normalized_东西分量PGV", "normalized_南北分量PGV", "normalized_竖向分量PGV"]
#
# # 使用VectorAssembler将标准化特征列组合成一个向量
# assembler = VectorAssembler(inputCols=scaled_feature_columns, outputCol="features")
#
# # 将数据转换为包含"features"列的DataFrame
# df_assembled = assembler.transform(df)
#
# # 查看转换后的数据
# df_assembled.select("features").show(truncate=False)
#
# # 创建PCA模型，指定主成分数量（k）
# pca = PCA(k=9, inputCol="features", outputCol="pca_features")
#
# # 训练PCA模型
# pca_model = pca.fit(df_assembled)
#
# # 使用PCA模型对数据进行变换
# df_pca = pca_model.transform(df_assembled)
#
# # 查看PCA后的数据
# df_pca.select("pca_features").show(truncate=False)
# # 对数据集进行打乱
# df_pca = df_pca.sample(withReplacement=False, fraction=1.0, seed=1234)
#
# # 划分训练集和测试集
# (training_data, test_data) = df_pca.randomSplit([0.8, 0.2], seed=1234)
# # # 划分训练集和测试集
# # (training_data, test_data) = df_pca.randomSplit([0.8, 0.2])
#
# # 创建回归评估器
# evaluator_rmse = RegressionEvaluator(labelCol="震级", predictionCol="prediction", metricName="rmse")
#
#
# # 自定义准确率评估函数
# def compute_accuracy(predictions, labelCol, predictionCol, threshold=0.5):
#     predictions = predictions.withColumn("correct", (col(labelCol) - col(predictionCol)).between(-threshold, threshold))
#     accuracy = predictions.filter(col("correct")).count() / predictions.count()
#     return accuracy
#
#
# # 定义一个函数来进行模型训练和评估
# def train_and_evaluate(model, train_data, test_data, evaluator_rmse, threshold=0.5):
#     model_fit = model.fit(train_data)
#     predictions = model_fit.transform(test_data)
#     rmse = evaluator_rmse.evaluate(predictions)
#     accuracy = compute_accuracy(predictions, "震级", "prediction", threshold)
#     return rmse, accuracy
#
#
# # 线性回归
# lr = LinearRegression(featuresCol="pca_features", labelCol="震级", regParam=0.1, elasticNetParam=0.5)
# lr_rmse, lr_accuracy = train_and_evaluate(lr, training_data, test_data, evaluator_rmse)
# print(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")
#
# # 决策树回归
# dt = DecisionTreeRegressor(featuresCol="pca_features", labelCol="震级", maxDepth=10, minInstancesPerNode=2)
# dt_rmse, dt_accuracy = train_and_evaluate(dt, training_data, test_data, evaluator_rmse)
# print(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")
#
# # 随机森林回归
# rf = RandomForestRegressor(featuresCol="pca_features", labelCol="震级", numTrees=50, maxDepth=10)
# rf_rmse, rf_accuracy = train_and_evaluate(rf, training_data, test_data, evaluator_rmse)
# print(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")
#
# # 梯度提升回归树
# gbt = GBTRegressor(featuresCol="pca_features", labelCol="震级",  maxIter=50, maxDepth=10)
# gbt_rmse, gbt_accuracy = train_and_evaluate(gbt, training_data, test_data, evaluator_rmse)
# print(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")
#
#
# # 关闭SparkSession
# spark.stop()


# from pyspark.sql import SparkSession
# from pyspark.ml.feature import VectorAssembler, PCA
# from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml.tuning import CrossValidator
# from pyspark.ml.tuning import ParamGridBuilder
# from pyspark.sql.functions import col
#
# # 创建SparkSession并调整配置
# spark = SparkSession.builder \
#     .appName("PCAAndRegressionExample") \
#     .config("spark.executor.instances", "4") \
#     .config("spark.executor.cores", "4") \
#     .config("spark.executor.memory", "8g") \
#     .config("spark.driver.memory", "8g") \
#     .getOrCreate()
#
# # 从HDFS读取处理后的数据
# hdfs_path_processed = "hdfs://hadoop101:9000/user/lhr/big_data/processed_强震动参数数据集_2_1_normalized_MinMaxScaler.csv"
# df = spark.read.csv(hdfs_path_processed, header=True, inferSchema=True)
#
# # 查看数据结构
# df.printSchema()
# df.show(5)
#
# # 假设这些是你的标准化特征列
# scaled_feature_columns = ["normalized_震源深度", "normalized_震中距", "normalized_仪器烈度",
#                           "normalized_总峰值加速度PGA", "normalized_总峰值速度PGV", "normalized_参考Vs30",
#                           "normalized_东西分量PGA", "normalized_南北分量PGA", "normalized_竖向分量PGA",
#                           "normalized_东西分量PGV", "normalized_南北分量PGV", "normalized_竖向分量PGV"]
#
# # 使用VectorAssembler将标准化特征列组合成一个向量
# assembler = VectorAssembler(inputCols=scaled_feature_columns, outputCol="features")
#
# # 将数据转换为包含"features"列的DataFrame
# df_assembled = assembler.transform(df)
#
# # 查看转换后的数据
# df_assembled.select("features").show(truncate=False)
#
# # 创建PCA模型，指定主成分数量（k）
# pca = PCA(k=9, inputCol="features", outputCol="pca_features")
#
# # 训练PCA模型
# pca_model = pca.fit(df_assembled)
#
# # 使用PCA模型对数据进行变换
# df_pca = pca_model.transform(df_assembled)
#
# # 查看PCA后的数据
# df_pca.select("pca_features").show(truncate=False)
#
# # 对数据集进行打乱
# df_pca = df_pca.sample(withReplacement=False, fraction=1.0, seed=1234)
#
# # 划分训练集和测试集
# (training_data, test_data) = df_pca.randomSplit([0.8, 0.2], seed=1234)
#
# # 创建回归评估器
# evaluator_rmse = RegressionEvaluator(labelCol="震级", predictionCol="prediction", metricName="rmse")
#
# # 自定义准确率评估函数
# def compute_accuracy(predictions, labelCol, predictionCol, threshold=0.5):
#     predictions = predictions.withColumn("correct", (col(labelCol) - col(predictionCol)).between(-threshold, threshold))
#     accuracy = predictions.filter(col("correct")).count() / predictions.count()
#     return accuracy
#
# # 定义一个函数来进行模型训练和评估
# def train_and_evaluate(model, train_data, test_data, evaluator_rmse, threshold=0.5):
#     # 设置交叉验证
#     crossval = CrossValidator(estimator=model,
#                               estimatorParamMaps=ParamGridBuilder().build(),
#                               evaluator=evaluator_rmse,
#                               numFolds=5)  # 使用5折交叉验证
#     cv_model = crossval.fit(train_data)
#     predictions = cv_model.transform(test_data)
#     rmse = evaluator_rmse.evaluate(predictions)
#     accuracy = compute_accuracy(predictions, "震级", "prediction", threshold)
#     return rmse, accuracy
#
# # # 线性回归
# # lr = LinearRegression(featuresCol="pca_features", labelCol="震级", regParam=0.1, elasticNetParam=0.5)
# # lr_rmse, lr_accuracy = train_and_evaluate(lr, training_data, test_data, evaluator_rmse)
# # print(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")
#
# # 决策树回归
# dt = DecisionTreeRegressor(featuresCol="pca_features", labelCol="震级", maxDepth=10, minInstancesPerNode=2)
# dt_rmse, dt_accuracy = train_and_evaluate(dt, training_data, test_data, evaluator_rmse)
# print(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")
#
# # 随机森林回归
# rf = RandomForestRegressor(featuresCol="pca_features", labelCol="震级", numTrees=50, maxDepth=10)
# rf_rmse, rf_accuracy = train_and_evaluate(rf, training_data, test_data, evaluator_rmse)
# print(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")
#
# # 梯度提升回归树
# gbt = GBTRegressor(featuresCol="pca_features", labelCol="震级", maxIter=50, maxDepth=10)
# gbt_rmse, gbt_accuracy = train_and_evaluate(gbt, training_data, test_data, evaluator_rmse)
# print(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")
#
# # 关闭SparkSession
# spark.stop()


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

# 创建SparkSession
spark = SparkSession.builder.appName("PCAAndRegressionExample").getOrCreate()

# 从HDFS读取处理后的数据
hdfs_path_processed = "hdfs://hadoop101:9000/user/lhr/big_data/processed_强震动参数数据集_2_1_normalized_MinMaxScaler.csv"
df = spark.read.csv(hdfs_path_processed, header=True, inferSchema=True)

# 查看数据结构
df.printSchema()
df.show(5)

# 假设这些是你的标准化特征列
scaled_feature_columns = [
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
    "normalized_竖向分量PGV",
]

# 使用VectorAssembler将标准化特征列组合成一个向量
assembler = VectorAssembler(inputCols=scaled_feature_columns, outputCol="features")

# 将数据转换为包含"features"列的DataFrame
df_assembled = assembler.transform(df)

# 查看转换后的数据
df_assembled.select("features").show(truncate=False)

# 创建PCA模型，指定主成分数量（k）
pca = PCA(k=9, inputCol="features", outputCol="pca_features")

# 训练PCA模型
pca_model = pca.fit(df_assembled)

# 使用PCA模型对数据进行变换
df_pca = pca_model.transform(df_assembled)

# 查看PCA后的数据
df_pca.select("pca_features").show(truncate=False)

# 对数据集进行打乱
df_pca = df_pca.sample(withReplacement=False, fraction=1.0, seed=1234)

# 划分训练集和测试集
(training_data, test_data) = df_pca.randomSplit([0.8, 0.2], seed=1234)

# 创建回归评估器
evaluator_rmse = RegressionEvaluator(
    labelCol="震级", predictionCol="prediction", metricName="rmse"
)


# 自定义准确率评估函数
def compute_accuracy(predictions, labelCol, predictionCol, threshold=0.5):
    predictions = predictions.withColumn(
        "correct", (col(labelCol) - col(predictionCol)).between(-threshold, threshold)
    )
    accuracy = predictions.filter(col("correct")).count() / predictions.count()
    return accuracy


# 定义一个函数来进行模型训练和评估
def train_and_evaluate_with_cv(
    model, train_data, test_data, evaluator_rmse, threshold=0.5, numFolds=3
):
    paramGrid = ParamGridBuilder().build()  # 不设置网格参数，仅进行交叉验证
    crossval = CrossValidator(
        estimator=model,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator_rmse,
        numFolds=numFolds,
    )
    cv_model = crossval.fit(train_data)
    predictions = cv_model.transform(test_data)
    rmse = evaluator_rmse.evaluate(predictions)
    accuracy = compute_accuracy(predictions, "震级", "prediction", threshold)
    return rmse, accuracy


# 线性回归
lr = LinearRegression(
    featuresCol="pca_features", labelCol="震级", regParam=0.1, elasticNetParam=0.5
)
lr_rmse, lr_accuracy = train_and_evaluate_with_cv(
    lr, training_data, test_data, evaluator_rmse
)
print(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")

# 决策树回归
dt = DecisionTreeRegressor(
    featuresCol="pca_features", labelCol="震级", maxDepth=10, minInstancesPerNode=2
)
dt_rmse, dt_accuracy = train_and_evaluate_with_cv(
    dt, training_data, test_data, evaluator_rmse
)
print(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")

# 随机森林回归
rf = RandomForestRegressor(
    featuresCol="pca_features", labelCol="震级", numTrees=50, maxDepth=10
)
rf_rmse, rf_accuracy = train_and_evaluate_with_cv(
    rf, training_data, test_data, evaluator_rmse
)
print(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")

# 梯度提升回归树
gbt = GBTRegressor(featuresCol="pca_features", labelCol="震级", maxIter=50, maxDepth=10)
gbt_rmse, gbt_accuracy = train_and_evaluate_with_cv(
    gbt, training_data, test_data, evaluator_rmse
)
print(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")

# 关闭SparkSession
spark.stop()
