# from pyspark.sql import SparkSession
#
# # 创建SparkSession
# spark = SparkSession.builder \
#     .appName("PCAAndRegressionExample") \
#     .getOrCreate()
#
# # 从HDFS读取处理后的数据
# hdfs_path_processed = "hdfs://hadoop101:9000/user/lhr/big_data/processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv"
# df = spark.read.csv(hdfs_path_processed, header=True, inferSchema=True)
#
# # 查看数据结构
# df.printSchema()
# df.show(5)
#
# from pyspark.ml.feature import VectorAssembler
#
# # 假设这些是你的标准化特征列
# scaled_feature_columns = ["normalized_震源深度(Km)", "normalized_Ms7", "normalized_mL","normalized_mb7","normalized_mB8"]
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
#
#
# from pyspark.ml.feature import PCA
#
# # 创建PCA模型，指定主成分数量（k）
# pca = PCA(k=4, inputCol="features", outputCol="pca_features")
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
#
# from pyspark.ml.regression import LinearRegression
# from pyspark.ml.evaluation import RegressionEvaluator
#
# # 创建线性回归模型
# lr = LinearRegression(featuresCol="pca_features", labelCol="Ms")
#
# # 划分训练集和测试集
# (training_data, test_data) = df_pca.randomSplit([0.8, 0.2])
#
# # 训练模型
# lr_model = lr.fit(training_data)
#
# # 进行预测
# lr_predictions = lr_model.transform(test_data)
#
# # 评估模型
# evaluator = RegressionEvaluator(labelCol="Ms", predictionCol="prediction", metricName="rmse")
# lr_rmse = evaluator.evaluate(lr_predictions)
# print(f"Linear Regression RMSE: {lr_rmse}")
# from pyspark.ml.regression import DecisionTreeRegressor
#
# # 创建决策树回归模型
# dt = DecisionTreeRegressor(featuresCol="pca_features", labelCol="Ms")
#
# # 训练模型
# dt_model = dt.fit(training_data)
#
# # 进行预测
# dt_predictions = dt_model.transform(test_data)
#
# # 评估模型
# dt_rmse = evaluator.evaluate(dt_predictions)
# print(f"Decision Tree Regression RMSE: {dt_rmse}")
# from pyspark.ml.regression import RandomForestRegressor
#
# # 创建随机森林回归模型
# rf = RandomForestRegressor(featuresCol="pca_features", labelCol="Ms")
#
# # 训练模型
# rf_model = rf.fit(training_data)
#
# # 进行预测
# rf_predictions = rf_model.transform(test_data)
#
# # 评估模型
# rf_rmse = evaluator.evaluate(rf_predictions)
# print(f"Random Forest Regression RMSE: {rf_rmse}")
# from pyspark.ml.regression import GBTRegressor
#
# # 创建梯度提升回归树模型
# gbt = GBTRegressor(featuresCol="pca_features", labelCol="Ms")
#
# # 训练模型
# gbt_model = gbt.fit(training_data)
#
# # 进行预测
# gbt_predictions = gbt_model.transform(test_data)
#
# # 评估模型
# gbt_rmse = evaluator.evaluate(gbt_predictions)
# print(f"GBT Regression RMSE: {gbt_rmse}")


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
import pickle

# 创建SparkSession
spark = SparkSession.builder.appName("PCAAndRegressionExample").getOrCreate()

# 从HDFS读取处理后的数据
hdfs_path_processed = "hdfs://hadoop101:9000/user/lhr/big_data/processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv"
df = spark.read.csv(hdfs_path_processed, header=True, inferSchema=True)

# 查看数据结构
df.printSchema()
df.show(5)

# 假设这些是你的标准化特征列
scaled_feature_columns = [
    "normalized_震源深度(Km)",
    "normalized_Ms7",
    "normalized_mL",
    "normalized_mb7",
    "normalized_mB8",
]

# 使用VectorAssembler将标准化特征列组合成一个向量
assembler = VectorAssembler(inputCols=scaled_feature_columns, outputCol="features")

# 将数据转换为包含"features"列的DataFrame
df_assembled = assembler.transform(df)

# 查看转换后的数据
df_assembled.select("features").show(truncate=False)

# 创建PCA模型，指定主成分数量（k）
pca = PCA(k=4, inputCol="features", outputCol="pca_features")

# 训练PCA模型
pca_model = pca.fit(df_assembled)

# 使用PCA模型对数据进行变换
df_pca = pca_model.transform(df_assembled)

# 查看PCA后的数据
df_pca.select("pca_features").show(truncate=False)

# 划分训练集和测试集
(training_data, test_data) = df_pca.randomSplit([0.8, 0.2])

# 创建回归评估器
evaluator_rmse = RegressionEvaluator(
    labelCol="Ms", predictionCol="prediction", metricName="rmse"
)


# 自定义准确率评估函数
def compute_accuracy(predictions, labelCol, predictionCol, threshold=0.5):
    predictions = predictions.withColumn(
        "correct", (col(labelCol) - col(predictionCol)).between(-threshold, threshold)
    )
    accuracy = predictions.filter(col("correct")).count() / predictions.count()
    return accuracy


# 定义一个函数来进行模型训练和评估
def train_and_evaluate(model, train_data, test_data, evaluator_rmse, threshold=0.5):
    model_fit = model.fit(train_data)
    predictions = model_fit.transform(test_data)
    rmse = evaluator_rmse.evaluate(predictions)
    accuracy = compute_accuracy(predictions, "Ms", "prediction", threshold)
    return rmse, accuracy


# 定义一个函数来进行模型训练和评估
def train_and_evaluate2(
    model, param_grid, train_data, test_data, evaluator_rmse, threshold=0.5
):
    crossval = CrossValidator(
        estimator=model,
        estimatorParamMaps=param_grid,
        evaluator=evaluator_rmse,
        numFolds=3,
    )
    cv_model = crossval.fit(train_data)
    predictions = cv_model.transform(test_data)
    rmse = evaluator_rmse.evaluate(predictions)
    accuracy = compute_accuracy(predictions, "Ms", "prediction", threshold)
    return rmse, accuracy


# 线性回归
lr = LinearRegression(featuresCol="pca_features", labelCol="Ms")
param_grid_lr = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.1, 0.5])
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
    .build()
)
lr_rmse, lr_accuracy = train_and_evaluate2(
    lr, param_grid_lr, training_data, test_data, evaluator_rmse
)
print(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")

# 决策树回归
dt = DecisionTreeRegressor(featuresCol="pca_features", labelCol="Ms")
param_grid_dt = (
    ParamGridBuilder()
    .addGrid(dt.maxDepth, [5, 10, 15])
    .addGrid(dt.minInstancesPerNode, [1, 2, 4])
    .build()
)
dt_rmse, dt_accuracy = train_and_evaluate2(
    dt, param_grid_dt, training_data, test_data, evaluator_rmse
)
print(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")

# 随机森林回归
rf = RandomForestRegressor(featuresCol="pca_features", labelCol="Ms")
param_grid_rf = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [20, 50, 100])
    .addGrid(rf.maxDepth, [5, 10, 15])
    .build()
)
rf_rmse, rf_accuracy = train_and_evaluate2(
    rf, param_grid_rf, training_data, test_data, evaluator_rmse
)
print(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")

# 梯度提升回归树
gbt = GBTRegressor(featuresCol="pca_features", labelCol="Ms", maxIter=50, maxDepth=10)
gbt_rmse, gbt_accuracy = train_and_evaluate(
    gbt, training_data, test_data, evaluator_rmse
)
print(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")

# 关闭SparkSession
spark.stop()

# from pyspark.sql import SparkSession
# from pyspark.ml.feature import VectorAssembler, PCA
# from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# from pyspark.sql.functions import col
#
# # 创建SparkSession
# spark = SparkSession.builder \
#     .appName("PCAAndRegressionExample") \
#     .getOrCreate()
#
# # 从HDFS读取处理后的数据
# hdfs_path_processed = "hdfs://hadoop101:9000/user/lhr/big_data/processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv"
# df = spark.read.csv(hdfs_path_processed, header=True, inferSchema=True)
#
# # 查看数据结构
# df.printSchema()
# df.show(5)
#
# # 假设这些是你的标准化特征列
# scaled_feature_columns = ["normalized_震源深度(Km)", "normalized_Ms7", "normalized_mL", "normalized_mb7", "normalized_mB8"]
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
# pca = PCA(k=4, inputCol="features", outputCol="pca_features")
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
# # 划分训练集和测试集
# (training_data, test_data) = df_pca.randomSplit([0.8, 0.2])
#
# # 创建回归评估器
# evaluator_rmse = RegressionEvaluator(labelCol="Ms", predictionCol="prediction", metricName="rmse")
#
# # 自定义准确率评估函数
# def compute_accuracy(predictions, labelCol, predictionCol, threshold=0.5):
#     predictions = predictions.withColumn("correct", (col(labelCol) - col(predictionCol)).between(-threshold, threshold))
#     accuracy = predictions.filter(col("correct")).count() / predictions.count()
#     return accuracy
#
# # 定义一个函数来进行模型训练和评估
# def train_and_evaluate(model, param_grid, train_data, test_data, evaluator_rmse, threshold=0.5):
#     crossval = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator_rmse, numFolds=3)
#     cv_model = crossval.fit(train_data)
#     predictions = cv_model.transform(test_data)
#     rmse = evaluator_rmse.evaluate(predictions)
#     accuracy = compute_accuracy(predictions, "Ms", "prediction", threshold)
#     return rmse, accuracy
#
# # 线性回归
# lr = LinearRegression(featuresCol="pca_features", labelCol="Ms")
# param_grid_lr = ParamGridBuilder() \
#     .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
#     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
#     .build()
# lr_rmse, lr_accuracy = train_and_evaluate(lr, param_grid_lr, training_data, test_data, evaluator_rmse)
# print(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")
#
# # 决策树回归
# dt = DecisionTreeRegressor(featuresCol="pca_features", labelCol="Ms")
# param_grid_dt = ParamGridBuilder() \
#     .addGrid(dt.maxDepth, [5, 10, 15]) \
#     .addGrid(dt.minInstancesPerNode, [1, 2, 4]) \
#     .build()
# dt_rmse, dt_accuracy = train_and_evaluate(dt, param_grid_dt, training_data, test_data, evaluator_rmse)
# print(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")
#
# # 随机森林回归
# rf = RandomForestRegressor(featuresCol="pca_features", labelCol="Ms")
# param_grid_rf = ParamGridBuilder() \
#     .addGrid(rf.numTrees, [20, 50, 100]) \
#     .addGrid(rf.maxDepth, [5, 10, 15]) \
#     .build()
# rf_rmse, rf_accuracy = train_and_evaluate(rf, param_grid_rf, training_data, test_data, evaluator_rmse)
# print(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")
#
# # 梯度提升回归树
# gbt = GBTRegressor(featuresCol="pca_features", labelCol="Ms")
# param_grid_gbt = ParamGridBuilder() \
#     .addGrid(gbt.maxIter, [20, 50, 100]) \
#     .addGrid(gbt.maxDepth, [5, 10, 15]) \
#     .build()
# gbt_rmse, gbt_accuracy = train_and_evaluate(gbt, param_grid_gbt, training_data, test_data, evaluator_rmse)
# print(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")
#
# # 关闭SparkSession
# spark.stop()
