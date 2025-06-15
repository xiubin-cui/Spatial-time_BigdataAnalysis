from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene, QTableView, QVBoxLayout, QWidget
from untitled import QtWidgets, Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from PyQt5.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem
from PIL import Image
import numpy as np
from PyQt5 import QtCore
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col
import data_view
from collections import Counter
from pyecharts.charts import Line, Pie, Map, WordCloud, Bar, Boxplot, HeatMap
from pyecharts import options as opts
import json
class main_Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.text_file_path = ''
        self.image_file_path = ''
        self.selected_text = ''
        self.is_batch_image_predict = 1
        # 连接按钮的点击事件到 open_file_dialog 方法
        self.open_file.clicked.connect(self.open_file_dialog)
        self.open_image.clicked.connect(self.open_image_dialog)
        self.open_image_batch.clicked.connect(self.open_image_batch_fun)
        self.predict_image.clicked.connect(self.predict_image_fun)
        self.predict_text.clicked.connect(self.predict_text_fun)
        self.text_change.currentIndexChanged.connect(self.on_combobox_changed)
        self.vision_data.clicked.connect(self.vision_data_fun)
    def open_file_dialog(self):
        # 打开文件对话框
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "所有文件 (*);;CSV 文件 (*.csv)",
                                                   options=options)
        if file_name:
            self.text_file_path = file_name
            print(self.text_file_path)
            # 读取 CSV 文件到 DataFrame
            df = pd.read_csv(file_name)

            # 创建 QStandardItemModel
            model = QStandardItemModel()
            model.setRowCount(len(df))
            model.setColumnCount(len(df.columns))

            # 设置表头
            model.setHorizontalHeaderLabels(df.columns.tolist())

            # 填充数据
            for row in range(len(df)):
                for col in range(len(df.columns)):
                    item = QStandardItem(str(df.iat[row, col]))
                    model.setItem(row, col, item)

            # 将模型设置到 QTableView
            self.text_view.setModel(model)

            # # 显示文件路径的消息框
            # QMessageBox.information(None, "选择的文件", f"选择的文件: {file_name}")
    def open_image_dialog(self):
        # 打开图片文件对话框
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(None, "选择图片", "", "图片文件 (*.png *.jpg *.bmp)",
                                                   options=options)
        if file_name:
            self.is_batch_image_predict = 0
            self.image_file_path = file_name
            print(self.image_file_path)
            # 使用 PIL 打开并转换图像
            image_path = self.image_file_path  # 图像文件路径

            # 使用 PIL 打开并转换图像
            image = Image.open(image_path).convert('RGB')

            # 获取 QLabel 的大小
            label_width, label_height = self.image_view.size().width(), self.image_view.size().height()

            # 获取图像的原始大小
            img_width, img_height = image.size

            # 计算缩放比例
            scale = min(label_width / img_width, label_height / img_height)

            # 计算缩放后的大小
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            # 缩放图像
            image = image.resize((new_width, new_height), Image.ANTIALIAS)

            # 将 PIL 图像转换为 NumPy 数组
            image_np = np.array(image)

            # 转换为 QImage
            height, width, channels = image_np.shape
            bytes_per_line = 3 * width
            qimage = QImage(image_np.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # 将 QImage 转换为 QPixmap
            pixmap = QPixmap.fromImage(qimage)

            # 设置到 QLabel 控件
            self.image_view.setPixmap(pixmap)

            # 确保 QLabel 控件自适应图片大小
            self.image_view.setScaledContents(True)
            # 显示图片路径的消息框
            # QMessageBox.information(None, "选择的图片", f"选择的图片: {file_name}")
            # # TODO: 加载和显示图片到某个控件
    def open_image_batch_fun(self):
        # 打开文件夹选择对话框
        options = QFileDialog.Options()
        folder_name = QFileDialog.getExistingDirectory(None, "选择文件夹", "", options=options)
        if folder_name:
            self.is_batch_image_predict = 1
            self.image_file_path = folder_name
            print(self.image_file_path)
            QMessageBox.information(None, "图像批量导入", "已经导入成功，请点击图像预测！")
    def predict_image_fun(self):
        if self.is_batch_image_predict == 0:
            # 设置数据路径
            data_dir = './data_source'  # 替换为你的数据路径
            self.image_result_acc.setText('85%-89%')
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Using device: {DEVICE}')

            # 定义图像预处理
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

            # 加载整个模型
            model = torch.load('./fuquqi_base_model_18_0.1_nolaji_source.pth') #BUG
            model.to(DEVICE)

            # 加载并预处理图片
            image_path = self.image_file_path  # 替换为你的图片路径
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)  # 增加批量维度

            # 进行预测
            model.eval()
            with torch.no_grad():
                image = image.to(DEVICE)
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()

            print(f'Predicted class: {predicted_class}')
            if predicted_class == 0:
                self.image_result.setText('飓风')
                print('飓风')
            elif predicted_class == 1:
                self.image_result.setText('地震')
                print("地震")
            elif predicted_class == 2:
                self.image_result.setText('洪水')
                print("洪水")
            elif predicted_class == 3:
                self.image_result.setText('野火')
                print('野火')
        if self.is_batch_image_predict == 1:
            data_dir = r'./data_source'  # 替换为你的数据路径
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Using device: {DEVICE}')

            # 数据预处理
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            model = torch.load(
                r'D:\source\python\torch_big_data\data_vision\fuquqi_base_model_18_0.1_nolaji_source.pth')
            model.to(DEVICE)
            test_dataset = datasets.ImageFolder(root=f'{self.image_file_path}', transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            def predict(model, test_loader, device):
                model.eval()
                predictions = []
                true_labels = []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        predictions.extend(preds.cpu().numpy())
                        true_labels.extend(labels.cpu().numpy())
                return true_labels, predictions

            # 进行预测
            true_labels, predictions = predict(model, test_loader, DEVICE)
            # 转换为 numpy 数组
            true_labels = np.array(true_labels)
            predictions = np.array(predictions)

            # 计算准确率
            accuracy = np.mean(true_labels == predictions)
            # 定义标签映射字典
            label_map = {
                0: "Cyclone",
                1: "Earthquake",
                2: "Flood",
                3: "Wildfire"
            }
            # 打开文件并写入数据
            with open('./深度学习模型/image_batch_results.txt', 'w', encoding='utf-8') as file:
                # 写入列标题
                file.write("真实值\t预测值\n")
                # 写入数据，替换标签值为真实名称
                for true, pred in zip(true_labels, predictions):
                    true_name = label_map.get(true, "未知")
                    pred_name = label_map.get(pred, "未知")
                    file.write(f"{true_name}\t{pred_name}\n")
            self.image_result.setText(
                'D:/source/python/torch_big_data/data_vision/深度学习模型/image_batch_results.txt')
            self.image_result_acc.setText(f'{accuracy * 100}%')

            # 输出真实值和预测值
            print("真实值:", true_labels)
            print("预测值:", predictions)
    def on_combobox_changed(self):
        self.selected_text = self.text_change.currentText()
        print(self.selected_text, type(self.selected_text))
        if self.selected_text == "文本类型选择":
            QMessageBox.information(None, "文本类型选择", "请选择你的文本类型！")
    def predict_text_fun(self):
        # 创建SparkSession
        if self.selected_text != "文本类型选择":
            predict_acc = []
            if self.selected_text == "中国地震目录":
                self.text_result.setText(f"D:/source/python/torch_big_data/data_vision/{self.selected_text}")
                # 创建SparkSession
                spark = SparkSession.builder \
                    .appName("PCAAndRegressionExample") \
                    .getOrCreate()
                os.makedirs(self.selected_text, exist_ok=True)

                # 从HDFS读取处理后的数据
                hdfs_path_processed = self.text_file_path
                df = spark.read.csv(hdfs_path_processed, header=True, inferSchema=True)

                # 查看数据结构
                df.printSchema()
                df.show(5)

                # 假设这些是你的标准化特征列
                scaled_feature_columns = ["normalized_震源深度(Km)", "normalized_Ms7", "normalized_mL",
                                          "normalized_mb7",
                                          "normalized_mB8"]

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
                evaluator_rmse = RegressionEvaluator(labelCol="Ms", predictionCol="prediction", metricName="rmse")

                # 自定义准确率评估函数
                def compute_accuracy(predictions, labelCol, predictionCol, threshold=0.5):
                    predictions = predictions.withColumn("correct",
                                                         (col(labelCol) - col(predictionCol)).between(-threshold,
                                                                                                      threshold))
                    accuracy = predictions.filter(col("correct")).count() / predictions.count()
                    return accuracy

                # 定义一个函数来进行模型训练和评估
                def train_and_evaluate(model, train_data, test_data, evaluator_rmse, threshold=0.5, output_path=None):
                    model_fit = model.fit(train_data)
                    predictions = model_fit.transform(test_data)
                    rmse = evaluator_rmse.evaluate(predictions)
                    accuracy = compute_accuracy(predictions, "Ms", "prediction", threshold)
                    selected_columns = ["发震时刻(国际时)", "经度(°)", "纬度(°)", "震源深度(Km)", "Ms7",
                                        "mL", "mb7", "mB8", "Ms", "地点", "prediction"
                                        ]
                    # 选择列
                    predictions_filtered = predictions.select(*selected_columns)
                    predictions_filtered.write.csv(f'./{self.selected_text}/{output_path}', header=True,
                                                   mode='overwrite')
                    return rmse, accuracy

                # 定义一个函数来进行模型训练和评估
                def train_and_evaluate2(model, param_grid, train_data, test_data, evaluator_rmse, threshold=0.5,
                                        output_path=None):
                    crossval = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator_rmse,
                                              numFolds=3)
                    cv_model = crossval.fit(train_data)
                    predictions = cv_model.transform(test_data)
                    print(predictions)
                    rmse = evaluator_rmse.evaluate(predictions)
                    accuracy = compute_accuracy(predictions, "Ms", "prediction", threshold)
                    selected_columns = ["发震时刻(国际时)", "经度(°)", "纬度(°)", "震源深度(Km)", "Ms7",
                                        "mL", "mb7", "mB8", "Ms", "地点", "prediction"
                                        ]

                    # 选择列
                    predictions_filtered = predictions.select(*selected_columns)
                    predictions_filtered.write.csv(f'./{self.selected_text}/{output_path}', header=True,
                                                   mode='overwrite')
                    return rmse, accuracy

                lr_predictions_path = "lr_predictions.csv"
                dt_predictions_path = "dt_predictions.csv"
                rf_predictions_path = "rf_predictions.csv"
                gbt_predictions_path = "gbt_predictions.csv"

                # 线性回归
                lr = LinearRegression(featuresCol="pca_features", labelCol="Ms")
                param_grid_lr = ParamGridBuilder() \
                    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
                    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
                    .build()
                lr_rmse, lr_accuracy = train_and_evaluate2(lr, param_grid_lr, training_data, test_data, evaluator_rmse,
                                                           output_path=lr_predictions_path)
                print(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")
                predict_acc.append(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")

                # 决策树回归
                dt = DecisionTreeRegressor(featuresCol="pca_features", labelCol="Ms")
                param_grid_dt = ParamGridBuilder() \
                    .addGrid(dt.maxDepth, [5, 10, 15]) \
                    .addGrid(dt.minInstancesPerNode, [1, 2, 4]) \
                    .build()
                dt_rmse, dt_accuracy = train_and_evaluate2(dt, param_grid_dt, training_data, test_data, evaluator_rmse,
                                                           output_path=dt_predictions_path)
                print(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")
                predict_acc.append(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")

                # 随机森林回归
                rf = RandomForestRegressor(featuresCol="pca_features", labelCol="Ms")
                param_grid_rf = ParamGridBuilder() \
                    .addGrid(rf.numTrees, [20, 50, 100]) \
                    .addGrid(rf.maxDepth, [5, 10, 15]) \
                    .build()
                rf_rmse, rf_accuracy = train_and_evaluate2(rf, param_grid_rf, training_data, test_data, evaluator_rmse,
                                                           output_path=rf_predictions_path)
                print(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")
                predict_acc.append(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")

                # 梯度提升回归树
                gbt = GBTRegressor(featuresCol="pca_features", labelCol="Ms", maxIter=50, maxDepth=10)
                gbt_rmse, gbt_accuracy = train_and_evaluate(gbt, training_data, test_data, evaluator_rmse,
                                                            output_path=gbt_predictions_path)
                print(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")
                predict_acc.append(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")
                predict_acc_str = '\n'.join(predict_acc)
                self.text_result_acc.setText(predict_acc_str)

                # 关闭SparkSession
                # spark.stop()
            elif self.selected_text == "全球地震目录":
                self.text_result.setText(f"D:/source/python/torch_big_data/data_vision/{self.selected_text}")
                # 创建SparkSession
                spark = SparkSession.builder \
                    .appName("PCAAndRegressionExample") \
                    .getOrCreate()
                os.makedirs(self.selected_text, exist_ok=True)

                # 从HDFS读取处理后的数据
                hdfs_path_processed = self.text_file_path
                df = spark.read.csv(hdfs_path_processed, header=True, inferSchema=True)

                # 查看数据结构
                df.printSchema()
                df.show(5)

                # 假设这些是你的标准化特征列
                scaled_feature_columns = ["normalized_震源深度(Km)", "normalized_Ms7", "normalized_mL",
                                          "normalized_mb7",
                                          "normalized_mB8"]

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
                evaluator_rmse = RegressionEvaluator(labelCol="Ms", predictionCol="prediction", metricName="rmse")

                # 自定义准确率评估函数
                def compute_accuracy(predictions, labelCol, predictionCol, threshold=0.5):
                    predictions = predictions.withColumn("correct",
                                                         (col(labelCol) - col(predictionCol)).between(-threshold,
                                                                                                      threshold))
                    accuracy = predictions.filter(col("correct")).count() / predictions.count()
                    return accuracy

                # 定义一个函数来进行模型训练和评估
                def train_and_evaluate(model, train_data, test_data, evaluator_rmse, threshold=0.5, output_path=None):
                    model_fit = model.fit(train_data)
                    predictions = model_fit.transform(test_data)
                    rmse = evaluator_rmse.evaluate(predictions)
                    accuracy = compute_accuracy(predictions, "Ms", "prediction", threshold)
                    selected_columns = ["发震时刻(国际时)", "经度(°)", "纬度(°)", "震源深度(Km)", "Ms7",
                                        "mL", "mb7", "mB8", "Ms", "地点", "prediction"
                                        ]
                    # 选择列
                    predictions_filtered = predictions.select(*selected_columns)
                    predictions_filtered.write.csv(f'./{self.selected_text}/{output_path}', header=True,
                                                   mode='overwrite')
                    return rmse, accuracy

                # 定义一个函数来进行模型训练和评估
                def train_and_evaluate2(model, param_grid, train_data, test_data, evaluator_rmse, threshold=0.5,
                                        output_path=None):
                    crossval = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator_rmse,
                                              numFolds=3)
                    cv_model = crossval.fit(train_data)
                    predictions = cv_model.transform(test_data)
                    print(predictions)
                    rmse = evaluator_rmse.evaluate(predictions)
                    accuracy = compute_accuracy(predictions, "Ms", "prediction", threshold)
                    selected_columns = ["发震时刻(国际时)", "经度(°)", "纬度(°)", "震源深度(Km)", "Ms7",
                                        "mL", "mb7", "mB8", "Ms", "地点", "prediction"
                                        ]
                    # 选择列
                    predictions_filtered = predictions.select(*selected_columns)
                    predictions_filtered.write.csv(f'./{self.selected_text}/{output_path}', header=True,
                                                   mode='overwrite')
                    return rmse, accuracy

                lr_predictions_path = "lr_predictions.csv"
                dt_predictions_path = "dt_predictions.csv"
                rf_predictions_path = "rf_predictions.csv"
                gbt_predictions_path = "gbt_predictions.csv"

                # 线性回归
                lr = LinearRegression(featuresCol="pca_features", labelCol="Ms")
                param_grid_lr = ParamGridBuilder() \
                    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
                    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
                    .build()
                lr_rmse, lr_accuracy = train_and_evaluate2(lr, param_grid_lr, training_data, test_data, evaluator_rmse,
                                                           output_path=lr_predictions_path)
                print(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")
                predict_acc.append(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")

                # 决策树回归
                dt = DecisionTreeRegressor(featuresCol="pca_features", labelCol="Ms")
                param_grid_dt = ParamGridBuilder() \
                    .addGrid(dt.maxDepth, [5, 10, 15]) \
                    .addGrid(dt.minInstancesPerNode, [1, 2, 4]) \
                    .build()
                dt_rmse, dt_accuracy = train_and_evaluate2(dt, param_grid_dt, training_data, test_data, evaluator_rmse,
                                                           output_path=dt_predictions_path)
                print(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")
                predict_acc.append(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")

                # 随机森林回归
                rf = RandomForestRegressor(featuresCol="pca_features", labelCol="Ms", numTrees=50, maxDepth=10)
                rf_rmse, rf_accuracy = train_and_evaluate(rf, training_data, test_data, evaluator_rmse,
                                                          output_path=rf_predictions_path)
                print(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")
                predict_acc.append(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")

                # 梯度提升回归树
                gbt = GBTRegressor(featuresCol="pca_features", labelCol="Ms", maxIter=50, maxDepth=10)
                gbt_rmse, gbt_accuracy = train_and_evaluate(gbt, training_data, test_data, evaluator_rmse,
                                                            output_path=gbt_predictions_path)
                print(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")
                predict_acc.append(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")
                predict_acc_str = '\n'.join(predict_acc)
                self.text_result_acc.setText(predict_acc_str)
            elif self.selected_text == "强震动参数数据集":
                self.text_result.setText(f"D:/source/python/torch_big_data/data_vision/{self.selected_text}")
                # 创建SparkSession
                spark = SparkSession.builder \
                    .appName("PCAAndRegressionExample") \
                    .getOrCreate()
                os.makedirs(self.selected_text, exist_ok=True)

                # 从HDFS读取处理后的数据
                hdfs_path_processed = self.text_file_path
                df = spark.read.csv(hdfs_path_processed, header=True, inferSchema=True)

                # 查看数据结构
                df.printSchema()
                df.show(5)

                # 假设这些是你的标准化特征列
                scaled_feature_columns = ["normalized_震源深度", "normalized_震中距", "normalized_仪器烈度",
                                          "normalized_总峰值加速度PGA", "normalized_总峰值速度PGV",
                                          "normalized_参考Vs30",
                                          "normalized_东西分量PGA", "normalized_南北分量PGA", "normalized_竖向分量PGA",
                                          "normalized_东西分量PGV", "normalized_南北分量PGV", "normalized_竖向分量PGV"]

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
                evaluator_rmse = RegressionEvaluator(labelCol="震级", predictionCol="prediction", metricName="rmse")

                # 自定义准确率评估函数
                def compute_accuracy(predictions, labelCol, predictionCol, threshold=0.5):
                    predictions = predictions.withColumn("correct",
                                                         (col(labelCol) - col(predictionCol)).between(-threshold,
                                                                                                      threshold))
                    accuracy = predictions.filter(col("correct")).count() / predictions.count()
                    return accuracy

                # 定义一个函数来进行模型训练和评估
                def train_and_evaluate_with_cv(model, train_data, test_data, evaluator_rmse, threshold=0.5, numFolds=3,
                                               output_path=None):
                    paramGrid = ParamGridBuilder().build()  # 不设置网格参数，仅进行交叉验证
                    crossval = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator_rmse,
                                              numFolds=numFolds)
                    cv_model = crossval.fit(train_data)
                    predictions = cv_model.transform(test_data)
                    rmse = evaluator_rmse.evaluate(predictions)
                    accuracy = compute_accuracy(predictions, "震级", "prediction", threshold)
                    selected_columns = ['事件编号', '发震时间', '震中纬度', '震中经度', '震源深度', '发震地点', '震级',
                                        '台网代码', '台站编码', '台站名称', '台站纬度', '台站经度', '震中距',
                                        '仪器烈度', '总峰值加速度PGA', '总峰值速度PGV', '场地标签', '参考Vs30',
                                        '东西分量PGA', '南北分量PGA', '竖向分量PGA', '东西分量PGV', '南北分量PGV',
                                        '竖向分量PGV', 'prediction'
                                        ]
                    # 选择列
                    predictions_filtered = predictions.select(*selected_columns)
                    predictions_filtered.write.csv(f'./{self.selected_text}/{output_path}', header=True,
                                                   mode='overwrite')
                    return rmse, accuracy

                lr_predictions_path = "lr_predictions.csv"
                dt_predictions_path = "dt_predictions.csv"
                rf_predictions_path = "rf_predictions.csv"
                gbt_predictions_path = "gbt_predictions.csv"

                # 线性回归
                lr = LinearRegression(featuresCol="pca_features", labelCol="震级", regParam=0.1, elasticNetParam=0.5)
                lr_rmse, lr_accuracy = train_and_evaluate_with_cv(lr, training_data, test_data, evaluator_rmse,
                                                                  output_path=lr_predictions_path)
                print(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")
                predict_acc.append(f"Linear Regression RMSE: {lr_rmse}, Accuracy: {lr_accuracy}")

                # 决策树回归
                dt = DecisionTreeRegressor(featuresCol="pca_features", labelCol="震级", maxDepth=10,
                                           minInstancesPerNode=2)
                dt_rmse, dt_accuracy = train_and_evaluate_with_cv(dt, training_data, test_data, evaluator_rmse,
                                                                  output_path=dt_predictions_path)
                print(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")
                predict_acc.append(f"Decision Tree Regression RMSE: {dt_rmse}, Accuracy: {dt_accuracy}")

                # 随机森林回归
                rf = RandomForestRegressor(featuresCol="pca_features", labelCol="震级", numTrees=50, maxDepth=10)
                rf_rmse, rf_accuracy = train_and_evaluate_with_cv(rf, training_data, test_data, evaluator_rmse,
                                                                  output_path=rf_predictions_path)
                print(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")
                predict_acc.append(f"Random Forest Regression RMSE: {rf_rmse}, Accuracy: {rf_accuracy}")

                # 梯度提升回归树
                gbt = GBTRegressor(featuresCol="pca_features", labelCol="震级", maxIter=50, maxDepth=10)
                gbt_rmse, gbt_accuracy = train_and_evaluate_with_cv(gbt, training_data, test_data, evaluator_rmse,
                                                                    output_path=gbt_predictions_path)
                print(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")
                predict_acc.append(f"GBT Regression RMSE: {gbt_rmse}, Accuracy: {gbt_accuracy}")
                predict_acc_str = '\n'.join(predict_acc)
                self.text_result_acc.setText(predict_acc_str)
            else:
                QMessageBox.information(None, "文本类型选择", "请选择你的文本类型！")
    def vision_data_fun(self):
        self.data_vision = data_view_window(self)
        self.data_vision.show()
        # 初始化列表
        # train_losses = [
        #     2.7587, 1.4161, 1.1737, 1.1362, 1.0965,
        #     0.9885, 0.9133, 0.8075, 0.7512, 0.6856,
        #     0.6317, 0.5827, 0.5537, 0.5053, 0.4990,
        #     0.4732, 0.4433, 0.4221, 0.4032, 0.3971,
        #     0.3641, 0.3592, 0.3699, 0.3503, 0.3586,
        #     0.3613, 0.3015, 0.2773, 0.2805, 0.2500
        # ]
        #
        # val_losses = [
        #     2.5687, 1.2598, 1.1933, 1.1373, 1.1389,
        #     1.0364, 0.9160, 0.7808, 0.7927, 1.0739,
        #     0.7208, 0.5966, 0.8929, 0.6198, 0.5763,
        #     0.6187, 0.5206, 0.5405, 0.4991, 0.5453,
        #     0.5680, 0.4734, 0.4711, 0.5792, 0.6316,
        #     0.4844, 0.4241, 0.4461, 0.4485, 0.4295
        # ]
        #
        # val_accuracies = [
        #     0.32, 0.33, 0.43, 0.43, 0.48,
        #     0.48, 0.62, 0.67, 0.68, 0.56,
        #     0.74, 0.81, 0.65, 0.79, 0.78,
        #     0.76, 0.82, 0.78, 0.83, 0.81,
        #     0.82, 0.85, 0.84, 0.83, 0.80,
        #     0.83, 0.86, 0.87, 0.87, 0.86
        # ]

        # 可视化数据功能
        # QMessageBox.information(None, "可视化数据", "可视化数据功能尚未实现。")
class data_view_window(QtWidgets.QMainWindow, data_view.Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.selected_model = ''
        self.selected_picture = ''
        self.model_change.currentIndexChanged.connect(self.model_change_fun)
        self.picture_change.currentIndexChanged.connect(self.picture_change_fun)
        self.view_text_change.currentIndexChanged.connect(self.view_text_change_fun)

        # 创建布局管理器
        layout = QVBoxLayout()
        self.view_place_widget.setLayout(layout)

        # 创建 QWebEngineView
        self.web_view = QWebEngineView()

        # 将 QWebEngineView 添加到布局管理器中
        layout.addWidget(self.web_view)

        html_file_path = r'D:\source\python\torch_big_data\data_vision\gapminder_scatter_animation.html'

        # self.web_view.setHtml(html_content)

    def view_text_change_fun(self):

        if self.view_text_change.currentText() == '强震动参数数据集':
            if self.model_change.currentText() == "数据描述":
                self.picture_change.setVisible(False)
        else:
            self.picture_change.setVisible(True)
            # self.picture_change.setVisible(False)

    def model_change_fun(self):
        self.selected_model = self.model_change.currentText()
        print(self.selected_model)
        if self.model_change.currentText() == "数据描述":
            self.view_text_change.setVisible(True)
            self.picture_change.clear()
            # 添加新的选项
            new_options = ["可视化图选择", "饼状图", "柱状图", '地图', '盒须图', '词云图', '散点图', "折线图"]
            self.picture_change.addItems(new_options)
        elif self.model_change.currentText() == "深度学习模型":
            self.view_text_change.setVisible(False)
            self.picture_change.clear()
            # 添加新的选项
            new_options = ["可视化图选择", 'train折线图', 'val折线图','混淆矩阵图']
            self.picture_change.addItems(new_options)
        elif self.model_change.currentText() == "线性回归模型":
            self.view_text_change.setVisible(True)
            self.picture_change.clear()
            # 添加新的选项
            new_options = ["可视化图选择", '折线图', '混淆图']
            self.picture_change.addItems(new_options)
        elif self.model_change.currentText() == "决策树回归模型":
            self.view_text_change.setVisible(True)
            self.picture_change.clear()
            # 添加新的选项
            new_options = ["可视化图选择", '折线图', '混淆图']
            self.picture_change.addItems(new_options)
        elif self.model_change.currentText() == "随机森林回归模型":
            self.view_text_change.setVisible(True)
            self.picture_change.clear()
            # 添加新的选项
            new_options = ["可视化图选择", '折线图', '混淆图']
            self.picture_change.addItems(new_options)
        elif self.model_change.currentText() == "梯度提升回归树模型":
            self.view_text_change.setVisible(True)
            self.picture_change.clear()
            # 添加新的选项
            new_options = ["可视化图选择", '折线图', '混淆图']
            self.picture_change.addItems(new_options)
        else:
            self.view_text_change.setVisible(True)
            self.picture_change.clear()
            # 添加新的选项
            new_options = ["可视化图选择"]
            self.picture_change.addItems(new_options)
            QMessageBox.information(None, "可视化类型选择", "请选择你的可视化类型！")

    def picture_change_fun(self):
        if self.model_change.currentText() == "数据描述":
            # if self.view_text_change.currentText() == "文本类型选择":
            #     QMessageBox.information(None, "文本类型选择", "请选择你的文本类型！")
            if self.view_text_change.currentText() == '中国地震目录':
                if self.picture_change.currentText() == "饼状图":
                    df = pd.read_csv('./processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv')
                    # 统计特定列元素的出现次数
                    column_name = '地点'  # 替换为你实际的列名
                    counts = df[column_name].value_counts()

                    # 准备数据
                    data = counts.reset_index().values.tolist()
                    data = [[item[0], item[1]] for item in data]
                    # 创建饼状图
                    pie_chart = (
                        Pie()
                        .add(
                            series_name="地点统计",
                            data_pair=data,
                            radius=["20%", "45%"],  # 设置内外半径
                            label_opts=opts.LabelOpts(formatter="{b}: {d}%")  # 设置标签格式
                        )
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="地点出现次数的饼状图", pos_left="right", pos_top="top"),
                            # 设置标题
                            legend_opts=opts.LegendOpts(orient="vertical", pos_left="left")  # 设置图例位置
                        )
                        .set_series_opts(
                            tooltip_opts=opts.TooltipOpts(
                                is_show=True,
                                trigger="item",  # 悬停触发类型
                                formatter="{b}: {c} ({d}%)"  # 悬停时显示格式
                            ),
                            label_opts=opts.LabelOpts(
                                is_show=True,
                                position="outside",  # 标签显示位置
                                formatter="{b}: {d}%"  # 标签格式
                            ),
                            center=["70%", "50%"],  # 设置饼图中心位置为图表区域的中心
                            # 动态特效
                            emphasis_opts=opts.EmphasisOpts(
                                label_opts=opts.LabelOpts(
                                    font_size=18,
                                    font_weight="bold",
                                    color="#FF6347"  # 悬停时字体颜色
                                ),
                                itemstyle_opts=opts.ItemStyleOpts(
                                    color="rgba(255, 99, 71, 0.8)"  # 悬停时扇区颜色
                                )
                            )
                        )
                    )

                    # 保存到 HTML 文件

                    html_file_path = './中国地震目录/数据描述_饼状图_html.html'

                    # 保存到 HTML 文件
                    pie_chart.render(html_file_path)
                    print(f"pie chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())

                elif self.picture_change.currentText() == '柱状图':
                    QMessageBox.information(None, "柱状图", "未实现该部分")
                elif self.picture_change.currentText() == '地图':
                    # 从 JSON 文件中读取经纬度范围数据
                    with open('./province_bounds.json', 'r', encoding='utf-8') as f:
                        province_bounds = json.load(f)

                    # 读取经纬度数据
                    df = pd.read_csv('./processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv', encoding='utf-8')

                    # 准备数据
                    def get_province(lat, lon):
                        for province, bounds in province_bounds.items():
                            if bounds["lon"][0] <= lon <= bounds["lon"][1] and bounds["lat"][0] <= lat <= bounds["lat"][
                                1]:
                                return province
                        return None

                    # 使用经纬度来推断省份
                    df['省份'] = df.apply(lambda row: get_province(row['纬度(°)'], row['经度(°)']), axis=1)
                    # 统计省份出现次数
                    province_counts = pd.Series(df['省份']).value_counts()

                    # 补充所有省份为 0 次出现数据
                    all_provinces = list(province_bounds.keys())
                    province_counts = province_counts.reindex(all_provinces, fill_value=0)

                    # 统计省份出现次数
                    data = province_counts.reset_index().values.tolist()
                    data = [[item[0], item[1]] for item in data if item[0] is not None]

                    # 创建地图
                    map_chart = (
                        Map()
                        .add(
                            series_name="省份地震次数",
                            data_pair=data,
                            maptype="china",  # 显示中国地图
                            is_map_symbol_show=False  # 不显示地图上的标记点
                        )
                        .set_global_opts(
                            # title_opts=opts.TitleOpts(title="地点出现次数地图", pos_left="right"),  # 设置标题及位置
                            visualmap_opts=opts.VisualMapOpts(
                                min_=0,
                                max_=province_counts.max(),
                                is_calculable=True,
                                orient="vertical",
                                pos_left="right",
                                pos_top="center",
                                is_piecewise=True,  # 设置分段显示
                                pieces=[
                                    {"min": 0, "max": 0, "label": "0", "color": "#E0E0E0"},
                                    {"min": 1, "max": 10, "label": "1-10", "color": "#D4E157"},
                                    {"min": 11, "max": 50, "label": "11-50", "color": "#FFC107"},
                                    {"min": 51, "max": 100, "label": "51-100", "color": "#FF5722"},
                                    {"min": 101, "max": 1000, "label": "101+", "color": "#F44336"},
                                ]
                            )
                        )
                        .set_series_opts(
                            label_opts=opts.LabelOpts(is_show=True, formatter="{b}: {c}"),
                            emphasis_opts=opts.EmphasisOpts(
                                label_opts=opts.LabelOpts(
                                    font_size=18,
                                    font_weight="bold",
                                    color="#FF6347"  # 悬停时字体颜色
                                ),
                                itemstyle_opts=opts.ItemStyleOpts(
                                    color="rgba(255, 99, 71, 0.8)"  # 悬停时区域颜色
                                )
                            )
                        )
                    )
                    html_file_path = "./中国地震目录/数据描述_地图.html"

                    # 渲染图表
                    map_chart.render(html_file_path)
                    print(f"pie chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())

                elif self.picture_change.currentText() == '盒须图':
                    QMessageBox.information(None, "盒须图", "未实现该部分")
                elif self.picture_change.currentText() == '词云图':
                    # 读取 CSV 文件
                    data = pd.read_csv('./processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv')

                    # 提取地点数据
                    locations = data['地点'].tolist()

                    # 统计地点出现频率
                    location_counts = Counter(locations)

                    # 将统计结果转化为词云数据格式
                    wordcloud_data = [{'name': location, 'value': count} for location, count in location_counts.items()]
                    # 转换为元组列表
                    wordcloud_data = [(item['name'], item['value']) for item in wordcloud_data]
                    # 创建词云图对象
                    wordcloud = WordCloud()
                    # wordcloud_data =[{'name': '中国西藏自治区', 'value': 352}, {'name': '塔吉克斯坦-中国新疆维吾尔自治区边境地区', 'value': 219}, {'name': '中国东南部', 'value': 92}, {'name': '中国东北部', 'value': 142}, {'name': '中国新疆维吾尔自治区南部', 'value': 471}, {'name': '中国台湾地区', 'value': 263}, {'name': '中国台湾', 'value': 323}, {'name': '中国甘肃省', 'value': 106}, {'name': '中国新疆维吾尔自治区北部', 'value': 241}, {'name': '中国青海省', 'value': 152}, {'name': '中国西藏自治区-印度边境地区', 'value': 34}, {'name': '中国云南省', 'value': 248}, {'name': '吉尔吉斯斯坦-中国新疆维吾尔自治区边境地区', 'value': 112}, {'name': '中国台湾东北', 'value': 25}, {'name': '中国四川省', 'value': 548}, {'name': '中国台湾东南', 'value': 5}, {'name': '哈萨克斯坦-中国新疆维吾尔自治区边境地区', 'value': 37}, {'name': '中国西藏自治区西部-印度边境地区', 'value': 8}, {'name': '俄罗斯东部-中国东北边境地区', 'value': 21}, {'name': '缅甸-中国边境地区', 'value': 129}, {'name': '中国内蒙古自治区西部', 'value': 60}, {'name': '中国东海', 'value': 23}, {'name': '克什米尔-中国新疆维吾尔自治区边境地区', 'value': 24}, {'name': '克什米尔-中国西藏自治区边境地区', 'value': 29}, {'name': '中国黄海', 'value': 12}, {'name': '中国东南沿海', 'value': 38}, {'name': '中国南海', 'value': 1}]
                    # 添加数据
                    wordcloud.add(
                        series_name='地点词云',
                        data_pair=wordcloud_data,
                        word_size_range=[10, 40],  # 设置词云中词的大小范围
                        shape='star',  # 词云形状
                        # background_color='white'  # 背景颜色
                    )

                    # 设置全局配置
                    wordcloud.set_global_opts(
                        title_opts=opts.TitleOpts(title="地震次数"),
                        tooltip_opts=opts.TooltipOpts(is_show=True, trigger="item"),  # 鼠标悬浮显示
                        visualmap_opts=opts.VisualMapOpts(max_=120, min_=15, is_show=False)  # 动态特效
                    )

                    # 渲染到本地 HTML 文件
                    # wordcloud.render('wordcloud.html')
                    html_file_path = "./中国地震目录/数据描述_词云图.html"

                    # 渲染图表
                    wordcloud.render(html_file_path)
                    print(f"pie chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())

                elif self.picture_change.currentText() == '散点图':
                    QMessageBox.information(None, "散点图", "未实现该部分")
                elif self.picture_change.currentText() == '折线图':
                    # 读取 CSV 文件
                    df = pd.read_csv('./processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv')

                    # 提取列数据
                    depth = df['震源深度(Km)'].tolist()
                    ms7 = df['Ms7'].tolist()
                    ml = df['mL'].tolist()
                    mb7 = df['mb7'].tolist()
                    mb8 = df['mB8'].tolist()
                    ms = df['Ms'].tolist()

                    # 创建折线图
                    line_chart = (
                        Line()
                        .add_xaxis(list(df.index))  # 使用 DataFrame 的索引作为 X 轴
                        .add_yaxis("震源深度(Km)", depth, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .add_yaxis("Ms7", ms7, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .add_yaxis("mL", ml, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .add_yaxis("mb7", mb7, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .add_yaxis("mB8", mb8, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .add_yaxis("Ms", ms, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="Earthquake Data Line Chart"),
                            xaxis_opts=opts.AxisOpts(name="Index"),
                            yaxis_opts=opts.AxisOpts(name="Values"),
                            datazoom_opts=opts.DataZoomOpts(),  # 添加数据缩放控件
                            tooltip_opts=opts.TooltipOpts(is_show=True)  # 显示悬停数据
                        )
                    )
                    html_file_path = './中国地震目录/数据描述_折线图_html.html'

                    # 保存到 HTML 文件
                    line_chart.render(html_file_path)
                    print(f"Line chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())

            elif self.view_text_change.currentText() == '全球地震目录':
                if self.picture_change.currentText() == "饼状图":
                    df = pd.read_csv('./processed_全球地震台网地震目录_2_1_normalized_MinMaxScaler.csv',
                                     encoding='utf-8')
                    # 统计特定列元素的出现次数
                    column_name = '地点'  # 替换为你实际的列名
                    counts = df[column_name].value_counts()
                    # 选择前15个最频繁的地点
                    top_15_counts = counts.head(20)
                    print(top_15_counts)
                    # 准备数据
                    data = top_15_counts.reset_index().values.tolist()
                    data = [[item[0], item[1]] for item in data]
                    # 创建饼状图
                    pie_chart = (
                        Pie()
                        .add(
                            series_name="地点统计",
                            data_pair=data,
                            radius=["20%", "45%"],  # 设置内外半径
                            label_opts=opts.LabelOpts(formatter="{b}: {d}%")  # 设置标签格式
                        )
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="地震次数的饼状图", pos_left="right", pos_top="top"),
                            # 设置标题
                            legend_opts=opts.LegendOpts(orient="vertical", pos_left="left")  # 设置图例位置
                        )
                        .set_series_opts(
                            tooltip_opts=opts.TooltipOpts(
                                is_show=True,
                                trigger="item",  # 悬停触发类型
                                formatter="{b}: {c} ({d}%)"  # 悬停时显示格式
                            ),
                            label_opts=opts.LabelOpts(
                                is_show=True,
                                position="outside",  # 标签显示位置
                                formatter="{b}: {d}%"  # 标签格式
                            ),
                            center=["70%", "50%"],  # 设置饼图中心位置为图表区域的中心
                            # 动态特效
                            emphasis_opts=opts.EmphasisOpts(
                                label_opts=opts.LabelOpts(
                                    font_size=18,
                                    font_weight="bold",
                                    color="#FF6347"  # 悬停时字体颜色
                                ),
                                itemstyle_opts=opts.ItemStyleOpts(
                                    color="rgba(255, 99, 71, 0.8)"  # 悬停时扇区颜色
                                )
                            )
                        )
                    )

                    # 保存到 HTML 文件

                    html_file_path = './全球地震目录/数据描述_饼状图_html.html'

                    # 保存到 HTML 文件
                    pie_chart.render(html_file_path)
                    print(f"pie chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())
                elif self.picture_change.currentText() == '柱状图':
                    QMessageBox.information(None, "柱状图", "未实现该部分")
                elif self.picture_change.currentText() == '地图':
                    QMessageBox.information(None, "地图", "未实现该部分")
                elif self.picture_change.currentText() == '盒须图':
                    QMessageBox.information(None, "盒须图", "未实现该部分")
                elif self.picture_change.currentText() == '词云图':
                    # 读取 CSV 文件
                    data = pd.read_csv('./processed_全球地震台网地震目录_2_1_normalized_MinMaxScaler.csv')

                    # 提取地点数据
                    locations = data['地点'].tolist()

                    # 统计地点出现频率
                    location_counts = Counter(locations)
                    # 选择前15个最频繁的地点
                    top_15_locations = location_counts.most_common(30)

                    # 将统计结果转化为词云数据格式
                    wordcloud_data = [{'name': location, 'value': count} for location, count in top_15_locations]
                    # 转换为元组列表
                    wordcloud_data = [(item['name'], item['value']) for item in wordcloud_data]
                    # 创建词云图对象
                    wordcloud = WordCloud()
                    # wordcloud_data =[{'name': '中国西藏自治区', 'value': 352}, {'name': '塔吉克斯坦-中国新疆维吾尔自治区边境地区', 'value': 219}, {'name': '中国东南部', 'value': 92}, {'name': '中国东北部', 'value': 142}, {'name': '中国新疆维吾尔自治区南部', 'value': 471}, {'name': '中国台湾地区', 'value': 263}, {'name': '中国台湾', 'value': 323}, {'name': '中国甘肃省', 'value': 106}, {'name': '中国新疆维吾尔自治区北部', 'value': 241}, {'name': '中国青海省', 'value': 152}, {'name': '中国西藏自治区-印度边境地区', 'value': 34}, {'name': '中国云南省', 'value': 248}, {'name': '吉尔吉斯斯坦-中国新疆维吾尔自治区边境地区', 'value': 112}, {'name': '中国台湾东北', 'value': 25}, {'name': '中国四川省', 'value': 548}, {'name': '中国台湾东南', 'value': 5}, {'name': '哈萨克斯坦-中国新疆维吾尔自治区边境地区', 'value': 37}, {'name': '中国西藏自治区西部-印度边境地区', 'value': 8}, {'name': '俄罗斯东部-中国东北边境地区', 'value': 21}, {'name': '缅甸-中国边境地区', 'value': 129}, {'name': '中国内蒙古自治区西部', 'value': 60}, {'name': '中国东海', 'value': 23}, {'name': '克什米尔-中国新疆维吾尔自治区边境地区', 'value': 24}, {'name': '克什米尔-中国西藏自治区边境地区', 'value': 29}, {'name': '中国黄海', 'value': 12}, {'name': '中国东南沿海', 'value': 38}, {'name': '中国南海', 'value': 1}]
                    # 添加数据
                    wordcloud.add(
                        series_name='地点词云',
                        data_pair=wordcloud_data,
                        word_size_range=[10, 40],  # 设置词云中词的大小范围
                        shape='star',  # 词云形状
                        # background_color='white'  # 背景颜色
                    )

                    # 设置全局配置
                    wordcloud.set_global_opts(
                        title_opts=opts.TitleOpts(title="地震次数"),
                        tooltip_opts=opts.TooltipOpts(is_show=True, trigger="item"),  # 鼠标悬浮显示
                        visualmap_opts=opts.VisualMapOpts(max_=120, min_=15, is_show=False)  # 动态特效
                    )

                    # 渲染到本地 HTML 文件
                    # wordcloud.render('wordcloud.html')
                    html_file_path = "./全球地震目录/数据描述_词云图.html"

                    # 渲染图表
                    wordcloud.render(html_file_path)
                    print(f"pie chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())

                elif self.picture_change.currentText() == '散点图':
                    QMessageBox.information(None, "散点图", "未实现该部分")
                elif self.picture_change.currentText() == '折线图':
                    # 读取 CSV 文件
                    df = pd.read_csv('./processed_全球地震台网地震目录_2_1_normalized_MinMaxScaler.csv')
                    # 提取列数据
                    depth = df['震源深度(Km)'].tolist()
                    ms7 = df['Ms7'].tolist()
                    ml = df['mL'].tolist()
                    mb7 = df['mb7'].tolist()
                    mb8 = df['mB8'].tolist()
                    ms = df['Ms'].tolist()

                    # 创建折线图
                    line_chart = (
                        Line()
                        .add_xaxis(list(df.index))  # 使用 DataFrame 的索引作为 X 轴
                        .add_yaxis("震源深度(Km)", depth, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .add_yaxis("Ms7", ms7, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .add_yaxis("mL", ml, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .add_yaxis("mb7", mb7, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .add_yaxis("mB8", mb8, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .add_yaxis("Ms", ms, is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="Earthquake Data Line Chart"),
                            xaxis_opts=opts.AxisOpts(name="Index"),
                            yaxis_opts=opts.AxisOpts(name="Values"),
                            datazoom_opts=opts.DataZoomOpts(),  # 添加数据缩放控件
                            tooltip_opts=opts.TooltipOpts(is_show=True)  # 显示悬停数据
                        )
                    )
                    html_file_path = './全球地震目录/数据描述_折线图_html.html'

                    # 保存到 HTML 文件
                    line_chart.render(html_file_path)
                    print(f"Line chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())

            elif self.view_text_change.currentText() == '强震动参数数据集':
                pass

            elif self.view_text_change.currentText() == '地震灾情数据列表':
                if self.picture_change.currentText() == "饼状图":
                    QMessageBox.information(None, "饼状图", "未实现该部分")
                elif self.picture_change.currentText() == '柱状图':
                    # 读取CSV文件
                    data = pd.read_csv('./processed_地震灾情数据列表_scend.csv')

                    # 删除死亡人数为0的行
                    data_cleaned = data[data['死亡人数'] != 0]

                    # 提取数据
                    dates = data_cleaned['发震时间(utc+8)']
                    deaths = data_cleaned['死亡人数']

                    # 创建柱状图
                    bar = (
                        Bar()
                        .add_xaxis(dates.tolist())
                        .add_yaxis("死亡人数", deaths.tolist())
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="地震死亡人数柱状图"),
                            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),  # x轴标签旋转
                            yaxis_opts=opts.AxisOpts(name="死亡人数"),
                            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),  # 鼠标悬浮显示
                            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
                        )
                    )

                    # 渲染到本地 HTML 文件
                    # bar.render("./地震灾情/earthquake_deaths_bar_chart_cleaned.html")
                    html_file_path = './地震灾情/数据描述_柱状图.html'

                    # 保存到 HTML 文件
                    bar.render(html_file_path)
                    print(f"Line chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())

                elif self.picture_change.currentText() == '地图':
                    # 读取省份边界数据
                    with open('./province_bounds.json', 'r', encoding='utf-8') as f:
                        province_bounds = json.load(f)

                    # 读取地震数据
                    df = pd.read_csv('./processed_地震灾情数据列表_scend.csv', encoding='utf-8')

                    # 准备省份数据
                    def get_province(lat, lon):
                        for province, bounds in province_bounds.items():
                            if bounds["lon"][0] <= lon <= bounds["lon"][1] and bounds["lat"][0] <= lat <= bounds["lat"][
                                1]:
                                return province
                        return None

                    # 使用经纬度推断省份
                    df['省份'] = df.apply(lambda row: get_province(row['纬度'], row['经度']), axis=1)

                    # 统计每个省份的死亡人数和经济损失
                    province_stats = df.groupby('省份').agg(
                        {'死亡人数': 'sum', '直接经济损（万元）': 'sum'}).reset_index()

                    # 补充所有省份为 0 次出现数据
                    all_provinces = list(province_bounds.keys())
                    province_stats = province_stats.set_index('省份').reindex(all_provinces, fill_value=0).reset_index()

                    # 准备地图数据
                    death_data = province_stats[['省份', '死亡人数']].values.tolist()
                    economic_data = province_stats[['省份', '直接经济损（万元）']].values.tolist()

                    # 创建地图
                    map_chart = (
                        Map()
                        .add(
                            series_name="死亡人数",
                            data_pair=death_data,
                            maptype="china",  # 显示中国地图
                            is_map_symbol_show=False,
                            label_opts=opts.LabelOpts(is_show=True),  # 不显示标签
                        )
                        # .add(
                        #     series_name="经济损失",
                        #     data_pair=economic_data,
                        #     maptype="china",  # 显示中国地图
                        #     is_map_symbol_show=False,
                        #     label_opts=opts.LabelOpts(is_show=False),  # 不显示标签
                        # )
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="省份死亡人数"),
                            visualmap_opts=[
                                opts.VisualMapOpts(
                                    series_index=0,  # 针对死亡人数系列
                                    min_=0,
                                    max_=province_stats['死亡人数'].max(),
                                    is_calculable=True,
                                    orient="vertical",
                                    pos_left="right",
                                    pos_top="center",
                                    is_piecewise=True,
                                    pieces=[
                                        {"min": 0, "max": 0, "label": "0", "color": "#E0E0E0"},
                                        {"min": 1, "max": 10, "label": "1-10", "color": "#D4E157"},
                                        {"min": 11, "max": 50, "label": "11-50", "color": "#FFC107"},
                                        {"min": 51, "max": 100, "label": "51-100", "color": "#FF5722"},
                                        {"min": 101, "max": 9999999, "label": "101+", "color": "#F44336"},
                                    ]
                                ),
                                # opts.VisualMapOpts(
                                #     series_index=1,  # 针对经济损失系列
                                #     min_=0,
                                #     max_=province_stats['直接经济损（万元）'].max(),
                                #     is_calculable=True,
                                #     orient="vertical",
                                #     pos_left="left",
                                #     pos_top="center",
                                #     is_piecewise=True,
                                #     pieces=[
                                #         {"min": 0, "max": 0, "label": "0", "color": "#E0E0E0"},
                                #         {"min": 1, "max": 1000, "label": "1-1000万", "color": "#D4E157"},
                                #         {"min": 1001, "max": 10000, "label": "1001-10000万", "color": "#FFC107"},
                                #         {"min": 10001, "max": 50000, "label": "10001-50000万", "color": "#FF5722"},
                                #         {"min": 50001, "max": 100000, "label": "50001-100000万", "color": "#F44336"},
                                #     ]
                                # )
                            ]
                        )
                        .set_series_opts(
                            # tooltip_opts=opts.TooltipOpts(
                            #     is_show=True,
                            #     formatter=(
                            #         lambda params: (
                            #             f"{params.name}<br>死亡人数: {params.value[0]}<br>经济损失: {params.value[1]}万元"
                            #         )
                            #     )
                            # ),
                            emphasis_opts=opts.EmphasisOpts(
                                label_opts=opts.LabelOpts(
                                    font_size=18,
                                    font_weight="bold",
                                    color="#FF6347"  # 悬停时字体颜色
                                ),
                                itemstyle_opts=opts.ItemStyleOpts(
                                    color="rgba(255, 99, 71, 0.8)"  # 悬停时区域颜色
                                )
                            )
                        )
                    )
                    html_file_path = "./地震灾情/数据描述_地图.html"

                    # 渲染图表
                    map_chart.render(html_file_path)
                    print(f"pie chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())

                elif self.picture_change.currentText() == '盒须图':
                    # 读取CSV文件
                    data = pd.read_csv('./processed_地震灾情数据列表_scend.csv')

                    # 删除任何包含NaN的行
                    data_cleaned = data.dropna(subset=['震级(M)', '直接经济损（万元）', '死亡人数'])

                    # 提取震级、经济损失和死亡人数数据
                    magnitudes = data_cleaned['震级(M)'].tolist()
                    economic_losses = data_cleaned['直接经济损（万元）'].tolist()
                    deaths = data_cleaned['死亡人数'].tolist()

                    # 为盒须图准备数据
                    boxplot_data = [
                        magnitudes,
                        economic_losses,
                        deaths
                    ]

                    # 创建盒须图
                    boxplot = (
                        Boxplot()
                        .add_xaxis(["震级", "经济损失", "死亡人数"])
                        .add_yaxis("数据分布", boxplot_data)
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="地震数据盒须图"),
                            yaxis_opts=opts.AxisOpts(name="数值"),
                            tooltip_opts=opts.TooltipOpts(
                                trigger="item",
                                axis_pointer_type="cross",
                                formatter="{b}<br/>最小值: {c[0]}<br/>下四分位数: {c[1]}<br/>中位数: {c[2]}<br/>上四分位数: {c[3]}<br/>最大值: {c[4]}"
                            ),
                            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
                        )
                    )

                    # 渲染到本地 HTML 文件
                    # boxplot.render("earthquake_data_boxplot.html")
                    html_file_path = './全球地震目录/数据描述_盒须图.html'

                    # 保存到 HTML 文件
                    boxplot.render(html_file_path)
                    print(f"Line chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())

                elif self.picture_change.currentText() == '词云图':
                    QMessageBox.information(None, "词云图", "未实现该部分")
                elif self.picture_change.currentText() == '散点图':
                    QMessageBox.information(None, "散点图", "未实现该部分")
                elif self.picture_change.currentText() == '折线图':
                    QMessageBox.information(None, "折线图", "未实现该部分")


        elif self.model_change.currentText() == "深度学习模型":
            if self.picture_change.currentText() == 'train折线图':
                # 训练损失数据
                train_losses = [
                    2.7587, 1.4161, 1.1737, 1.1362, 1.0965,
                    0.9885, 0.9133, 0.8075, 0.7512, 0.6856,
                    0.6317, 0.5827, 0.5537, 0.5053, 0.4990,
                    0.4732, 0.4433, 0.4221, 0.4032, 0.3971,
                    0.3641, 0.3592, 0.3699, 0.3503, 0.3586,
                    0.3613, 0.3015, 0.2773, 0.2805, 0.2500
                ]

                # 创建折线图
                line_chart = (
                    Line()
                    .add_xaxis(list(range(len(train_losses))))
                    .add_yaxis("训练损失", train_losses)
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="训练损失折线图"),
                        xaxis_opts=opts.AxisOpts(name="Epoch"),
                        yaxis_opts=opts.AxisOpts(name="损失值")
                    )
                )

                # 渲染图表
                html_file_path = './深度学习模型/train_losses.html'
                # line_chart.render("train_losses_line_chart.html")
                # 保存到 HTML 文件
                line_chart.render(html_file_path)
                print(f"Line chart saved to {html_file_path}")

                file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                self.web_view.setUrl(file_url)
                self.web_view.setFixedSize(self.view_place_widget.size())
            elif self.picture_change.currentText() == 'val折线图':
                # 验证损失值和准确率数据
                val_losses = [
                    2.5687, 1.2598, 1.1933, 1.1373, 1.1389,
                    1.0364, 0.9160, 0.7808, 0.7927, 1.0739,
                    0.7208, 0.5966, 0.8929, 0.6198, 0.5763,
                    0.6187, 0.5206, 0.5405, 0.4991, 0.5453,
                    0.5680, 0.4734, 0.4711, 0.5792, 0.6316,
                    0.4844, 0.4241, 0.4461, 0.4485, 0.4295,
                    0.4255
                ]

                val_accuracies = [
                    0.32, 0.33, 0.43, 0.43, 0.48,
                    0.48, 0.62, 0.67, 0.68, 0.56,
                    0.74, 0.81, 0.65, 0.79, 0.78,
                    0.76, 0.82, 0.78, 0.83, 0.81,
                    0.82, 0.85, 0.84, 0.83, 0.80,
                    0.83, 0.86, 0.87, 0.87, 0.86,
                    0.89
                ]

                # 创建折线图
                line_chart = (
                    Line()
                    .add_xaxis(list(range(len(val_losses))))
                    .add_yaxis("验证损失", val_losses, is_smooth=True, linestyle_opts=opts.LineStyleOpts(width=2))
                    .add_yaxis("验证准确率", val_accuracies, is_smooth=True, linestyle_opts=opts.LineStyleOpts(width=2))
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="验证损失与准确率折线图"),
                        xaxis_opts=opts.AxisOpts(name="Epoch"),
                        yaxis_opts=opts.AxisOpts(name="值", position="left"),
                        tooltip_opts=opts.TooltipOpts(
                            trigger="axis",
                            axis_pointer_type="cross",
                            textstyle_opts=opts.TextStyleOpts(font_size=14)
                        ),
                        datazoom_opts=opts.DataZoomOpts()
                    )
                )

                # 渲染图表
                # line_chart.render("val_metrics_line_chart.html")
                html_file_path = './深度学习模型/val_losses.html'
                # line_chart.render("train_losses_line_chart.html")
                # 保存到 HTML 文件
                line_chart.render(html_file_path)
                print(f"Line chart saved to {html_file_path}")

                file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                self.web_view.setUrl(file_url)
                self.web_view.setFixedSize(self.view_place_widget.size())
            elif self.picture_change.currentText() == "混淆矩阵图":
                # 1. 读取数据
                file_path = './深度学习模型/image_batch_results.txt'  # 替换为你的文件路径
                df = pd.read_csv(file_path, delimiter='\s+', skiprows=1, names=['真实值', '预测值'])

                # 获取唯一的标签
                unique_labels_1 = df['真实值'].unique()  # 获取真实值的唯一标签
                unique_labels_2 = df['预测值'].unique()  # 获取预测值的唯一标签
                labels = pd.Index(unique_labels_1).union(pd.Index(unique_labels_2))  # 合并两个唯一标签列表

                # 创建标签到索引的映射（使用循环和 i）
                label_index = {}
                for i in range(len(labels)):
                    label_index[labels[i]] = i

                # 初始化混淆矩阵
                conf_matrix = pd.DataFrame(
                    0,
                    index=labels,
                    columns=labels
                )

                # 使用 Pandas groupby 和 size 来统计数据
                grouped = df.groupby(['真实值', '预测值']).size().reset_index(name='count')

                # 填充混淆矩阵
                for _, row in grouped.iterrows():
                    true_label = row['真实值']
                    pred_label = row['预测值']
                    count = row['count']
                    conf_matrix.at[true_label, pred_label] = count

                # 2. 准备热力图数据
                heatmap_data = []
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        heatmap_data.append([i, j, conf_matrix.iloc[i, j]])

                # 将所有数据转换为整数类型
                heatmap_data_int = [[int(cell) for cell in row] for row in heatmap_data]

                # 3. 绘制混淆矩阵
                heatmap = (
                    HeatMap()
                    .add_xaxis(labels.tolist())  # x轴标签
                    .add_yaxis(
                        "混淆矩阵",
                        labels.tolist(),
                        heatmap_data_int,
                        label_opts=opts.LabelOpts(is_show=True, color='black'),  # 显示数据标签
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="混淆矩阵"),
                        xaxis_opts=opts.AxisOpts(
                            type_="category",
                            axislabel_opts=opts.LabelOpts(rotate=45)  # 旋转x轴标签
                        ),
                        yaxis_opts=opts.AxisOpts(type_="category"),
                        visualmap_opts=opts.VisualMapOpts(),
                    )
                    .set_series_opts(
                        label_opts=opts.LabelOpts(is_show=True, color='black'),  # 显示数据标签
                        itemstyle_opts=opts.ItemStyleOpts(
                            border_color='#333',
                            border_width=1
                        )
                    )
                )

                # 渲染到HTML文件
                # heatmap.render("confusion_matrix.html")
                html_file_path = './深度学习模型/混淆矩阵图.html'
                # line_chart.render("train_losses_line_chart.html")
                # 保存到 HTML 文件
                heatmap.render(html_file_path)
                print(f"Line chart saved to {html_file_path}")

                file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                self.web_view.setUrl(file_url)
                self.web_view.setFixedSize(self.view_place_widget.size())

        elif self.model_change.currentText() == "线性回归模型":
            # if self.view_text_change.currentText() == "文本类型选择":
            #     QMessageBox.information(None, "文本类型选择", "请选择你的文本类型！")
            # el
            if self.view_text_change.currentText() == '中国地震目录':
                if self.picture_change.currentText() == "可视化图选择":
                    QMessageBox.information(None, "可视化图选择", "请选择你的可视化图！")
                elif self.picture_change.currentText() == '折线图':
                    # Step 1: 创建 SparkSession
                    spark = SparkSession.builder \
                        .appName("Read CSV with PySpark") \
                        .getOrCreate()

                    # Step 2: 读取 CSV 文件
                    df = spark.read.csv(r'D:\source\python\torch_big_data\data_vision\中国地震目录\lr_predictions.csv',
                                        header=True, inferSchema=True)

                    # 显示数据（可选）
                    df.show()

                    # Step 3: 转换为 Pandas DataFrame
                    pandas_df = df.toPandas()

                    # 处理时间列，将其转换为字符串类型，以适应 pyecharts 的时间轴
                    pandas_df['发震时刻(国际时)'] = pd.to_datetime(pandas_df['发震时刻(国际时)']).dt.strftime(
                        '%Y-%m-%d %H:%M:%S')

                    # 提取数据
                    x_data = pandas_df['发震时刻(国际时)'].tolist()  # 使用时间列作为 X 轴数据
                    y_data_prediction = pandas_df['prediction'].tolist()  # 预测值
                    y_data_ms = pandas_df['Ms'].tolist()  # Ms 值作为另一条线

                    # Step 4: 使用 pyecharts 创建折线图
                    line_chart = (
                        Line()
                        .add_xaxis(x_data)  # X 轴数据（时间）
                        .add_yaxis("预测值", y_data_prediction,
                                   label_opts=opts.LabelOpts(is_show=False))  # 添加预测值的折线，并隐藏数据点标签
                        .add_yaxis("Ms 值", y_data_ms, label_opts=opts.LabelOpts(is_show=False))  # 添加 Ms 值的折线，并隐藏数据点标签
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="Ms 和 Prediction 折线图"),
                            xaxis_opts=opts.AxisOpts(name="发震时刻(国际时)", type_="time"),  # X 轴为时间类型
                            # yaxis_opts=opts.AxisOpts(name="值"),
                            tooltip_opts=opts.TooltipOpts(
                                trigger="axis",
                                axis_pointer_type="cross",
                                # formatter=(
                                #     "<div>发震时刻: {b0}</div>"
                                #     "<div>预测值: {c[0]}</div>"
                                #     "<div>Ms 值: {c[1]}</div>"
                                # )
                            ),
                            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
                        )
                    )

                    # Step 5: 渲染到本地 HTML 文件
                    # line_chart.render("ms_vs_prediction_line_chart.html")
                    html_file_path = './中国地震目录/线性回归模型_折线图.html'

                    # 保存到 HTML 文件
                    line_chart.render(html_file_path)
                    print(f"Line chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())

                elif self.picture_change.currentText() == '混淆图':
                    pass
            elif self.view_text_change.currentText() == '全球地震目录':
                if self.picture_change.currentText() == "可视化图选择":
                    QMessageBox.information(None, "可视化图选择", "请选择你的可视化图！")
                elif self.picture_change.currentText() == '折线图':
                    # Step 1: 创建 SparkSession
                    spark = SparkSession.builder \
                        .appName("Read CSV with PySpark") \
                        .getOrCreate()

                    # Step 2: 读取 CSV 文件
                    df = spark.read.csv(r'D:\source\python\torch_big_data\data_vision\全球地震目录\lr_predictions.csv',
                                        header=True, inferSchema=True)

                    # 显示数据（可选）
                    df.show()

                    # Step 3: 转换为 Pandas DataFrame
                    pandas_df = df.toPandas()

                    # 处理时间列，将其转换为字符串类型，以适应 pyecharts 的时间轴
                    pandas_df['发震时刻(国际时)'] = pd.to_datetime(pandas_df['发震时刻(国际时)']).dt.strftime(
                        '%Y-%m-%d %H:%M:%S')

                    # 提取数据
                    x_data = pandas_df['发震时刻(国际时)'].tolist()  # 使用时间列作为 X 轴数据
                    y_data_prediction = pandas_df['prediction'].tolist()  # 预测值
                    y_data_ms = pandas_df['Ms'].tolist()  # Ms 值作为另一条线

                    # Step 4: 使用 pyecharts 创建折线图
                    line_chart = (
                        Line()
                        .add_xaxis(x_data)  # X 轴数据（时间）
                        .add_yaxis("预测值", y_data_prediction,
                                   label_opts=opts.LabelOpts(is_show=False))  # 添加预测值的折线，并隐藏数据点标签
                        .add_yaxis("Ms 值", y_data_ms, label_opts=opts.LabelOpts(is_show=False))  # 添加 Ms 值的折线，并隐藏数据点标签
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="Ms 和 Prediction 折线图"),
                            xaxis_opts=opts.AxisOpts(name="发震时刻(国际时)", type_="time"),  # X 轴为时间类型
                            # yaxis_opts=opts.AxisOpts(name="值"),
                            tooltip_opts=opts.TooltipOpts(
                                trigger="axis",
                                axis_pointer_type="cross",
                                # formatter=(
                                #     "<div>发震时刻: {b0}</div>"
                                #     "<div>预测值: {c[0]}</div>"
                                #     "<div>Ms 值: {c[1]}</div>"
                                # )
                            ),
                            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
                        )
                    )

                    # Step 5: 渲染到本地 HTML 文件
                    # line_chart.render("ms_vs_prediction_line_chart.html")
                    html_file_path = './全球地震目录/线性回归模型_折线图.html'

                    # 保存到 HTML 文件
                    line_chart.render(html_file_path)
                    print(f"Line chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())
                elif self.picture_change.currentText() == '混淆图':
                    pass
            elif self.view_text_change.currentText() == '强震动参数数据集':
                if self.picture_change.currentText() == "可视化图选择":
                    QMessageBox.information(None, "可视化图选择", "请选择你的可视化图！")
                elif self.picture_change.currentText() == '折线图':
                    # Step 1: 创建 SparkSession
                    spark = SparkSession.builder \
                        .appName("Read CSV with PySpark") \
                        .getOrCreate()

                    # Step 2: 读取 CSV 文件
                    df = spark.read.csv(
                        r'D:\source\python\torch_big_data\data_vision\强震动参数数据集\lr_predictions.csv',
                        header=True, inferSchema=True)

                    # 显示数据（可选）
                    df.show()

                    # Step 3: 转换为 Pandas DataFrame
                    pandas_df = df.toPandas()

                    # 处理时间列，将其转换为字符串类型，以适应 pyecharts 的时间轴
                    pandas_df['发震时刻(国际时)'] = pd.to_datetime(pandas_df['发震时刻(国际时)']).dt.strftime(
                        '%Y-%m-%d %H:%M:%S')

                    # 提取数据
                    x_data = pandas_df['发震时刻(国际时)'].tolist()  # 使用时间列作为 X 轴数据
                    y_data_prediction = pandas_df['prediction'].tolist()  # 预测值
                    y_data_ms = pandas_df['震级'].tolist()  # Ms 值作为另一条线

                    # Step 4: 使用 pyecharts 创建折线图
                    line_chart = (
                        Line()
                        .add_xaxis(x_data)  # X 轴数据（时间）
                        .add_yaxis("预测值", y_data_prediction,
                                   label_opts=opts.LabelOpts(is_show=False))  # 添加预测值的折线，并隐藏数据点标签
                        .add_yaxis("Ms 值", y_data_ms, label_opts=opts.LabelOpts(is_show=False))  # 添加 Ms 值的折线，并隐藏数据点标签
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="Ms 和 Prediction 折线图"),
                            xaxis_opts=opts.AxisOpts(name="发震时刻(国际时)", type_="time"),  # X 轴为时间类型
                            # yaxis_opts=opts.AxisOpts(name="值"),
                            tooltip_opts=opts.TooltipOpts(
                                trigger="axis",
                                axis_pointer_type="cross",
                                # formatter=(
                                #     "<div>发震时刻: {b0}</div>"
                                #     "<div>预测值: {c[0]}</div>"
                                #     "<div>Ms 值: {c[1]}</div>"
                                # )
                            ),
                            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
                        )
                    )

                    # Step 5: 渲染到本地 HTML 文件
                    # line_chart.render("ms_vs_prediction_line_chart.html")
                    html_file_path = './强震动参数数据集/线性回归模型_折线图.html'

                    # 保存到 HTML 文件
                    line_chart.render(html_file_path)
                    print(f"Line chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())
                elif self.picture_change.currentText() == '混淆图':
                    pass

        elif self.model_change.currentText() == "决策树回归模型":
            if self.view_text_change.currentText() == '中国地震目录':
                if self.picture_change.currentText() == "可视化图选择":
                    QMessageBox.information(None, "可视化图选择", "请选择你的可视化图！")
                elif self.picture_change.currentText() == '折线图':
                    # Step 1: 创建 SparkSession
                    spark = SparkSession.builder \
                        .appName("Read CSV with PySpark") \
                        .getOrCreate()

                    # Step 2: 读取 CSV 文件
                    df = spark.read.csv(
                        r'D:\source\python\torch_big_data\data_vision\中国地震目录\dt_predictions.csv',
                        header=True, inferSchema=True)

                    # 显示数据（可选）
                    df.show()

                    # Step 3: 转换为 Pandas DataFrame
                    pandas_df = df.toPandas()

                    # 处理时间列，将其转换为字符串类型，以适应 pyecharts 的时间轴
                    pandas_df['发震时刻(国际时)'] = pd.to_datetime(pandas_df['发震时刻(国际时)']).dt.strftime(
                        '%Y-%m-%d %H:%M:%S')

                    # 提取数据
                    x_data = pandas_df['发震时刻(国际时)'].tolist()  # 使用时间列作为 X 轴数据
                    y_data_prediction = pandas_df['prediction'].tolist()  # 预测值
                    y_data_ms = pandas_df['Ms'].tolist()  # Ms 值作为另一条线

                    # Step 4: 使用 pyecharts 创建折线图
                    line_chart = (
                        Line()
                        .add_xaxis(x_data)  # X 轴数据（时间）
                        .add_yaxis("预测值", y_data_prediction,
                                   label_opts=opts.LabelOpts(is_show=False))  # 添加预测值的折线，并隐藏数据点标签
                        .add_yaxis("Ms 值", y_data_ms, label_opts=opts.LabelOpts(is_show=False))  # 添加 Ms 值的折线，并隐藏数据点标签
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="Ms 和 Prediction 折线图"),
                            xaxis_opts=opts.AxisOpts(name="发震时刻(国际时)", type_="time"),  # X 轴为时间类型
                            # yaxis_opts=opts.AxisOpts(name="值"),
                            tooltip_opts=opts.TooltipOpts(
                                trigger="axis",
                                axis_pointer_type="cross",
                                # formatter=(
                                #     "<div>发震时刻: {b0}</div>"
                                #     "<div>预测值: {c[0]}</div>"
                                #     "<div>Ms 值: {c[1]}</div>"
                                # )
                            ),
                            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
                        )
                    )

                    # Step 5: 渲染到本地 HTML 文件
                    # line_chart.render("ms_vs_prediction_line_chart.html")
                    html_file_path = './中国地震目录/决策树回归模型_折线图.html'

                    # 保存到 HTML 文件
                    line_chart.render(html_file_path)
                    print(f"Line chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())
                elif self.picture_change.currentText() == '混淆图':
                    pass
            elif self.view_text_change.currentText() == '全球地震目录':
                if self.picture_change.currentText() == "可视化图选择":
                    QMessageBox.information(None, "可视化图选择", "请选择你的可视化图！")
                elif self.picture_change.currentText() == '折线图':
                    # Step 1: 创建 SparkSession
                    spark = SparkSession.builder \
                        .appName("Read CSV with PySpark") \
                        .getOrCreate()

                    # Step 2: 读取 CSV 文件
                    df = spark.read.csv(
                        r'D:\source\python\torch_big_data\data_vision\全球地震目录\dt_predictions.csv',
                        header=True, inferSchema=True)

                    # 显示数据（可选）
                    df.show()

                    # Step 3: 转换为 Pandas DataFrame
                    pandas_df = df.toPandas()

                    # 处理时间列，将其转换为字符串类型，以适应 pyecharts 的时间轴
                    pandas_df['发震时刻(国际时)'] = pd.to_datetime(pandas_df['发震时刻(国际时)']).dt.strftime(
                        '%Y-%m-%d %H:%M:%S')

                    # 提取数据
                    x_data = pandas_df['发震时刻(国际时)'].tolist()  # 使用时间列作为 X 轴数据
                    y_data_prediction = pandas_df['prediction'].tolist()  # 预测值
                    y_data_ms = pandas_df['Ms'].tolist()  # Ms 值作为另一条线

                    # Step 4: 使用 pyecharts 创建折线图
                    line_chart = (
                        Line()
                        .add_xaxis(x_data)  # X 轴数据（时间）
                        .add_yaxis("预测值", y_data_prediction,
                                   label_opts=opts.LabelOpts(is_show=False))  # 添加预测值的折线，并隐藏数据点标签
                        .add_yaxis("Ms 值", y_data_ms, label_opts=opts.LabelOpts(is_show=False))  # 添加 Ms 值的折线，并隐藏数据点标签
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="Ms 和 Prediction 折线图"),
                            xaxis_opts=opts.AxisOpts(name="发震时刻(国际时)", type_="time"),  # X 轴为时间类型
                            # yaxis_opts=opts.AxisOpts(name="值"),
                            tooltip_opts=opts.TooltipOpts(
                                trigger="axis",
                                axis_pointer_type="cross",
                                # formatter=(
                                #     "<div>发震时刻: {b0}</div>"
                                #     "<div>预测值: {c[0]}</div>"
                                #     "<div>Ms 值: {c[1]}</div>"
                                # )
                            ),
                            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
                        )
                    )

                    # Step 5: 渲染到本地 HTML 文件
                    # line_chart.render("ms_vs_prediction_line_chart.html")
                    html_file_path = './全球地震目录/决策树回归模型_折线图.html'

                    # 保存到 HTML 文件
                    line_chart.render(html_file_path)
                    print(f"Line chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())
                elif self.picture_change.currentText() == '混淆图':
                    pass
            elif self.view_text_change.currentText() == '强震动参数数据集':
                if self.picture_change.currentText() == "可视化图选择":
                    QMessageBox.information(None, "可视化图选择", "请选择你的可视化图！")
                elif self.picture_change.currentText() == '折线图':
                    # Step 1: 创建 SparkSession
                    spark = SparkSession.builder \
                        .appName("Read CSV with PySpark") \
                        .getOrCreate()

                    # Step 2: 读取 CSV 文件
                    df = spark.read.csv(
                        r'D:\source\python\torch_big_data\data_vision\强震动参数数据集\dt_predictions.csv',
                        header=True, inferSchema=True)

                    # 显示数据（可选）
                    df.show()

                    # Step 3: 转换为 Pandas DataFrame
                    pandas_df = df.toPandas()

                    # 处理时间列，将其转换为字符串类型，以适应 pyecharts 的时间轴
                    pandas_df['发震时刻(国际时)'] = pd.to_datetime(pandas_df['发震时刻(国际时)']).dt.strftime(
                        '%Y-%m-%d %H:%M:%S')

                    # 提取数据
                    x_data = pandas_df['发震时刻(国际时)'].tolist()  # 使用时间列作为 X 轴数据
                    y_data_prediction = pandas_df['prediction'].tolist()  # 预测值
                    y_data_ms = pandas_df['震级'].tolist()  # Ms 值作为另一条线

                    # Step 4: 使用 pyecharts 创建折线图
                    line_chart = (
                        Line()
                        .add_xaxis(x_data)  # X 轴数据（时间）
                        .add_yaxis("预测值", y_data_prediction,
                                   label_opts=opts.LabelOpts(is_show=False))  # 添加预测值的折线，并隐藏数据点标签
                        .add_yaxis("Ms 值", y_data_ms, label_opts=opts.LabelOpts(is_show=False))  # 添加 Ms 值的折线，并隐藏数据点标签
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title="Ms 和 Prediction 折线图"),
                            xaxis_opts=opts.AxisOpts(name="发震时刻(国际时)", type_="time"),  # X 轴为时间类型
                            # yaxis_opts=opts.AxisOpts(name="值"),
                            tooltip_opts=opts.TooltipOpts(
                                trigger="axis",
                                axis_pointer_type="cross",
                                # formatter=(
                                #     "<div>发震时刻: {b0}</div>"
                                #     "<div>预测值: {c[0]}</div>"
                                #     "<div>Ms 值: {c[1]}</div>"
                                # )
                            ),
                            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
                        )
                    )

                    # Step 5: 渲染到本地 HTML 文件
                    # line_chart.render("ms_vs_prediction_line_chart.html")
                    html_file_path = './强震动参数数据集/决策树回归模型_折线图.html'

                    # 保存到 HTML 文件
                    line_chart.render(html_file_path)
                    print(f"Line chart saved to {html_file_path}")

                    file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))
                    self.web_view.setUrl(file_url)
                    self.web_view.setFixedSize(self.view_place_widget.size())
                elif self.picture_change.currentText() == '混淆图':
                    pass

        elif self.model_change.currentText() == "随机森林回归模型":
            if self.view_text_change.currentText() == "文本类型选择":
                QMessageBox.information(None, "文本类型选择", "请选择你的文本类型！")
            elif self.view_text_change.currentText() == '中国地震目录':
                if self.picture_change.currentText() == "可视化图选择":
                    QMessageBox.information(None, "可视化图选择", "请选择你的可视化图！")
                elif self.picture_change.currentText() == '折线图':
                    pass
                elif self.picture_change.currentText() == '混淆图':
                    pass
            elif self.view_text_change.currentText() == '全球地震目录':
                if self.picture_change.currentText() == "可视化图选择":
                    QMessageBox.information(None, "可视化图选择", "请选择你的可视化图！")
                elif self.picture_change.currentText() == '折线图':
                    pass
                elif self.picture_change.currentText() == '混淆图':
                    pass
            elif self.view_text_change.currentText() == '强震动参数数据集':
                if self.picture_change.currentText() == "可视化图选择":
                    QMessageBox.information(None, "可视化图选择", "请选择你的可视化图！")
                elif self.picture_change.currentText() == '折线图':
                    pass
                elif self.picture_change.currentText() == '混淆图':
                    pass

        elif self.model_change.currentText() == "梯度提升回归树模型":
            if self.view_text_change.currentText() == "文本类型选择":
                QMessageBox.information(None, "文本类型选择", "请选择你的文本类型！")
            elif self.view_text_change.currentText() == '中国地震目录':
                if self.picture_change.currentText() == "可视化图选择":
                    pass
                elif self.picture_change.currentText() == '折线图':
                    pass
                elif self.picture_change.currentText() == '混淆图':
                    pass
            elif self.view_text_change.currentText() == '全球地震目录':
                if self.picture_change.currentText() == "可视化图选择":
                    pass
                elif self.picture_change.currentText() == '折线图':
                    pass
                elif self.picture_change.currentText() == '混淆图':
                    pass
            elif self.view_text_change.currentText() == '强震动参数数据集':
                if self.picture_change.currentText() == "可视化图选择":
                    pass
                elif self.picture_change.currentText() == '折线图':
                    pass
                elif self.picture_change.currentText() == '混淆图':
                    pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = main_Window()  # 根据上面所创建的类的名字更改
    MainWindow.show()
    sys.exit(app.exec_())
