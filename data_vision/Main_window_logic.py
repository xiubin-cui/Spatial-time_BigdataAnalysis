# -*- coding: utf-8 -*-
"""
主窗口逻辑实现，处理用户交互、文件操作、预测和数据可视化功能。
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableView, QVBoxLayout
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from .Main_window_UI_interface import Ui_MainWindow
from .Data_Visualization_UI import Ui_MainWindow as VisualizationUi
from .Confusion_matrix_heat_map import create_confusion_matrix, create_heatmap
from .config import (
    BATCH_RESULT_PATH,
    CONFUSION_MATRIX_PATH,
    TEXT_TYPES,
    VISUALIZATION_TYPES,
    VISUALIZATION_CHARTS,
)
from .utils import safe_read_csv, display_image, show_message, logger
from pathlib import Path
from typing import Optional, Tuple, List

# 图像预处理，与 Model_prediction.py 一致
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 图像分类类别，与 Model_prediction.py 的 label_map 一致
CLASS_NAMES = ["Cyclone", "Earthquake", "Flood", "Wildfire"]

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    主窗口类，继承 QMainWindow 和 UI 定义，处理用户交互和业务逻辑。
    """

    # Define _dataset_configs once at the class level
    _dataset_configs = {
        "中国地震台网地震目录数据集训练": {
            "feature_columns": ["震源深度(Km)", "mL", "mb", "mB"],
            "label_column": "Ms7",
            "description": "中国地震台网数据"
        },
        "全球地震台网地震目录数据集训练": {
            "feature_columns": ["震源深度(Km)", "Ms7", "mL", "mb", "mB"],
            "label_column": "Ms",
            "description": "全球地震台网数据"
        },
        "强震动参数数据集训练": {
            "feature_columns": [
                "震源深度", "震中距", "仪器烈度", "总峰值加速度PGA", "总峰值速度PGV",
                "参考Vs30", "东西分量PGA", "南北分量PGA", "竖向分量PGA",
                "东西分量PGV", "南北分量PGV", "竖向分量PGV"
            ],
            "label_column": "震级",
            "description": "强震动参数数据"
        }
    }

    def __init__(self):
        """
        初始化主窗口，设置 UI 并连接信号槽。
        """
        super().__init__()
        self.setupUi(self)
        self.text_file_path: str = ""
        self.image_file_path: str = ""
        self.selected_text_type: str = ""
        self.is_batch_image_predict: bool = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_model = self._load_image_model()
        self.text_model = None  # 文本模型将在 predict_text 中训练
        self._initialize_text_type_dropdown()  # 初始化下拉框选项
        self._connect_signals()
        logger.info("主窗口初始化完成")

    def _load_image_model(self) -> nn.Module:
        """
        加载 ResNet-18 模型，与 Model_prediction.py 一致。
        """
        try:
            model_path = Path("E:/CUG/Spatial_time_BigdataAnalysis/project/spatial_time_bigdata_analysis/image_process/resnet18_0.01_source.pth")
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件 '{model_path}' 不存在")
            
            model = models.resnet18(pretrained=False)
            num_classes = len(CLASS_NAMES)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            logger.info(f"成功加载图像模型: {model_path}")
            return model
        except FileNotFoundError as e:
            logger.error(f"模型文件未找到: {e}")
            show_message("错误", f"模型文件未找到: {e}", "critical")
            return None
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            show_message("错误", f"无法加载模型: {e}", "critical")
            return None

    def _connect_signals(self) -> None:
        """
        连接按钮和下拉框的信号到对应的槽函数。
        """
        self.btn_open_file.clicked.connect(self.open_text_file)
        self.btn_open_image.clicked.connect(self.open_single_image)
        self.btn_open_image_batch.clicked.connect(self.open_image_batch)
        self.btn_predict_image.clicked.connect(self.predict_image)
        self.btn_predict_text.clicked.connect(self.predict_text)
        self.cbx_text_type.currentIndexChanged.connect(self.on_text_type_changed)
        self.btn_vision_data.clicked.connect(self.open_visualization_window)

    def _initialize_text_type_dropdown(self) -> None:
        """
        初始化 cbx_text_type 下拉框，确保选项与 class-level _dataset_configs 一致。
        """
        self.cbx_text_type.clear()
        # Use the class-level _dataset_configs
        self.cbx_text_type.addItems(["文本类型选择"] + list(self._dataset_configs.keys()))
        logger.info("文本类型下拉框初始化完成")

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化 DataFrame 的列名，去除空格并统一全角/半角括号。
        """
        new_columns = []
        for col in df.columns:
            cleaned_col = str(col).strip()
            cleaned_col = cleaned_col.replace('（', '(').replace('）', ')')
            new_columns.append(cleaned_col)
        df.columns = new_columns
        return df

    def open_text_file(self) -> None:
        """
        打开文本文件选择对话框，加载并显示数据到表格。
        """
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "选择文件", str(Path.cwd()), "Text 文件 (*.txt);;CSV 文件 (*.csv)"
            )
            if file_name:
                self.text_file_path = file_name
                df = safe_read_csv(
                    file_name,
                    names=["真实值", "预测值"] if "batch_results" in file_name else None,
                )
                
                # Normalize column names immediately after reading
                df = self._normalize_column_names(df)
                logger.info(f"Loaded and normalized columns: {df.columns.tolist()}")

                model = QStandardItemModel()
                model.setRowCount(len(df))
                model.setColumnCount(len(df.columns))
                model.setHorizontalHeaderLabels(df.columns.tolist())
                for row in range(len(df)):
                    for col in range(len(df.columns)):
                        item = QStandardItem(str(df.iat[row, col]))
                        model.setItem(row, col, item)
                self.table_text_view.setModel(model)
                self.txt_text_result.setText(file_name)
                logger.info(f"成功加载文本文件: {file_name}")
        except Exception as e:
            logger.error(f"打开文本文件失败: {e}")
            show_message("错误", f"无法加载文本文件: {e}", "critical")

    def open_single_image(self) -> None:
        """
        打开单个图片选择对话框，加载并显示图片。
        """
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "选择图片", str(Path.cwd()), "图片文件 (*.png *.jpg *.bmp)"
            )
            if file_name:
                self.is_batch_image_predict = False
                self.image_file_path = file_name
                display_image(
                    file_name,
                    self.lbl_image_view,
                    (self.lbl_image_view.width(), self.lbl_image_view.height()),
                )
                self.txt_image_result.setText(file_name)
                logger.info(f"成功加载图片: {file_name}")
        except Exception as e:
            logger.error(f"打开图片失败: {e}")
            show_message("错误", f"无法加载图片: {e}", "critical")

    def open_image_batch(self) -> None:
        """
        打开文件夹选择对话框，设置批量图片预测路径。
        """
        try:
            folder_name = QFileDialog.getExistingDirectory(
                self, "选择图片文件夹", str(Path.cwd())
            )
            if folder_name:
                self.is_batch_image_predict = True
                self.image_file_path = folder_name
                self.txt_image_result.setText(folder_name)
                show_message("提示", "批量图片导入成功，请点击图像预测！")
                logger.info(f"批量图片文件夹: {folder_name}")
        except Exception as e:
            logger.error(f"打开图片文件夹失败: {e}")
            show_message("错误", f"无法选择图片文件夹: {e}", "critical")

    def _predict_single_image(self, image_path: str) -> Tuple[str, float]:
        """
        对单张图像进行预测，与 Model_prediction.py 一致。
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image = image_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.image_model(image)
                probabilities = F.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probabilities, dim=1)
                pred = CLASS_NAMES[pred_idx.item()]
                conf = conf.item()
            return pred, conf
        except Exception as e:
            logger.error(f"预测单张图像 {image_path} 失败: {e}")
            raise

    def predict_image(self) -> None:
        """
        执行图像预测，使用 ResNet-18 模型，参考 Model_prediction.py。
        """
        if not self.image_model:
            show_message("错误", "图像模型未加载！", "critical")
            return

        try:
            if not self.image_file_path:
                show_message("警告", "请先选择图片或图片文件夹！", "warning")
                return

            incorrect_file = Path("output/incorrect_images.txt")
            incorrect_file.parent.mkdir(parents=True, exist_ok=True)

            if self.is_batch_image_predict:
                # 批量预测
                image_files = [f for f in Path(self.image_file_path).glob("*.[jp][pn][gf]")]
                if not image_files:
                    show_message("警告", "图片文件夹中没有支持的图片文件！", "warning")
                    return

                predictions = []
                true_labels = []
                incorrect_paths = []
                correct = 0
                total = len(image_files)

                for image_path in image_files:
                    pred, conf = self._predict_single_image(image_path)
                    predictions.append([str(image_path), pred, conf])
                    # Assuming 'earthquake' in path means it's an earthquake image for true label
                    true_label = "Earthquake" if "earthquake" in str(image_path).lower() else "Non-Earthquake"
                    true_labels.append(true_label)
                    if pred == true_label:
                        correct += 1
                    else:
                        incorrect_paths.append(str(image_path))

                # 保存预测结果
                df = pd.DataFrame(predictions, columns=["图像路径", "预测值", "置信度"])
                df.to_csv(BATCH_RESULT_PATH, index=False, encoding="utf-8")
                accuracy = correct / total * 100 if total > 0 else 0

                # 保存错误预测路径
                if incorrect_paths:
                    with open(incorrect_file, "w", encoding="utf-8") as f:
                        for path in incorrect_paths:
                            f.write(f"{path}\n")
                    logger.info(f"错误图像路径已保存至: {incorrect_file} ({len(incorrect_paths)} 条)")
                else:
                    logger.info("没有预测错误的图像")

                self.txt_image_result.setText(str(BATCH_RESULT_PATH))
                self.txt_image_acc.setText(f"{accuracy:.2f}%")
                show_message("提示", "批量图像预测完成，结果已保存！")
                logger.info(f"批量图像预测完成: {self.image_file_path}, 准确率: {accuracy:.2f}%")
            else:
                # 单张图像预测
                pred, conf = self._predict_single_image(self.image_file_path)
                self.txt_image_result.setText(pred)
                self.txt_image_acc.setText(f"{conf:.2%}")
                show_message("提示", "单张图像预测完成！")
                logger.info(f"单张图像预测完成: {self.image_file_path}, 预测: {pred}, 置信度: {conf:.2%}")

        except Exception as e:
            logger.error(f"图像预测失败: {e}")
            show_message("错误", f"图像预测失败: {e}", "critical")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("已清理 CUDA 缓存")

    def _prepare_text_data(self, df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.Series]:
        """
        预处理文本数据，包括类型转换、缺失值处理和特征选择。
        
        Args:
            df (pd.DataFrame): 输入数据框
            config (dict): 数据集配置，包含特征列和标签列
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 处理后的特征和标签
        """
        # 验证必要列是否存在
        required_cols = config["feature_columns"] + [config["label_column"]]
        print(f"DataFrame columns: {df.columns.tolist()}")
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据集中缺少必要列: {', '.join(missing_cols)}")

        # 选择特征和标签
        X = df[config["feature_columns"]].copy()
        y = df[config["label_column"]].copy()

        # 转换为数值类型，处理非数值数据
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        # 去除包含 NaN 的行
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        if X.empty or y.empty:
            raise ValueError("数据清洗后为空，请检查文件内容！")

        return X, y

    def _train_and_predict(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, np.ndarray]:
        """
        训练随机森林模型并进行预测。
        
        Args:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标数据
        
        Returns:
            Tuple[float, np.ndarray]: RMSE 和完整数据集的预测结果
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练模型
        self.text_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.text_model.fit(X_train, y_train)
        logger.info("随机森林回归模型训练完成")

        # 预测测试集并计算 RMSE
        y_pred_test = self.text_model.predict(X_test)
        # Calculate MSE, then take the square root for RMSE
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse) # Manual RMSE calculation

        # 预测完整数据集
        y_full_pred = self.text_model.predict(X)

        return rmse, y_full_pred

    def predict_text(self) -> None:
        """
        执行文本预测，使用用户提供的数据集训练 RandomForestRegressor 模型。
        """
        try:
            if not self.text_file_path:
                show_message("警告", "请先选择文本文件！", "warning")
                return
            if self.selected_text_type == "文本类型选择":
                show_message("警告", "请选择文本类型！", "warning")
                return

            # 加载文本数据
            df = safe_read_csv(self.text_file_path)
            
            # Normalize column names immediately after reading
            df = self._normalize_column_names(df)
            logger.info(f"Loaded and normalized columns: {df.columns.tolist()}")


            # Use the class-level _dataset_configs
            if self.selected_text_type not in self._dataset_configs:
                available_types = ", ".join(self._dataset_configs.keys())
                logger.error(f"不支持的文本类型: {self.selected_text_type}, 可用类型: {available_types}")
                show_message("错误", f"不支持的文本类型: {self.selected_text_type}！可用类型: {available_types}", "critical")
                return
            config = self._dataset_configs[self.selected_text_type]

            logger.info(f"为 '{self.selected_text_type}' 数据集准备数据。期望的特征列: {config['feature_columns']}，标签列: {config['label_column']}")

            X, y = self._prepare_text_data(df, config)

            rmse, y_full_pred = self._train_and_predict(X, y)

            output_path = Path(self.text_file_path).parent / "predictions.csv"
            result_df = pd.DataFrame({config["label_column"]: y, "预测值": y_full_pred})
            result_df.to_csv(output_path, index=False, encoding="utf-8-sig")  # 使用 utf-8-sig 避免中文乱码

            # 更新 UI
            self.txt_text_result.setText(str(output_path))
            self.txt_text_acc.setText(f"{self.selected_text_type} RMSE: {rmse:.4f}")
            show_message("提示", "文本预测完成！")
            logger.info(f"文本预测完成: {self.text_file_path}, RMSE: {rmse:.4f}")

        except ValueError as ve:
            logger.error(f"数据处理错误: {ve}")
            show_message("错误", str(ve), "critical")
        except Exception as e:
            logger.error(f"文本预测失败: {e}")
            show_message("错误", f"文本预测失败: {e}", "critical")

    def on_text_type_changed(self) -> None:
        """
        文本类型下拉框选择变化时触发。
        """
        self.selected_text_type = self.cbx_text_type.currentText()
        if self.selected_text_type == "文本类型选择":
            show_message("提示", "请选择文本类型！")
        logger.info(f"选择文本类型: {self.selected_text_type}")

    def open_visualization_window(self) -> None:
        """
        打开数据可视化窗口。
        """
        try:
            self.visualization_window = VisualizationWindow(self)
            self.visualization_window.show()
            logger.info("打开数据可视化窗口")
        except Exception as e:
            logger.error(f"打开可视化窗口失败: {e}")
            show_message("错误", f"无法打开可视化窗口: {e}", "critical")

class VisualizationWindow(QMainWindow, VisualizationUi):
    """
    数据可视化窗口类，继承 QMainWindow 和 UI 定义，处理可视化逻辑。
    """

    def __init__(self, parent: Optional[QMainWindow] = None):
        """
        初始化可视化窗口，设置 UI 并连接信号槽。
        """
        super().__init__(parent)
        self.setupUi(self)
        self.selected_model: str = ""
        self.selected_text: str = ""
        self.selected_chart: str = ""
        self.web_view = QWebEngineView()
        layout = QVBoxLayout(self.view_place_widget)
        layout.addWidget(self.web_view)
        self._connect_signals()
        logger.info("可视化窗口初始化完成")

    def _connect_signals(self) -> None:
        """
        连接下拉框的信号到对应的槽函数。
        """
        self.cbx_model_type.currentIndexChanged.connect(self.on_model_changed)
        self.cbx_text_type.currentIndexChanged.connect(self.on_text_changed)
        self.cbx_chart_type.currentIndexChanged.connect(self.on_chart_changed)

    def on_model_changed(self) -> None:
        """
        可视化类型下拉框选择变化时触发，更新图表选项。
        """
        self.selected_model = self.cbx_model_type.currentText()
        self.cbx_chart_type.clear()
        if self.selected_model in VISUALIZATION_CHARTS:
            self.cbx_chart_type.addItems(
                ["可视化图选择"] + VISUALIZATION_CHARTS[self.selected_model]
            )
            self.cbx_text_type.setVisible(self.selected_model != "深度学习模型")
        else:
            self.cbx_chart_type.addItem("可视化图选择")
            show_message("提示", "请选择可视化类型！")
        logger.info(f"选择可视化类型: {self.selected_model}")

    def on_text_changed(self) -> None:
        """
        文本类型下拉框选择变化时触发。
        """
        self.selected_text = self.cbx_text_type.currentText()
        if (
            self.selected_text == "强震动参数数据集"
            and self.selected_model == "数据描述"
        ):
            self.cbx_chart_type.setVisible(False)
        else:
            self.cbx_chart_type.setVisible(True)
        logger.info(f"选择文本类型: {self.selected_text}")

    def on_chart_changed(self) -> None:
        """
        图表类型下拉框选择变化时触发，生成并显示图表。
        """
        self.selected_chart = self.cbx_chart_type.currentText()
        if self.selected_chart == "可视化图选择":
            show_message("提示", "请选择可视化图！")
            return

        try:
            if (
                self.selected_model == "深度学习模型"
                and self.selected_chart == "混淆矩阵图"
            ):
                self._render_confusion_matrix()
            else:
                show_message("提示", f"{self.selected_chart} 未实现！")
                logger.info(f"尝试渲染未实现的图表: {self.selected_chart}")
        except Exception as e:
            logger.error(f"渲染图表失败: {e}")
            show_message("错误", f"无法渲染图表: {e}", "critical")

    def _render_confusion_matrix(self) -> None:
        """
        渲染混淆矩阵热力图。
        """
        try:
            df = safe_read_csv(BATCH_RESULT_PATH, names=["图像路径", "真实值", "预测值"])
            conf_matrix, heatmap_data, labels = create_confusion_matrix(df)
            create_heatmap(
                conf_matrix, heatmap_data, labels, str(CONFUSION_MATRIX_PATH)
            )
            self.web_view.setUrl(
                QUrl.fromLocalFile(os.path.abspath(str(CONFUSION_MATRIX_PATH)))
            )
            self.web_view.setFixedSize(self.view_place_widget.size())
            logger.info(f"渲染混淆矩阵热力图: {CONFUSION_MATRIX_PATH}")
        except Exception as e:
            logger.error(f"渲染混淆矩阵失败: {e}")
            show_message("错误", f"无法渲染混淆矩阵: {e}", "critical")