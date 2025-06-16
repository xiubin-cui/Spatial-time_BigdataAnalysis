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
from PyQt5.QtCore import QUrl, Qt # 导入Qt用于对齐
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from .Main_window_UI_interface import Ui_MainWindow
from .Data_Visualization_UI import Ui_MainWindow as VisualizationUi
from .Confusion_matrix_heat_map import create_confusion_matrix, create_heatmap
from .config import (
    BATCH_RESULT_PATH,
    CONFUSION_MATRIX_PATH,
    DATASET_CONFIGS, # 从 config 导入数据集配置
    IMAGE_MODEL_PATH, # 从 config 导入图像模型路径
    IMAGE_CLASS_NAMES, # 从 config 导入图像分类类别
    VISUALIZATION_CHARTS, # 从 config 导入可视化图表类型
    TEXT_TYPES # 从 config 导入文本类型
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

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    主窗口类，继承 QMainWindow 和 UI 定义，处理用户交互和业务逻辑。
    """

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
        加载 ResNet-18 模型。
        """
        try:
            model_path = IMAGE_MODEL_PATH # 从 config.py 获取模型路径
            if not model_path.exists():
                logger.error(f"模型文件 '{model_path}' 不存在")
                raise FileNotFoundError(f"模型文件 '{model_path}' 不存在，请检查 config.py 配置或文件路径。")
            
            model = models.resnet18(pretrained=False)
            num_classes = len(IMAGE_CLASS_NAMES) # 从 config.py 获取类别名称
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval() # 设置为评估模式
            logger.info(f"成功加载图像模型: {model_path}")
            return model
        except FileNotFoundError as e:
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
        初始化 cbx_text_type 下拉框，确保选项与 config.py 中的 DATASET_CONFIGS 一致。
        """
        self.cbx_text_type.clear()
        self.cbx_text_type.addItems(["文本类型选择"] + list(DATASET_CONFIGS.keys()))
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
                self, "选择文件", str(Path.cwd()), "CSV 文件 (*.csv);;Text 文件 (*.txt);;所有文件 (*.*)"
            )
            if file_name:
                self.text_file_path = file_name
                # 对于带有头部的一般CSV文件，不要传递 'names' 参数，让 pandas 自动推断头部
                # safe_read_csv 会处理编码和分隔符推断
                df = safe_read_csv(file_name) 
                
                df = self._normalize_column_names(df)
                logger.info(f"Loaded and normalized columns: {df.columns.tolist()}")

                model = QStandardItemModel()
                model.setRowCount(len(df))
                model.setColumnCount(len(df.columns))
                model.setHorizontalHeaderLabels(df.columns.tolist())
                for row in range(len(df)):
                    for col in range(len(df.columns)):
                        item = QStandardItem(str(df.iat[row, col]))
                        item.setTextAlignment(Qt.AlignCenter) # 居中对齐
                        model.setItem(row, col, item)
                self.table_text_view.setModel(model)
                self.txt_text_result.setText(file_name)
                logger.info(f"成功加载文本文件: {file_name}")
        except FileNotFoundError as e:
            show_message("错误", str(e), "critical")
        except pd.errors.EmptyDataError:
            show_message("警告", "选择的文件是空的。", "warning")
        except Exception as e:
            logger.error(f"打开文本文件失败: {e}")
            show_message("错误", f"无法加载文本文件: {e}", "critical")

    def open_single_image(self) -> None:
        """
        打开单个图片选择对话框，加载并显示图片。
        """
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "选择图片", str(Path.cwd()), "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
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
        对单张图像进行预测。
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image = image_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.image_model(image)
                probabilities = F.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probabilities, dim=1)
                pred = IMAGE_CLASS_NAMES[pred_idx.item()]
                conf = conf.item()
            return pred, conf
        except Exception as e:
            logger.error(f"预测单张图像 {image_path} 失败: {e}")
            # 重新抛出异常，让上层函数捕获并显示错误
            raise

    def predict_image(self) -> None:
        """
        执行图像预测，使用 ResNet-18 模型。
        """
        if not self.image_model:
            show_message("错误", "图像模型未加载！请检查模型文件是否存在。", "critical")
            return

        try:
            if not self.image_file_path:
                show_message("警告", "请先选择图片或图片文件夹！", "warning")
                return

            incorrect_file_path = Path(CONFUSION_MATRIX_PATH).parent / "incorrect_image_predictions.csv" # 存储错误的预测
            incorrect_file_path.parent.mkdir(parents=True, exist_ok=True)


            if self.is_batch_image_predict:
                # 批量预测
                # 兼容多种图片格式
                image_files = []
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]:
                    image_files.extend(list(Path(self.image_file_path).glob(ext)))
                
                if not image_files:
                    show_message("警告", "图片文件夹中没有支持的图片文件！支持 .png, .jpg, .jpeg, .gif, .bmp", "warning")
                    return

                predictions_list = []
                incorrect_predictions = [] # 用于存储错误预测的详细信息

                for image_path in image_files:
                    try:
                        pred, conf = self._predict_single_image(image_path)
                        
                        # 从文件名或路径中推断真实标签，这里需要根据实际数据命名约定来调整
                        # 示例：如果文件名包含 'earthquake', 则认为是 'Earthquake'
                        # 否则，默认为 'Unknown' 或其他非地震类别
                        true_label_inferred = "Unknown" 
                        for class_name in IMAGE_CLASS_NAMES:
                            if class_name.lower() in str(image_path).lower():
                                true_label_inferred = class_name
                                break

                        predictions_list.append([str(image_path), true_label_inferred, pred, conf])
                        
                        if pred != true_label_inferred:
                            incorrect_predictions.append([str(image_path), true_label_inferred, pred, conf])

                    except Exception as e:
                        logger.warning(f"处理图片 {image_path} 时发生错误，跳过: {e}")
                        predictions_list.append([str(image_path), "N/A", "Error", "N/A"]) # 记录错误图片

                # 保存所有预测结果
                df_predictions = pd.DataFrame(predictions_list, columns=["图像路径", "真实标签(推断)", "预测值", "置信度"])
                df_predictions.to_csv(BATCH_RESULT_PATH, index=False, encoding="utf-8-sig") # 使用 utf-8-sig
                
                # 保存错误预测结果
                if incorrect_predictions:
                    df_incorrect = pd.DataFrame(incorrect_predictions, columns=["图像路径", "真实标签(推断)", "预测值", "置信度"])
                    df_incorrect.to_csv(incorrect_file_path, index=False, encoding="utf-8-sig")
                    logger.info(f"错误预测的图片路径和结果已保存至: {incorrect_file_path} ({len(incorrect_predictions)} 条)")
                else:
                    logger.info("没有预测错误的图像。")

                # 计算准确率 (仅针对有真实标签推断的图片)
                correct_predictions = df_predictions[df_predictions['真实标签(推断)'] != 'Unknown'] # 排除无法推断真实标签的
                if not correct_predictions.empty:
                    accuracy = (correct_predictions['真实标签(推断)'] == correct_predictions['预测值']).mean() * 100
                    self.txt_image_acc.setText(f"{accuracy:.2f}%")
                    logger.info(f"批量图像预测完成: {self.image_file_path}, 准确率: {accuracy:.2f}%")
                else:
                    self.txt_image_acc.setText("N/A")
                    logger.warning("没有可用于计算准确率的图片 (未推断出真实标签)。")

                self.txt_image_result.setText(str(BATCH_RESULT_PATH))
                show_message("提示", f"批量图像预测完成，结果已保存至: {BATCH_RESULT_PATH} 和 {incorrect_file_path} (如果存在错误预测)。")

            else:
                # 单张图像预测
                pred, conf = self._predict_single_image(self.image_file_path)
                self.txt_image_result.setText(pred)
                self.txt_image_acc.setText(f"{conf:.2%}")
                show_message("提示", "单张图像预测完成！")
                logger.info(f"单张图像预测完成: {self.image_file_path}, 预测: {pred}, 置信度: {conf:.2%}")

        except FileNotFoundError as e:
            show_message("错误", str(e), "critical")
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
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据集中缺少必要列: {', '.join(missing_cols)}。请检查文件内容或配置。")

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
            raise ValueError("数据清洗后为空，请检查文件内容或选择的数据集是否包含有效数值数据。")

        logger.info(f"数据预处理完成。样本数: {len(X)}")
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
        if len(X) < 2: # 至少需要2个样本进行训练
            raise ValueError("数据集样本数量不足，无法进行训练和预测。请提供更多数据。")

        # 划分训练集和测试集
        # 确保测试集至少有一个样本，如果总样本量很小，可能需要调整 test_size
        test_size = 0.2 if len(X) * 0.2 >= 1 else (1 / len(X) if len(X) > 0 else 0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 训练模型
        self.text_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # 使用所有可用核心
        self.text_model.fit(X_train, y_train)
        logger.info("随机森林回归模型训练完成")

        # 预测测试集并计算 RMSE
        y_pred_test = self.text_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse) 

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

            # 从 config.py 获取数据集配置
            if self.selected_text_type not in DATASET_CONFIGS:
                available_types = ", ".join(DATASET_CONFIGS.keys())
                logger.error(f"不支持的文本类型: {self.selected_text_type}, 可用类型: {available_types}")
                show_message("错误", f"不支持的文本类型: {self.selected_text_type}！可用类型: {available_types}", "critical")
                return
            
            config = DATASET_CONFIGS[self.selected_text_type]
            
            # 使用 safe_read_csv 读取用户选择的文件，而不是 config 中预设的
            df = safe_read_csv(self.text_file_path)
            df = self._normalize_column_names(df)
            logger.info(f"Loaded and normalized columns for text prediction: {df.columns.tolist()}")

            logger.info(f"为 '{self.selected_text_type}' 数据集准备数据。期望的特征列: {config['feature_columns']}，标签列: {config['label_column']}")

            X, y = self._prepare_text_data(df, config)

            rmse, y_full_pred = self._train_and_predict(X, y)

            output_path = Path(self.text_file_path).parent / "predictions.csv"
            # 确保预测结果的列名与原始标签列名一致
            result_df = pd.DataFrame({config["label_column"]: y, "预测值": y_full_pred})
            result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

            # 更新 UI
            self.txt_text_result.setText(str(output_path))
            self.txt_text_acc.setText(f"{self.selected_text_type} RMSE: {rmse:.4f}")
            show_message("提示", "文本预测完成！结果已保存。")
            logger.info(f"文本预测完成: {self.text_file_path}, RMSE: {rmse:.4f}")

        except FileNotFoundError as e:
            show_message("错误", str(e), "critical")
        except pd.errors.EmptyDataError:
            show_message("警告", "选择的文件是空的，无法进行文本预测。", "warning")
        except ValueError as ve:
            logger.error(f"数据处理错误: {ve}")
            show_message("错误", str(ve), "critical")
        except Exception as e:
            logger.error(f"文本预测失败: {e}", exc_info=True) # 打印完整堆栈信息
            show_message("错误", f"文本预测失败: {e}", "critical")

    def on_text_type_changed(self) -> None:
        """
        文本类型下拉框选择变化时触发。
        """
        self.selected_text_type = self.cbx_text_type.currentText()
        if self.selected_text_type == "文本类型选择":
            show_message("提示", "请选择一个具体的文本类型以进行预测。")
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
            # 文本类型下拉框只在非深度学习模型时可见
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
        # 这里可以添加更多基于 selected_text 和 selected_model 联动 cbx_chart_type 的逻辑
        # 例如，如果某些图表只适用于特定文本类型
        if (
            self.selected_text == "强震动参数数据集训练" # 更改为匹配 config 中的键
            and self.selected_model == "数据描述"
        ):
            self.cbx_chart_type.setVisible(False) # 示例：特定组合下隐藏图表选择
            show_message("提示", "当前文本类型与模型组合下无可用图表。", "information")
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
            elif (
                self.selected_model == "数据描述"
                and self.selected_chart == "散点图"
            ):
                # 示例：渲染散点图的逻辑 (需要根据实际数据和 pyecharts 编写)
                # data_config = DATASET_CONFIGS.get(self.selected_text)
                # if data_config:
                #     df = safe_read_csv(data_config["file_path"])
                #     # ... 散点图生成逻辑 ...
                #     # html_path = OUTPUT_DIR / "scatter_plot.html"
                #     # self.web_view.setUrl(QUrl.fromLocalFile(os.path.abspath(str(html_path))))
                show_message("提示", f"'{self.selected_model}' 的 '{self.selected_chart}' 暂未实现。")
                logger.info(f"尝试渲染未实现的图表: {self.selected_chart} for {self.selected_model}")
            else:
                show_message("提示", f"'{self.selected_model}' 的 '{self.selected_chart}' 暂未实现！")
                logger.info(f"尝试渲染未实现的图表: {self.selected_chart} for {self.selected_model}")
        except FileNotFoundError as e:
            show_message("错误", str(e), "critical")
        except pd.errors.EmptyDataError:
            show_message("警告", "数据文件为空，无法生成图表。", "warning")
        except ValueError as ve:
            logger.error(f"数据格式错误或缺失必要列: {ve}")
            show_message("错误", f"数据格式错误或缺失必要列: {ve}", "critical")
        except Exception as e:
            logger.error(f"渲染图表失败: {e}", exc_info=True)
            show_message("错误", f"无法渲染图表: {e}", "critical")

    def _render_confusion_matrix(self) -> None:
        """
        渲染混淆矩阵热力图。
        """
        try:
            # 混淆矩阵的输入文件应为 BATCH_RESULT_PATH (CSV)
            # 这里的 names 参数通常用于没有头部行的文件。如果 BATCH_RESULT_PATH 已经有头部，则不需要
            # 根据 config.py 的修改，BATCH_RESULT_PATH 现在是 CSV 并应该有头部
            df = safe_read_csv(BATCH_RESULT_PATH) 
            
            # 确保 '真实标签(推断)' 和 '预测值' 列存在
            required_cols = ["真实标签(推断)", "预测值"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"混淆矩阵所需列缺失。需要: {required_cols}，实际: {df.columns.tolist()}")

            conf_matrix, heatmap_data, labels = create_confusion_matrix(df)
            create_heatmap(
                conf_matrix, heatmap_data, labels, str(CONFUSION_MATRIX_PATH)
            )
            # 将 HTML 文件路径转换为 QUrl
            self.web_view.setUrl(
                QUrl.fromLocalFile(os.path.abspath(str(CONFUSION_MATRIX_PATH)))
            )
            self.web_view.setFixedSize(self.view_place_widget.size())
            logger.info(f"渲染混淆矩阵热力图: {CONFUSION_MATRIX_PATH}")
        except FileNotFoundError as e:
            show_message("错误", str(e), "critical")
        except pd.errors.EmptyDataError:
            show_message("警告", "混淆矩阵所需的数据文件为空。", "warning")
        except ValueError as ve:
            logger.error(f"生成混淆矩阵数据错误: {ve}")
            show_message("错误", f"生成混淆矩阵数据错误: {ve}", "critical")
        except Exception as e:
            logger.error(f"渲染混淆矩阵失败: {e}", exc_info=True)
            show_message("错误", f"无法渲染混淆矩阵: {e}", "critical")