# -*- coding: utf-8 -*-
"""
主窗口逻辑实现，处理用户交互、文件操作、预测和数据可视化功能。
"""

import os
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableView, QVBoxLayout
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from Main_window_UI_interface import Ui_MainWindow
from Data_Visualization_UI import Ui_MainWindow as VisualizationUi
from Confusion_matrix_heat_map import create_confusion_matrix, create_heatmap
from config import (
    BATCH_RESULT_PATH,
    CONFUSION_MATRIX_PATH,
    TEXT_TYPES,
    VISUALIZATION_TYPES,
    VISUALIZATION_CHARTS,
)
from utils import safe_read_csv, display_image, show_message, logger
from pathlib import Path
from typing import Optional


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
        self._connect_signals()
        logger.info("主窗口初始化完成")

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
                    names=(
                        ["真实值", "预测值"] if "batch_results" in file_name else None
                    ),
                )
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
            logger.error(f"打开文本文件失败: {str(e)}")
            show_message("错误", "无法加载文本文件", "critical")

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
            logger.error(f"打开图片失败: {str(e)}")
            show_message("错误", "无法加载图片", "critical")

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
            logger.error(f"打开图片文件夹失败: {str(e)}")
            show_message("错误", "无法选择图片文件夹", "critical")

    def predict_image(self) -> None:
        """
        执行图像预测（占位，假设模型存在）。

        Raises:
            NotImplementedError: 模型预测功能未实现
        """
        try:
            if not self.image_file_path:
                show_message("警告", "请先选择图片或图片文件夹！", "warning")
                return

            # 占位：假设使用深度学习模型进行预测
            if self.is_batch_image_predict:
                self.txt_image_result.setText(str(BATCH_RESULT_PATH))
                self.txt_image_acc.setText("85% (示例)")
                show_message("提示", "批量图像预测完成，结果已保存！")
            else:
                self.txt_image_result.setText("地震 (示例)")
                self.txt_image_acc.setText("89% (示例)")
                show_message("提示", "单张图像预测完成！")
            logger.info(f"图像预测完成: {self.image_file_path}")

        except NotImplementedError:
            logger.error("图像预测功能未实现")
            show_message("错误", "图像预测功能尚未实现", "critical")
        except Exception as e:
            logger.error(f"图像预测失败: {str(e)}")
            show_message("错误", "图像预测失败", "critical")

    def predict_text(self) -> None:
        """
        执行文本预测（占位，假设模型存在）。

        Raises:
            NotImplementedError: 模型预测功能未实现
        """
        try:
            if not self.text_file_path:
                show_message("警告", "请先选择文本文件！", "warning")
                return
            if self.selected_text_type == "文本类型选择":
                show_message("警告", "请选择文本类型！", "warning")
                return

            # 占位：假设使用机器学习模型进行预测
            self.txt_text_result.setText(
                str(Path(self.text_file_path).parent / "predictions.csv")
            )
            self.txt_text_acc.setText(f"{self.selected_text_type} RMSE: 0.1234 (示例)")
            show_message("提示", "文本预测完成！")
            logger.info(f"文本预测完成: {self.text_file_path}")

        except NotImplementedError:
            logger.error("文本预测功能未实现")
            show_message("错误", "文本预测功能尚未实现", "critical")
        except Exception as e:
            logger.error(f"文本预测失败: {str(e)}")
            show_message("错误", "文本预测失败", "critical")

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
            logger.error(f"打开可视化窗口失败: {str(e)}")
            show_message("错误", "无法打开可视化窗口", "critical")


class VisualizationWindow(QMainWindow, VisualizationUi):
    """
    数据可视化窗口类，继承 QMainWindow 和 UI 定义，处理可视化逻辑。
    """

    def __init__(self, parent: Optional[QMainWindow] = None):
        """
        初始化可视化窗口，设置 UI 并连接信号槽。

        Args:
            parent (Optional[QMainWindow]): 父窗口，默认为 None
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
            logger.error(f"渲染图表失败: {str(e)}")
            show_message("错误", "无法渲染图表", "critical")

    def _render_confusion_matrix(self) -> None:
        """
        渲染混淆矩阵热力图。
        """
        try:
            df = safe_read_csv(BATCH_RESULT_PATH, names=["真实值", "预测值"])
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
            logger.error(f"渲染混淆矩阵失败: {str(e)}")
            show_message("错误", "无法渲染混淆矩阵", "critical")
