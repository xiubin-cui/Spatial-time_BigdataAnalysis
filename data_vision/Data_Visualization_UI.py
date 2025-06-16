# -*- coding: utf-8 -*-
"""
数据可视化窗口 UI 界面，定义可视化界面的布局和控件。
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from .config import VISUALIZATION_TYPES, TEXT_TYPES
from .utils import logger


class Ui_MainWindow:
    """
    数据可视化窗口 UI 类，负责设置界面布局和控件。
    """

    def setupUi(self, MainWindow: QtWidgets.QMainWindow) -> None:
        """
        设置主窗口的 UI 布局和控件。

        Args:
            MainWindow (QtWidgets.QMainWindow): 主窗口实例
        """
        try:
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(1200, 731)
            MainWindow.setWindowTitle(self._translate("MainWindow", "地震数据可视化"))

            # 中央部件
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            MainWindow.setCentralWidget(self.centralwidget)

            # 设置布局
            self._setup_layout()

            # 初始化控件
            self._setup_controls()

            # 菜单栏
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 26))
            self.menubar.setObjectName("menubar")
            MainWindow.setMenuBar(self.menubar)

            # 状态栏
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)

            QtCore.QMetaObject.connectSlotsByName(MainWindow)
            logger.info("可视化窗口 UI 设置完成")

        except Exception as e:
            logger.error(f"设置可视化窗口 UI 时发生错误: {str(e)}")
            raise

    def _setup_layout(self) -> None:
        """
        设置中央部件的布局。
        """
        self.layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        # 顶部控件容器
        self.top_widget = QtWidgets.QWidget(self.centralwidget)
        self.top_layout = QtWidgets.QHBoxLayout(self.top_widget)
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.top_widget)

        # 可视化显示区域
        self.view_place_widget = QtWidgets.QWidget(self.centralwidget)
        self.view_place_widget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.view_place_widget.setObjectName("view_place_widget")
        self.layout.addWidget(self.view_place_widget)

    def _setup_controls(self) -> None:
        """
        初始化 UI 控件。
        """
        # 可视化类型选择下拉框
        self.cbx_model_type = QtWidgets.QComboBox(self.top_widget)
        self.cbx_model_type.setObjectName("cbx_model_type")
        self.cbx_model_type.addItems(["可视化类型选择"] + VISUALIZATION_TYPES)
        self.cbx_model_type.setFixedSize(200, 40)
        self.top_layout.addWidget(self.cbx_model_type)

        # 文本类型选择下拉框
        self.cbx_text_type = QtWidgets.QComboBox(self.top_widget)
        self.cbx_text_type.setObjectName("cbx_text_type")
        self.cbx_text_type.addItems(["文本类型选择"] + TEXT_TYPES)
        self.cbx_text_type.setFixedSize(200, 40)
        self.top_layout.addWidget(self.cbx_text_type)

        # 可视化图选择下拉框
        self.cbx_chart_type = QtWidgets.QComboBox(self.top_widget)
        self.cbx_chart_type.setObjectName("cbx_chart_type")
        self.cbx_chart_type.addItem("可视化图选择")
        self.cbx_chart_type.setFixedSize(200, 40)
        self.top_layout.addWidget(self.cbx_chart_type)

        # 伸缩项
        self.top_layout.addStretch()

        # 可视化结果展示区标签
        self.lbl_result = QtWidgets.QLabel(self.centralwidget)
        self.lbl_result.setText(self._translate("MainWindow", "可视化结果展示区"))
        self.lbl_result.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_result.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.layout.addWidget(self.lbl_result)

    def _translate(self, context: str, text: str) -> str:
        """
        翻译文本以支持多语言。

        Args:
            context (str): 翻译上下文
            text (str): 待翻译的文本

        Returns:
            str: 翻译后的文本
        """
        return QtCore.QCoreApplication.translate(context, text)

    def retranslateUi(self, MainWindow: QtWidgets.QMainWindow) -> None:
        """
        设置 UI 控件的文本内容。

        Args:
            MainWindow (QtWidgets.QMainWindow): 主窗口实例
        """
        try:
            MainWindow.setWindowTitle(self._translate("MainWindow", "地震数据可视化"))
            self.cbx_model_type.setItemText(
                0, self._translate("MainWindow", "可视化类型选择")
            )
            for i, vis_type in enumerate(VISUALIZATION_TYPES, 1):
                self.cbx_model_type.setItemText(
                    i, self._translate("MainWindow", vis_type)
                )
            self.cbx_chart_type.setItemText(
                0, self._translate("MainWindow", "可视化图选择")
            )
            self.lbl_result.setText(self._translate("MainWindow", "可视化结果展示区"))
            self.cbx_text_type.setItemText(
                0, self._translate("MainWindow", "文本类型选择")
            )
            for i, text_type in enumerate(TEXT_TYPES, 1):
                self.cbx_text_type.setItemText(
                    i, self._translate("MainWindow", text_type)
                )
            logger.info("可视化窗口 UI 文本翻译完成")
        except Exception as e:
            logger.error(f"翻译可视化窗口 UI 文本时发生错误: {str(e)}")
            raise
