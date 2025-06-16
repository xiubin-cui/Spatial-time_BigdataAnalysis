# -*- coding: utf-8 -*-
"""
数据可视化窗口 UI 界面，定义可视化界面的布局和控件。
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from .config import VISUALIZATION_TYPES, TEXT_TYPES, DATASET_CONFIGS # 导入 DATASET_CONFIGS
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

            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)
            logger.info("可视化窗口 UI 初始化完成")
        except Exception as e:
            logger.error(f"可视化窗口 UI 初始化失败: {e}")
            raise # 重新抛出异常，让上层捕获

    def _setup_layout(self) -> None:
        """
        设置界面布局。
        """
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_layout.setObjectName("main_layout")

        # 左侧控制面板
        self.control_panel = QtWidgets.QWidget(self.centralwidget)
        self.control_panel.setMinimumWidth(250)
        self.control_panel.setMaximumWidth(300)
        self.control_panel.setObjectName("control_panel")
        self.control_layout = QtWidgets.QVBoxLayout(self.control_panel)
        self.control_layout.setObjectName("control_layout")

        self.main_layout.addWidget(self.control_panel)

        # 右侧可视化结果展示区
        self.view_place_widget = QtWidgets.QWidget(self.centralwidget)
        self.view_place_widget.setObjectName("view_place_widget")
        self.view_layout = QtWidgets.QVBoxLayout(self.view_place_widget)
        self.view_layout.setObjectName("view_layout")

        self.lbl_result = QtWidgets.QLabel(self.view_place_widget)
        self.lbl_result.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lbl_result.setFont(font)
        self.view_layout.addWidget(self.lbl_result)

        self.main_layout.addWidget(self.view_place_widget)

    def _setup_controls(self) -> None:
        """
        初始化控件并添加到控制面板布局。
        """
        # 可视化类型选择下拉框
        self.cbx_model_type = QtWidgets.QComboBox(self.control_panel)
        self.cbx_model_type.setObjectName("cbx_model_type")
        self.cbx_model_type.addItem("") # Placeholder for "可视化类型选择"
        # 动态添加 VISUALIZATION_TYPES
        for vis_type in VISUALIZATION_TYPES:
            self.cbx_model_type.addItem(vis_type)
        self.control_layout.addWidget(self.cbx_model_type)

        # 文本类型选择下拉框 (数据集选择)
        self.cbx_text_type = QtWidgets.QComboBox(self.control_panel)
        self.cbx_text_type.setObjectName("cbx_text_type")
        self.cbx_text_type.addItem("") # Placeholder for "文本类型选择"
        # 动态添加 TEXT_TYPES
        for text_type in TEXT_TYPES:
            self.cbx_text_type.addItem(text_type)
        self.control_layout.addWidget(self.cbx_text_type)

        # 可视化图选择下拉框
        self.cbx_chart_type = QtWidgets.QComboBox(self.control_panel)
        self.cbx_chart_type.setObjectName("cbx_chart_type")
        self.cbx_chart_type.addItem("") # Placeholder for "可视化图选择"
        self.control_layout.addWidget(self.cbx_chart_type)

        # 添加一个弹簧，让控件靠上对齐
        self.control_layout.addStretch(1)

    def _translate(self, context: str, text: str) -> str:
        """
        辅助函数，用于翻译文本以支持多语言。

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
            # VISUALIZATION_TYPES 已经通过 _setup_controls 添加
            self.cbx_chart_type.setItemText(
                0, self._translate("MainWindow", "可视化图选择")
            )
            self.lbl_result.setText(self._translate("MainWindow", "可视化结果展示区"))
            self.cbx_text_type.setItemText(
                0, self._translate("MainWindow", "文本类型选择")
            )
            # TEXT_TYPES 已经通过 _setup_controls 添加
        except Exception as e:
            logger.error(f"重新设置 UI 文本失败: {e}")
            raise # 重新抛出异常