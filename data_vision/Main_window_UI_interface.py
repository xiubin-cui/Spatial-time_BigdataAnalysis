# -*- coding: utf-8 -*-
"""
主窗口 UI 界面，定义用户交互界面布局和控件。
由 PyQt5 UI 代码生成器生成，包含手动优化。
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from .config import TEXT_TYPES


class Ui_MainWindow:
    """
    主窗口 UI 类，负责设置界面布局和控件。
    """

    def setupUi(self, MainWindow: QtWidgets.QMainWindow) -> None:
        """
        设置主窗口的 UI 布局和控件。

        Args:
            MainWindow (QtWidgets.QMainWindow): 主窗口实例
        """
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(825, 673)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 文本文件打开按钮
        self.btn_open_file = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open_file.setGeometry(QtCore.QRect(400, 0, 131, 41))
        self.btn_open_file.setObjectName("btn_open_file")

        # 导入图片按钮
        self.btn_open_image = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open_image.setGeometry(QtCore.QRect(0, 0, 121, 41))
        self.btn_open_image.setObjectName("btn_open_image")

        # 待预测图片标签
        self.lbl_image = QtWidgets.QLabel(self.centralwidget)
        self.lbl_image.setGeometry(QtCore.QRect(140, 360, 91, 31))
        self.lbl_image.setObjectName("lbl_image")

        # 可视化界面按钮
        self.btn_vision_data = QtWidgets.QPushButton(self.centralwidget)
        self.btn_vision_data.setGeometry(QtCore.QRect(0, 500, 121, 41))
        self.btn_vision_data.setObjectName("btn_vision_data")

        # 文本数据显示表格
        self.table_text_view = QtWidgets.QTableView(self.centralwidget)
        self.table_text_view.setGeometry(QtCore.QRect(350, 60, 441, 291))
        self.table_text_view.setObjectName("table_text_view")

        # 图像预测按钮
        self.btn_predict_image = QtWidgets.QPushButton(self.centralwidget)
        self.btn_predict_image.setGeometry(QtCore.QRect(260, 0, 121, 41))
        self.btn_predict_image.setObjectName("btn_predict_image")

        # 图片预测结果输入框
        self.txt_image_result = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_image_result.setGeometry(QtCore.QRect(120, 400, 121, 41))
        self.txt_image_result.setObjectName("txt_image_result")

        # 图片预测结果标签
        self.lbl_image_result = QtWidgets.QLabel(self.centralwidget)
        self.lbl_image_result.setGeometry(QtCore.QRect(10, 400, 91, 41))
        self.lbl_image_result.setObjectName("lbl_image_result")

        # 文本预测结果存储路径标签
        self.lbl_text_result_path = QtWidgets.QLabel(self.centralwidget)
        self.lbl_text_result_path.setGeometry(QtCore.QRect(290, 400, 191, 41))
        self.lbl_text_result_path.setObjectName("lbl_text_result_path")

        # 文本训练预测按钮
        self.btn_predict_text = QtWidgets.QPushButton(self.centralwidget)
        self.btn_predict_text.setGeometry(QtCore.QRect(700, 0, 121, 41))
        self.btn_predict_text.setObjectName("btn_predict_text")

        # 文本预测结果输入框
        self.txt_text_result = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_text_result.setGeometry(QtCore.QRect(480, 400, 341, 41))
        self.txt_text_result.setObjectName("txt_text_result")

        # 待预测文本标签
        self.lbl_text = QtWidgets.QLabel(self.centralwidget)
        self.lbl_text.setGeometry(QtCore.QRect(520, 360, 91, 31))
        self.lbl_text.setObjectName("lbl_text")

        # 图像显示区域
        self.lbl_image_view = QtWidgets.QLabel(self.centralwidget)
        self.lbl_image_view.setGeometry(QtCore.QRect(10, 60, 331, 291))
        self.lbl_image_view.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lbl_image_view.setText("")
        self.lbl_image_view.setObjectName("lbl_image_view")

        # 文本类型选择下拉框
        self.cbx_text_type = QtWidgets.QComboBox(self.centralwidget)
        self.cbx_text_type.setGeometry(QtCore.QRect(560, 0, 131, 41))
        self.cbx_text_type.setObjectName("cbx_text_type")
        self.cbx_text_type.addItems(["文本类型选择"] + TEXT_TYPES)

        # 图片预测准确率标签
        self.lbl_image_acc = QtWidgets.QLabel(self.centralwidget)
        self.lbl_image_acc.setGeometry(QtCore.QRect(10, 450, 121, 41))
        self.lbl_image_acc.setObjectName("lbl_image_acc")

        # 图片预测准确率输入框
        self.txt_image_acc = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_image_acc.setGeometry(QtCore.QRect(120, 450, 121, 41))
        self.txt_image_acc.setObjectName("txt_image_acc")

        # 文本预测准确率标签
        self.lbl_text_acc = QtWidgets.QLabel(self.centralwidget)
        self.lbl_text_acc.setGeometry(QtCore.QRect(320, 450, 141, 41))
        self.lbl_text_acc.setObjectName("lbl_text_acc")

        # 文本预测准确率文本框
        self.txt_text_acc = QtWidgets.QTextEdit(self.centralwidget)
        self.txt_text_acc.setGeometry(QtCore.QRect(480, 450, 331, 161))
        self.txt_text_acc.setObjectName("txt_text_acc")

        # 导入图片文件按钮
        self.btn_open_image_batch = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open_image_batch.setGeometry(QtCore.QRect(130, 0, 121, 41))
        self.btn_open_image_batch.setObjectName("btn_open_image_batch")

        MainWindow.setCentralWidget(self.centralwidget)

        # 菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 825, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        # 状态栏
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow: QtWidgets.QMainWindow) -> None:
        """
        设置 UI 控件的文本内容，支持多语言。

        Args:
            MainWindow (QtWidgets.QMainWindow): 主窗口实例
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "地震数据分析与预测"))
        self.btn_open_file.setText(_translate("MainWindow", "文本文件打开"))
        self.btn_open_image.setText(_translate("MainWindow", "导入图片"))
        self.lbl_image.setText(_translate("MainWindow", "待预测图片"))
        self.btn_vision_data.setText(_translate("MainWindow", "可视化界面"))
        self.btn_predict_image.setText(_translate("MainWindow", "图像预测"))
        self.lbl_image_result.setText(_translate("MainWindow", "图片预测结果"))
        self.lbl_text_result_path.setText(
            _translate("MainWindow", "文本预测结果存储路径")
        )
        self.btn_predict_text.setText(_translate("MainWindow", "文本训练预测"))
        self.lbl_text.setText(_translate("MainWindow", "待预测文本"))
        self.cbx_text_type.setItemText(0, _translate("MainWindow", "文本类型选择"))
        for i, text_type in enumerate(TEXT_TYPES, 1):
            self.cbx_text_type.setItemText(i, _translate("MainWindow", text_type))
        self.lbl_image_acc.setText(_translate("MainWindow", "图片预测准确率"))
        self.lbl_text_acc.setText(_translate("MainWindow", "文本预测准确率"))
        self.btn_open_image_batch.setText(_translate("MainWindow", "导入图片文件"))
