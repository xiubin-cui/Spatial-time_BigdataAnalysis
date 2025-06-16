# -*- coding: utf-8 -*-
"""
主窗口 UI 界面，定义用户交互界面布局和控件。
由 PyQt5 UI 代码生成器生成，包含手动优化。
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from .config import TEXT_TYPES # 确保导入 TEXT_TYPES
from .utils import logger # 导入logger

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
        try:
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
            self.lbl_image.setGeometry(QtCore.QRect(0, 60, 121, 31))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.lbl_image.setFont(font)
            self.lbl_image.setObjectName("lbl_image")

            # 可视化界面按钮
            self.btn_vision_data = QtWidgets.QPushButton(self.centralwidget)
            self.btn_vision_data.setGeometry(QtCore.QRect(670, 0, 131, 41))
            self.btn_vision_data.setObjectName("btn_vision_data")

            # 图像预测按钮
            self.btn_predict_image = QtWidgets.QPushButton(self.centralwidget)
            self.btn_predict_image.setGeometry(QtCore.QRect(130, 0, 121, 41))
            self.btn_predict_image.setObjectName("btn_predict_image")

            # 图片预测结果标签
            self.lbl_image_result = QtWidgets.QLabel(self.centralwidget)
            self.lbl_image_result.setGeometry(QtCore.QRect(0, 360, 121, 31))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.lbl_image_result.setFont(font)
            self.lbl_image_result.setObjectName("lbl_image_result")

            # 文本预测结果存储路径标签
            self.lbl_text_result_path = QtWidgets.QLabel(self.centralwidget)
            self.lbl_text_result_path.setGeometry(QtCore.QRect(400, 360, 171, 31))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.lbl_text_result_path.setFont(font)
            self.lbl_text_result_path.setObjectName("lbl_text_result_path")

            # 文本训练预测按钮
            self.btn_predict_text = QtWidgets.QPushButton(self.centralwidget)
            self.btn_predict_text.setGeometry(QtCore.QRect(540, 0, 121, 41))
            self.btn_predict_text.setObjectName("btn_predict_text")

            # 待预测文本标签
            self.lbl_text = QtWidgets.QLabel(self.centralwidget)
            self.lbl_text.setGeometry(QtCore.QRect(400, 60, 121, 31))
            font = QtGui.QFont()
            font.setPointSize(12)
            font.setBold(False)
            font.setWeight(50)
            self.lbl_text.setFont(font)
            self.lbl_text.setObjectName("lbl_text")

            # 文本类型选择下拉框
            self.cbx_text_type = QtWidgets.QComboBox(self.centralwidget)
            self.cbx_text_type.setGeometry(QtCore.QRect(530, 60, 271, 31))
            self.cbx_text_type.setObjectName("cbx_text_type")
            self.cbx_text_type.addItem("") # Placeholder for "文本类型选择"
            # 动态添加 TEXT_TYPES
            for text_type in TEXT_TYPES:
                self.cbx_text_type.addItem(text_type)

            # 图像文件路径输入框
            self.txt_image_result = QtWidgets.QLineEdit(self.centralwidget)
            self.txt_image_result.setGeometry(QtCore.QRect(130, 360, 261, 31))
            self.txt_image_result.setObjectName("txt_image_result")
            self.txt_image_result.setReadOnly(True) # 设置为只读

            # 文本文件路径输入框
            self.txt_text_result = QtWidgets.QLineEdit(self.centralwidget)
            self.txt_text_result.setGeometry(QtCore.QRect(580, 360, 231, 31))
            self.txt_text_result.setObjectName("txt_text_result")
            self.txt_text_result.setReadOnly(True) # 设置为只读

            # 图像预测准确率标签
            self.lbl_image_acc = QtWidgets.QLabel(self.centralwidget)
            self.lbl_image_acc.setGeometry(QtCore.QRect(0, 410, 121, 31))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.lbl_image_acc.setFont(font)
            self.lbl_image_acc.setObjectName("lbl_image_acc")

            # 图像预测准确率显示框
            self.txt_image_acc = QtWidgets.QLineEdit(self.centralwidget)
            self.txt_image_acc.setGeometry(QtCore.QRect(130, 410, 261, 31))
            self.txt_image_acc.setObjectName("txt_image_acc")
            self.txt_image_acc.setReadOnly(True) # 设置为只读

            # 文本预测准确率标签
            self.lbl_text_acc = QtWidgets.QLabel(self.centralwidget)
            self.lbl_text_acc.setGeometry(QtCore.QRect(400, 410, 171, 31))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.lbl_text_acc.setFont(font)
            self.lbl_text_acc.setObjectName("lbl_text_acc")

            # 文本预测准确率显示框
            self.txt_text_acc = QtWidgets.QLineEdit(self.centralwidget)
            self.txt_text_acc.setGeometry(QtCore.QRect(580, 410, 231, 31))
            self.txt_text_acc.setObjectName("txt_text_acc")
            self.txt_text_acc.setReadOnly(True) # 设置为只读

            # 图片显示区域
            self.lbl_image_view = QtWidgets.QLabel(self.centralwidget)
            self.lbl_image_view.setGeometry(QtCore.QRect(0, 100, 391, 251))
            self.lbl_image_view.setStyleSheet("background-color: lightgray; border: 1px solid gray;")
            self.lbl_image_view.setText("") # 初始为空
            self.lbl_image_view.setAlignment(QtCore.Qt.AlignCenter) # 居中对齐
            self.lbl_image_view.setObjectName("lbl_image_view")

            # 文本表格显示区域
            self.table_text_view = QtWidgets.QTableView(self.centralwidget)
            self.table_text_view.setGeometry(QtCore.QRect(400, 100, 411, 251))
            self.table_text_view.setObjectName("table_text_view")

            # 批量导入图片按钮
            self.btn_open_image_batch = QtWidgets.QPushButton(self.centralwidget)
            self.btn_open_image_batch.setGeometry(QtCore.QRect(260, 0, 131, 41))
            self.btn_open_image_batch.setObjectName("btn_open_image_batch")

            MainWindow.setCentralWidget(self.centralwidget)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 825, 26))
            self.menubar.setObjectName("menubar")
            MainWindow.setMenuBar(self.menubar)
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)

            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)
            logger.info("主窗口 UI 初始化完成")
        except Exception as e:
            logger.error(f"主窗口 UI 初始化失败: {e}")
            raise # 重新抛出异常

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
        设置 UI 控件的文本内容，支持多语言。

        Args:
            MainWindow (QtWidgets.QMainWindow): 主窗口实例
        """
        _translate = self._translate
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
        # TEXT_TYPES 已经在 setupUi 中动态添加，这里只需设置 placeholder
        self.lbl_image_acc.setText(_translate("MainWindow", "图像预测准确率"))
        self.lbl_text_acc.setText(_translate("MainWindow", "文本预测准确率"))
        self.btn_open_image_batch.setText(_translate("MainWindow", "批量导入图片"))