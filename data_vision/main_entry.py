# -*- coding: utf-8 -*-
"""
主程序入口，初始化并显示主窗口。
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from data_vision.Main_window_logic import MainWindow
from .utils import logger


def main():
    """
    主函数，启动应用程序。
    """
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        logger.info("应用程序启动成功")
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"应用程序启动失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
