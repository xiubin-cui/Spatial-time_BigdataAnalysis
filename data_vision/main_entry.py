# -*- coding: utf-8 -*-
"""
主程序入口，初始化并显示主窗口。
"""

import sys
from PyQt5.QtWidgets import QApplication
from data_vision.Main_window_logic import MainWindow
from data_vision.utils import logger # 确保从 .utils 导入 logger

def main():
    """
    主函数，启动应用程序。
    """
    try:
        app = QApplication(sys.argv)
        # 设置应用程序图标（可选）
        # app.setWindowIcon(QIcon("path/to/your/icon.png")) 
        window = MainWindow()
        window.show()
        logger.info("应用程序启动成功")
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"应用程序启动失败: {e}", exc_info=True) # 打印完整堆栈信息
        sys.exit(1)


if __name__ == "__main__":
    main()