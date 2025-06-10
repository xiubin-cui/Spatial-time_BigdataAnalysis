import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from untitled import Ui_MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()  # 根据上面所创建的类的名字更改
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())