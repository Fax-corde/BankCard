import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from pyqt5_plugins.examplebuttonplugin import QtGui
from window import Ui_MainWindow
import detect


# noinspection PyMethodMayBeStatic,PyAttributeOutsideInit
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.imgpyth = ''
        self.cwd = os.getcwd()  # 获取当前程序文件位置
        self.actionopen.triggered.connect(self.openimage)
        self.actionrecognize.triggered.connect(self.a)
        self.actionrecognize2.triggered.connect(self.b)

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.imgpyth = imgName
        img = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img)

    def a(self):
        s1, s2, s3 = detect.fun1(self.imgpyth)
        self.plainTextEdit.clear()
        self.plainTextEdit_2.clear()
        self.plainTextEdit_3.clear()
        self.plainTextEdit.appendPlainText(s1)
        self.plainTextEdit_2.appendPlainText(s2)

    def b(self):
        s1, s2, s3 = detect.fun2()
        self.plainTextEdit_3.clear()
        self.plainTextEdit_3.appendPlainText(s3)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
