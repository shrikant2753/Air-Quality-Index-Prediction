import tkinter
import PyQt5.QtWidgets as QtWidgets
from tkinter import filedialog
from tkinter import messagebox 
from PyQt5 import QtCore, QtGui , QtWidgets


import math
import random
import string
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from array import array
from numpy import linalg as LA
import pickle

import matplotlib.pyplot as plt




print('******Start*****')
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)

class Ui_MainWindow1(object):
    

    def setupUii(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1200, 800)
        MainWindow.setStyleSheet(_fromUtf8("\n""background-image: url(III3.jpg);\n"""))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(750, 180, 111, 27))
        self.pushButton.clicked.connect(self.quit)
        self.pushButton.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(0, 0, 0);"))
       
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
#################################################################
        

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(550, 180, 131, 27))
        self.pushButton_2.clicked.connect(self.show1)
        self.pushButton_2.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(0, 0, 0);"))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(550, 220, 131, 27))
        self.pushButton_4.clicked.connect(self.show2)
        self.pushButton_4.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(0, 0, 0);"))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(550, 260, 131, 27))
        self.pushButton_5.clicked.connect(self.show3)
        self.pushButton_5.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(0, 0, 0);"))
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        
        

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
       
        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "AQI PREDICTION", None))
        self.pushButton_2.setText(_translate("MainWindow", "TEST", None))
        self.pushButton_4.setText(_translate("MainWindow", "TRAIN", None))
        self.pushButton_5.setText(_translate("MainWindow", "PERFORMANCE ANALYSIS", None))
        self.pushButton.setText(_translate("MainWindow", "Exit", None))

    def quit(self):
        print ('Process end')
        print ('******End******')
        quit()
         
    def show1(self):
        import clf_AQI_ideal_LR_TEST
                

    def show2(self):
        import clf_AQI_ideal_LR_TRN

    def show3(self):
        import clf_AQI_ideal_LR_VIS
            
                


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow1()
    ui.setupUii(MainWindow)
    MainWindow.move(550, 170)
    MainWindow.show()
    sys.exit(app.exec_())


