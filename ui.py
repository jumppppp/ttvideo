# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(868, 683)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupBox = QtWidgets.QGroupBox(parent=self.tab)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(parent=self.groupBox)
        self.label.setMinimumSize(QtCore.QSize(600, 400))
        self.label.setAutoFillBackground(False)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.tab)
        self.groupBox_2.setMinimumSize(QtCore.QSize(200, 0))
        self.groupBox_2.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_4 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 1, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_4.addWidget(self.pushButton_4, 7, 0, 1, 2)
        self.textEdit = QtWidgets.QTextEdit(parent=self.groupBox_2)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout_4.addWidget(self.textEdit, 8, 0, 1, 2)
        self.pushButton = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_4.addWidget(self.pushButton, 3, 0, 1, 2)
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 2, 0, 1, 1)
        self.comboBox_2 = QtWidgets.QComboBox(parent=self.groupBox_2)
        self.comboBox_2.setObjectName("comboBox_2")
        self.gridLayout_4.addWidget(self.comboBox_2, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 0, 0, 1, 2)
        self.comboBox = QtWidgets.QComboBox(parent=self.groupBox_2)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout_4.addWidget(self.comboBox, 1, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_4.addWidget(self.pushButton_2, 4, 0, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_4.addWidget(self.pushButton_3, 4, 1, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout_4.addWidget(self.pushButton_5, 6, 0, 1, 2)
        self.gridLayout_3.addWidget(self.groupBox_2, 0, 1, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.tab)
        self.groupBox_3.setMinimumSize(QtCore.QSize(0, 110))
        self.groupBox_3.setMaximumSize(QtCore.QSize(16777215, 210))
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.plainTextEdit_2 = QtWidgets.QPlainTextEdit(parent=self.groupBox_3)
        self.plainTextEdit_2.setMinimumSize(QtCore.QSize(0, 100))
        self.plainTextEdit_2.setMaximumSize(QtCore.QSize(16777215, 200))
        self.plainTextEdit_2.setObjectName("plainTextEdit_2")
        self.gridLayout_5.addWidget(self.plainTextEdit_2, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox_3, 1, 0, 1, 2)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 868, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "预览"))
        self.label.setText(_translate("MainWindow", "显示1"))
        self.groupBox_2.setTitle(_translate("MainWindow", "控制"))
        self.label_4.setText(_translate("MainWindow", "输出分辨率："))
        self.pushButton_4.setText(_translate("MainWindow", "添加屏蔽窗口PID"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:\'Microsoft YaHei UI\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">wechat</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">qq</p></body></html>"))
        self.textEdit.setPlaceholderText(_translate("MainWindow", "添加屏蔽窗口的PID，格式每行一个"))
        self.pushButton.setText(_translate("MainWindow", "开始录制"))
        self.label_3.setText(_translate("MainWindow", "输出帧率："))
        self.label_2.setText(_translate("MainWindow", "时间显示 0s"))
        self.pushButton_2.setText(_translate("MainWindow", "暂停/继续"))
        self.pushButton_3.setText(_translate("MainWindow", "结束录制"))
        self.pushButton_5.setText(_translate("MainWindow", "打开录制文件夹"))
        self.groupBox_3.setTitle(_translate("MainWindow", "输出"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "录屏"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2"))
