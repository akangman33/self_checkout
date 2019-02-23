#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QFont, QIcon, QColor
import os
import time
import face_recognition
import glob
import numpy as np
import pandas as pd
import qdarkstyle
from PIL import Image, ImageDraw



class SignInWidget(QtWidgets.QWidget):
    is_checkout_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super(SignInWidget, self).__init__(parent)

        # self.face_recong = face.Recognition()
        self.resize(900, 600)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0

    def set_ui(self):
        self.__layout_main = QtWidgets.QVBoxLayout()
        self.__layout_fun_button = QtWidgets.QHBoxLayout()

        self.setGeometry(50, 50, 1000, 1000)
        self.button_open_camera = QtWidgets.QPushButton(u'開啟相機', self)
        self.button_signin = QtWidgets.QPushButton(u'會員登入', self)
        self.button_open_camera.setIcon(QIcon("./light/appbar.camera.png"))
        self.button_open_camera.setIconSize(QtCore.QSize(250, 90))
        self.button_signin.setIcon(QIcon("./light/appbar.user.tie.png"))
        self.button_signin.setIconSize(QtCore.QSize(250, 90))
        # self.button_close = QtWidgets.QPushButton(u'退出')

        # Button 的颜色修改
        button_color = [self.button_open_camera, self.button_signin]
        for i in range(2):
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                          "QPushButton:hover{color:red}"
                                          "QPushButton{border-image:url(./background/new_robo.jpg)}"
                                          "QPushButton{text-align:right}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:10px}"
                                          "QPushButton{padding:2px 4px}")
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.button_open_camera.setFont(font)
        self.button_signin.setFont(font)
        # self.button_open_camera.setMinimumHeight(20)
        # self.button_signin.setMinimumHeight(20)
        # self.button_close.setMinimumHeight(50)

        # move()方法移动窗口在屏幕上的位置到x = 300，y = 300坐标。
        self.move(500, 0)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)
        self.Hlayout = QHBoxLayout()
        self.Hlayout.addWidget(self.label_show_camera)
        self.widget = QWidget()
        self.widget.setLayout(self.Hlayout)

        self.__layout_fun_button.addStretch(1)
        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addStretch(1)
        self.__layout_fun_button.addWidget(self.button_signin)
        self.__layout_fun_button.addStretch(1)
        # self.__layout_fun_button.addWidget(self.button_close)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.widget)

        # self.palette1 = QPalette()
        # self.palette1.setBrush(self.backgroundRole(), QColor(228, 255, 224))
        # self.setPalette(self.palette1)

        self.setLayout(self.__layout_main)
        self.setWindowTitle(u'臉部辨識登入')

        # 设置背景图片
        # self.palette1 = QPalette()
        # self.palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('img_2465_1920.jpg')))
        # self.setPalette(self.palette1)


    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.button_signin.clicked.connect(self.button_signin_click)
        self.timer_camera.timeout.connect(self.show_camera)
        # self.button_close.clicked.connect(self.close)

    def button_signin_click(self):
        if self.timer_camera.isActive() == False:
            msg = QtWidgets.QMessageBox.warning(self, u"警告", u"請先點選'打開相機'選項。",
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            rval, frame = self.cap.read()
            result = self.member_login(frame)
            if result == 'not found':
                msg = QtWidgets.QMessageBox.warning(self, u"提醒", u"請先於註冊再進行登錄。",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                msg = QtWidgets.QMessageBox.warning(self, u"提醒",
                                                    u"你好，歡迎{} 先生/小姐登入系統。".format(result),
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                self.is_checkout_signal.emit(result)

    # input cv image encoding and single member features
    def find_match(self, cv_encoding, feature):
        results_distance = face_recognition.face_distance(cv_encoding, feature)
        print(results_distance)
        results = face_recognition.compare_faces(cv_encoding, feature, tolerance=0.4)
        check_true = results.count(True)
        print('check_true:', check_true)
        # check if feature matches
        if (check_true >= 1):
            found = True
        else:
            found = False
        return found

    # input cv image and features list
    def member_login(self, frame):
        features_dir = './save_signup_features/member_features.csv'
        features_list = pd.read_csv(features_dir, index_col=0, encoding='utf-8')
        print('features_list:', features_list)
        df_cols = features_list.shape[1]
        print('features_list.shape:', features_list.shape)
        # print('df_cols:', df_cols)
        # change string back to float list
        for i in range(1, df_cols):
            current_col = features_list.columns[i]
            features_list[current_col] = features_list[current_col].apply(eval)

        df_cols = features_list.shape[1]
        cv_feature = face_recognition.face_encodings(frame)

        result_name = 'not found'
        if not cv_feature:
            return result_name

        cv_encoding = cv_feature[0]
        for i in range(1, df_cols):
            current_col = features_list.columns[i]
            # check current column
            found = self.find_match(features_list[current_col].tolist(), cv_encoding)
            if (found):
                result_name = current_col
                return result_name
        return result_name
        # return current column name if found

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"警告", u"請檢測相機與電腦是否連接正確。",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText(u'關閉相機')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'打開相機')

    def show_camera(self):
        flag, self.image = self.cap.read()
        # face = self.face_detect.align(self.image)
        # if face:
        #     pass
        # small_frame = cv2.resize(self.image, (0, 0), fx=0.25, fy=0.25)
        # rgb_small_frame = small_frame[:, :, ::-1]
        # face_locations = face_recognition.face_locations(rgb_small_frame)
        # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # for top, right, bottom, left in face_locations:
        #     top *= 4
        #     right *= 4
        #     bottom *= 4
        #     left *= 4
        #     cv2.rectangle(self.image, (left, top), (right, bottom), (0, 0, 255), 2)
        show = cv2.flip(self.image, 1)
        show = cv2.resize(show, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        w = 300
        h = 300
        x = (640 - w) // 2
        y = (480 - h) // 2
        show = cv2.rectangle(show, (x, y), (x + w, y + h), (255, 0, 0), 2)
        show = cv2.putText(show, 'Please align your face in the box.', (x - 95, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # print(show.shape[1], show.shape[0])
        # show.shape[1] = 640, show.shape[0] = 480
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # self.x += 1
        # self.label_move.move(self.x,100)

        # if self.x ==320:
        #     self.label_show_camera.raise_()

    # def closeEvent(self, event):
    #     ok = QtWidgets.QPushButton()
    #     cacel = QtWidgets.QPushButton()
    #
    #     msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"關閉", u"是否關閉！")
    #
    #     msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
    #     msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
    #     ok.setText(u'確定')
    #     cacel.setText(u'取消')
    #     # msg.setDetailedText('sdfsdff')
    #     if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
    #         event.ignore()
    #     else:
    #         #             self.socket_client.send_command(self.socket_client.current_user_command)
    #         if self.cap.isOpened():
    #             self.cap.release()
    #         if self.timer_camera.isActive():
    #             self.timer_camera.stop()
    #         event.accept()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon("./images/MainWindow.jpg"))
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainWindow = SignInWidget()
    mainWindow.show()
    sys.exit(app.exec_())