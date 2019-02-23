#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import cv2
import django
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



class SignUpWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(SignUpWidget, self).__init__(parent)

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
        self.count = 1

    def set_ui(self):

        self.__layout_main = QtWidgets.QVBoxLayout()
        self.__layout_fun_button = QtWidgets.QHBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'開啟相機')
        self.button_save_picture = QtWidgets.QPushButton(u'拍攝照片')
        self.button_open_camera.setIcon(QIcon("./light/appbar.camera.png"))
        self.button_open_camera.setIconSize(QtCore.QSize(250, 90))
        self.button_save_picture.setIcon(QIcon("./light/appbar.camera.switch.invert.png"))
        self.button_save_picture.setIconSize(QtCore.QSize(280, 90))
        # self.button_close = QtWidgets.QPushButton(u'退出')

        # Button 的颜色修改
        button_color = [self.button_open_camera, self.button_save_picture]
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
        self.button_save_picture.setFont(font)
        # self.button_open_camera.setMinimumHeight(20)
        # self.button_save_picture.setMinimumHeight(20)
        # self.button_close.setMinimumHeight(50)

        # move()方法移动窗口在屏幕上的位置到x = 300，y = 300坐标。
        self.move(500, 0)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)

        # ID、姓名輸入框
        self.formlayout = QFormLayout()
        font = QFont()
        font.setPixelSize(20)
        lineEditFont = QFont()
        lineEditFont.setPixelSize(20)
        self.IDLabel = QLabel(u'請輸入帳號：')
        self.IDLabel.setFont(font)
        self.IDLabel.setStyleSheet("color:white")
        self.IDLineEdit = QLineEdit()
        self.IDLineEdit.setFixedWidth(300)
        self.IDLineEdit.setFixedHeight(32)
        self.IDLineEdit.setFont(lineEditFont)
        # self.IDLineEdit.setMaxLength(10)
        self.formlayout.addRow(self.IDLabel, self.IDLineEdit)
        self.PWLabel = QLabel(u'請輸入密碼：')
        self.PWLabel.setStyleSheet("color:white")
        self.PWLabel.setFont(font)
        self.PWLineEdit = QLineEdit()
        self.PWLineEdit.setFixedWidth(300)
        self.PWLineEdit.setFixedHeight(32)
        self.PWLineEdit.setFont(lineEditFont)
        self.PWLineEdit.setEchoMode(QLineEdit.Password)
        # self.PWLineEdit.setMaxLength(16)
        self.formlayout.addRow(self.PWLabel, self.PWLineEdit)
        self.NameLabel = QLabel(u'請輸入姓名：')
        self.NameLabel.setStyleSheet("color:white")
        self.NameLabel.setFont(font)
        self.NameLineEdit = QLineEdit()
        self.NameLineEdit.setFixedWidth(300)
        self.NameLineEdit.setFixedHeight(32)
        self.NameLineEdit.setFont(lineEditFont)
        # self.NameLineEdit.setMaxLength(10)
        self.formlayout.addRow(self.NameLabel, self.NameLineEdit)
        self.widget = QWidget()
        self.widget.setLayout(self.formlayout)
        self.widget.setFixedHeight(250)
        self.widget.setFixedWidth(500)
        self.Hlayout = QHBoxLayout()
        self.Hlayout.addWidget(self.widget)
        self.Hlayout.addWidget(self.label_show_camera)
        self.widget = QWidget()
        self.widget.setLayout(self.Hlayout)

        self.__layout_fun_button.addStretch(1)
        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addStretch(1)
        self.__layout_fun_button.addWidget(self.button_save_picture)
        self.__layout_fun_button.addStretch(1)
        # self.__layout_fun_button.addWidget(self.button_close)

        self.__layout_main.addLayout(self.__layout_fun_button)

        self.__layout_main.addWidget(self.widget)

        # self.palette1 = QPalette()
        # self.palette1.setBrush(self.backgroundRole(), QColor(228, 255, 224))
        # self.setPalette(self.palette1)

        self.setLayout(self.__layout_main)
        self.setWindowTitle(u'會員註冊')

        '''
        # 设置背景图片
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('background.jpg')))
        self.setPalette(palette1)
        '''

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.button_save_picture.clicked.connect(self.button_save_picture_click)
        self.timer_camera.timeout.connect(self.show_camera)
        # self.button_close.clicked.connect(self.close)
        self.IDLineEdit.returnPressed.connect(self.button_save_picture_click)
        self.NameLineEdit.returnPressed.connect(self.button_save_picture_click)

    def button_save_picture_click(self):
        if self.timer_camera.isActive() == False:
            msg = QtWidgets.QMessageBox.warning(self, u"警告", u"請先點選'打開相機'選項。",
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            name = self.NameLineEdit.text()
            pw = self.PWLineEdit.text()
            id = self.IDLineEdit.text()
            print(name, id)
            path = "./save_signup_pictures/{}/"
            save_path = path.format(id)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
                for i in range(1, 6):
                    rval, frame = self.cap.read()
                    cv2.imwrite(save_path + name + str(i).zfill(4) + '.jpg', frame)
                    cv2.waitKey(1)
                    time.sleep(1)
            else:
                if id or name =='':
                    msg = QtWidgets.QMessageBox.warning(self, u"警告", u"ID與姓名不得為空。",
                                                        buttons=QtWidgets.QMessageBox.Ok,
                                                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    msg = QtWidgets.QMessageBox.warning(self, u"警告", u"此ID已被註冊過。",
                                                        buttons=QtWidgets.QMessageBox.Ok,
                                                        defaultButton=QtWidgets.QMessageBox.Ok)
            self.create_member(save_path, name)
            #調用Django程序操作model
            pathname = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, pathname)
            sys.path.insert(0, os.path.abspath(os.path.join(pathname, '..')))
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
            django.setup()
            from django.contrib.auth.models import User
            user = User.objects.create_user(id, '', pw)
            user.last_name = name
            user.save()
            msg = QtWidgets.QMessageBox.warning(self, u"提醒", u"已註冊完成。",
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)

    def create_member(self, img_dir, name):
        dummy_coding = [0] * 128
        member_features = []
        features_dir = './save_signup_features/member_features.csv'
        features_list = pd.read_csv(features_dir, index_col=0, encoding='utf-8')
        img_list = glob.glob(img_dir + '*')

        for (i, f) in enumerate(img_list):
            mem_img = face_recognition.load_image_file(f)
            face_encoding = face_recognition.face_encodings(mem_img)
            if face_encoding:
                member_features.append(list(face_encoding[0]))
            else:
                member_features.append(list(dummy_coding))

        # create new member feature
        df_cols = features_list.shape[1]
        # insert new member and save
        features_list.insert(df_cols, name, member_features)
        features_list.to_csv(features_dir, encoding='utf-8')

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
        show = cv2.flip(self.image, 1)
        show = cv2.resize(show, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        w = 300
        h = 300
        x = (640 - w) // 2
        y = (480 - h) // 2
        show = cv2.rectangle(show, (x, y), (x+w, y+h), (255, 0, 0), 2)
        show = cv2.putText(show, 'Please align your face in the box.', (x-95, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # print(show.shape[1], show.shape[0])
        # show.shape[1] = 640, show.shape[0] = 480
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # self.x += 1
        # self.label_move.move(self.x, 100)

        # if self.x ==320:
        #     self.label_show_camera.raise_()

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"關閉", u"是否關閉！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'確定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon("./images/MainWindow.jpg"))
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainWindow = SignUpWidget()
    mainWindow.show()
    sys.exit(app.exec_())
