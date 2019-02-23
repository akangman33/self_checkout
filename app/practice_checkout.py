#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QFont, QIcon, QColor
from PyQt5.QtWebEngineWidgets import *
import os
import time
import datetime
import sqlite3
import face_recognition
import glob
import numpy as np
import pandas as pd
import qdarkstyle
import copy
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from practice_signin import SignInWidget
from practice_detailbrowser import DetailBrowser
import django


class CheckOutWidget(QtWidgets.QWidget):
    def __init__(self, result):
        super().__init__()
        with open('./cat_to_name_13.json', 'r') as f:
            self.cat_to_name = json.load(f)
        self.model = self.rebuild_model(filename='resnet18_cuda_drinks_190211_ep110.pth')
        self.lines = []
        self.signin_name = result
        print('self.signin_name:', self.signin_name)
        # self.face_recong = face.Recognition()
        self.resize(900, 600)
        self.timer_camera = QtCore.QTimer()
        self.timer_checkout = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.CAM_NUM = 1
        # self.CAM_NUM = './Demo_T4.mp4'
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0

    def set_ui(self):
        self.__layout_main = QtWidgets.QVBoxLayout()
        self.__layout_fun_button = QtWidgets.QHBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'開啟相機')
        self.button_checkout = QtWidgets.QPushButton(u'進行商品結帳')
        self.button_detail = QtWidgets.QPushButton(u'商店消費查詢')
        self.button_open_camera.setIcon(QIcon("./light/appbar.camera.png"))
        self.button_open_camera.setIconSize(QtCore.QSize(250, 90))
        self.button_checkout.setIcon(QIcon("./light/appbar.billing.png"))
        self.button_checkout.setIconSize(QtCore.QSize(250, 90))
        self.button_detail.setIcon(QIcon("./light/appbar.billing.png"))
        self.button_detail.setIconSize(QtCore.QSize(250, 90))
        # self.button_close = QtWidgets.QPushButton(u'退出')

        # Button 的颜色修改
        button_color = [self.button_open_camera, self.button_checkout, self.button_detail]
        for i in range(3):
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
        self.button_checkout.setFont(font)
        self.button_detail.setFont(font)
        # self.button_open_camera.setMinimumHeight(20)
        # self.button_checkout.setMinimumHeight(20)
        # self.button_close.setMinimumHeight(50)

        # move()方法移动窗口在屏幕上的位置到x = 300，y = 300坐标。
        self.move(500, 0)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)
        self.label_show_picture = QtWidgets.QLabel()
        self.label_show_picture.setFixedSize(641, 481)
        self.label_show_picture.setAutoFillBackground(False)
        # 增加一个table
        self.table = QTableWidget()
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)  # 设置字体加粗
        self.table.setFont(font)
        self.table.horizontalHeader().setFont(font)  # 设置表头字体
        self.table.setColumnCount(3)  ##设置表格一共有五列
        self.table.setHorizontalHeaderLabels(['商品名稱', '數量', '價錢'])  # 设置表头文字
        # table.setFrameShape(QFrame.NoFrame)  ##设置无表格的外框
        self.table.horizontalHeader().setFixedHeight(50)  ##设置表头高度
        self.table.setColumnWidth(0, 500)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 150)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)  # 设置第五列宽度自动调整，充满屏幕
        # table.horizontalHeader().setStretchLastSection(True)  ##设置最后一列拉伸至最大
        # table.setSelectionMode(QAbstractItemView.SingleSelection)  # 设置只可以单选，可以使用ExtendedSelection进行多选
        # table.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置 不可选择单个单元格，只可选择一行。
        self.table.horizontalHeader().resizeSection(0, 200)  # 设置第一列的宽度为200
        self.table.horizontalHeader().setSectionsClickable(False)  # 可以禁止点击表头的列
        # table.sortItems(1, Qt.DescendingOrder)  # 设置按照第二列自动降序排序
        self.table.horizontalHeader().setStyleSheet('QHeaderView::section{background:#000079;color:#FFFFFF}')
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可更改
        self.table.setSortingEnabled(True)  # 设置表头可以自动排序

        self.Hlayout = QHBoxLayout()
        self.Hlayout.addWidget(self.label_show_camera)
        self.Hlayout.addWidget(self.label_show_picture)
        self.Hlayout.addWidget(self.table)
        self.widget = QWidget()
        self.widget.setLayout(self.Hlayout)

        self.__layout_fun_button.addStretch(1)
        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addStretch(1)
        self.__layout_fun_button.addWidget(self.button_checkout)
        self.__layout_fun_button.addStretch(1)
        self.__layout_fun_button.addWidget(self.button_detail)
        self.__layout_fun_button.addStretch(1)
        # self.__layout_fun_button.addWidget(self.button_close)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.widget)

        # self.palette1 = QPalette()
        # self.palette1.setBrush(self.backgroundRole(), QColor(228, 255, 224))
        # self.setPalette(self.palette1)

        self.setLayout(self.__layout_main)
        self.setWindowTitle(u'商品結帳')

        '''
        # 设置背景图片
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('background.jpg')))
        self.setPalette(palette1)
        '''

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.button_checkout.clicked.connect(self.button_checkout_click)
        self.button_detail.clicked.connect(self.button_detail_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_checkout.timeout.connect(self.show_checkout)
        # self.button_close.clicked.connect(self.close)
        self.table.cellChanged.connect(self.cellchange)

    def bounding_box_pre(self, img, save_path='', show_binary=False, binary_thred=90,
                         can1=10, can2=150, can3=1,
                         dilate0=0, erode1=0,
                         dilate1=0, erode2=0,
                         dilate2=0, choose=0):
        # img = cv2.imread(imgpath)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # _, binary = cv2.threshold(gray_img, binary_thred, 255,cv2.THRESH_BINARY)
        cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        _, binary = cv2.threshold(gray_img, binary_thred, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(img, can1, can2, can3)
        bin_edge = cv2.add(binary, edges)
        edge_0 = cv2.dilate(bin_edge, np.ones((3, 3)), iterations=dilate0)
        edge_1 = cv2.erode(edge_0, np.ones((3, 3)), iterations=erode1)
        edge_2 = cv2.dilate(edge_1, np.ones((3, 3)), iterations=dilate1)
        edge_3 = cv2.erode(edge_2, np.ones((3, 3)), iterations=erode2)
        edge_4 = cv2.dilate(edge_3, np.ones((3, 3)), iterations=dilate2)

        # cv2.imshow('total', (edges, edge_0, edge_1, edge_2))

        # if show_binary:
        #     cv2.imshow("Binary", binary)
        #     cv2.imshow("Edge", edges)
        #     cv2.imshow("Bin+Edge:", bin_edge)
        #     cv2.imshow('dilate0', edge_0)
        #     cv2.imshow('errode1', edge_1)
        #     cv2.imshow('dilate1', edge_2)
        #     cv2.imshow('errode2', edge_3)
        #     cv2.imshow('dilate2', edge_4)

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cnts, _ = cv2.findContours(edge_4, 1, 2)
        box_list = []
        if len(cnts) > 0:
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                box_list.append((x, y, w, h))

        return box_list

    def remove_box_in_box(self, test_list, log_enable=False):
        final_list = []
        bypass_flag_1 = 0
        for i in range(len(test_list)):
            bypass_flag = 0
            if bypass_flag_1:
                bypass_flag_1 = 0
                continue
            if log_enable:
                print('i:', i, test_list[i])
                print('*' * 35)
            for j in range(len(test_list) - i - 1):
                if log_enable:
                    print('[j+i+1]:', j + i + 1, test_list[j + i + 1])
                # check box in box condition in 2 cases
                if (test_list[i][0] < test_list[j + i + 1][0] and
                        test_list[i][1] < test_list[j + i + 1][1] and
                        test_list[i][0] + test_list[i][2] >= test_list[j + i + 1][0] + test_list[j + i + 1][2] and
                        test_list[i][1] + test_list[i][3] >= test_list[j + i + 1][1] + test_list[j + i + 1][3]):
                    if log_enable:
                        print("Found this box {} in current box {}!!".format(test_list[j + i + 1], test_list[i]))
                        print("Drop this box {}".format(test_list[j + i + 1]))
                    bypass_flag_1 = 1
                elif (test_list[i][0] > test_list[j + i + 1][0] and
                      test_list[i][1] > test_list[j + i + 1][1] and
                      test_list[i][0] + test_list[i][2] <= test_list[j + i + 1][0] + test_list[j + i + 1][2] and
                      test_list[i][1] + test_list[i][3] <= test_list[j + i + 1][1] + test_list[j + i + 1][3]):
                    if log_enable:
                        print("Found current box {} in this box {}!!".format(test_list[i], test_list[j + i + 1]))
                        print("Drop current box {}".format(test_list[i]))
                    bypass_flag = 1
                    break
            if log_enable:
                print('*' * 35)
            if bypass_flag:
                continue
            final_list.append(test_list[i])
        return final_list

    def box_filter(self, boxs):
        boxs_f = []
        if len(boxs) > 0:
            for box in boxs:
                if box[2] < 80 and box[3] < 80:  # filter small size(<80) boxs
                    continue
                elif box[3] > 350 or box[2] * box[3] > 50000:
                    continue
                elif box[0] + box[2] > 620:
                    continue
                elif box[0] < 20:
                    continue
                elif box[1] + box[3] > 460:
                    continue
                if box[1] < 5 or box[0] > 600:
                    continue
                else:
                    boxs_f.append(box)
            # print('boxs_f:', boxs_f)
            return boxs_f
        else:
            print('no boxs found')

    def six_box_color(self, box_cnt):
        if box_cnt % 6 == 0:
            color = (255, 0, 255)
        elif box_cnt % 6 == 1:
            color = (255, 255, 0)
        elif box_cnt % 6 == 2:
            color = (0, 255, 255)
        elif box_cnt % 6 == 3:
            color = (0, 255, 0)
        elif box_cnt % 6 == 4:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        return color

    def plot_boxs(self, frame0, boxs):
        # new_frame = frame0.copy()
        box_cnt = 0
        if len(boxs):
            for box in boxs:
                color = self.six_box_color(box_cnt)
                text = 'object' + str(box_cnt + 1)
                box_cnt += 1
                cv2.rectangle(frame0, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
                cv2.rectangle(frame0, (box[0] - 1, box[1] - 18), (box[0] + box[2] // 3 * 2, box[1]), color, -1)
                cv2.putText(frame0, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 0), 1, cv2.LINE_AA)
            # return frame0
        else:
            print('no boxs found')

    # crop_image >> process image with particular bounding box
    def crop_image_pre(self, img, box):
        new_img = np.zeros_like(img)
        w = box[2]
        h = box[3]
        x = (img.shape[1] - w) // 2
        y = (img.shape[0] - h) // 2
        new_img[y:y + h, x:x + w, :] = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :]
        return new_img

    # wrap crop image with box list
    def crop_img_with_box_list(self, img, box_list, model,
                               save_crop_image=False,
                               show_img_on_jupyter=False,
                               prediction=False):
        result_img = img.copy()
        cls_list = []
        probs_list = []
        for idx, box in enumerate(box_list):
            img_crop = self.crop_image_pre(img, box)
            if save_crop_image:
                cv2.imwrite('./crop_' + str(idx + 1) + '.jpg', img_crop)
            if show_img_on_jupyter:
                plt.figure(figsize=[7, 7])
                # plt.subplot(4,3,idx+1)
                plt.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
                plt.xticks([]), plt.yticks([])
            if prediction:
                print('Do prediction now')
                try:
                    probs, cls = self.model_predict(img_crop, model, topk=1)
                    color = self.six_box_color(idx)
                    cv2.rectangle(result_img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
                    cv2.rectangle(result_img, (box[0] - 1, box[1] - 18), (box[0] + box[2] // 4 * 3, box[1]), color, -1)
                    cv2.putText(result_img, cls[0] + '_' + str(probs[0]), (box[0], box[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 0, 0), 1, cv2.LINE_AA)
                    cls_list.append(cls[0])
                    probs_list.append(probs[0])

                    # cv2.imwrite('./crop_'+str(idx+1)+'.jpg', img_crop)
                    # top5_prdeiction('./crop_'+str(idx+1)+'.jpg', model)
                except:
                    print('prediction error')
        return (cls_list, probs_list), result_img

    def rebuild_model(self, filename='checkpoint.pth', device=torch.device("cpu")):

        if device == torch.device("cpu"):
            checkpoint = torch.load(filename, map_location=device)
        elif device == torch.device('cuda'):
            # checkpoint = torch.load(filename, map_location="cuda:0")
            checkpoint = torch.load(filename)
        else:
            print('Error!! checkpoint read fail, check your device setting.')
            return 1

        model = models.resnet18(pretrained=True)  # recall pre-train mode vgg16
        for params in model.parameters():  # freeze pre-train model parameters
            params.require_grad = False
        print('building resnet18 pretrain_model...')  # message
        model.to(device)  # set computaion unit

        model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 14),
            nn.LogSoftmax(dim=1)
        )
        print('building model classifier with full connection:',  # message
              '\ninput_size:512',
              '\nhidden_size:256',
              '\noutput_size:14'
              )
        dummy = {}  # dummy diction
        if type(checkpoint) == type(dummy):  # comfirm checkpoint is type of diction
            if 'state_dict' in list(checkpoint.keys()):
                print('update model stat_dict...')
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print('no state_dict in this checkpoint')
            if 'class_to_idx' in list(checkpoint.keys()):
                print('update model class_to_idx...')
                model.class_to_idx = checkpoint['class_to_idx']
            else:
                print('no class_to_idx in this checkpoint')
        else:
            print('no stat_dict & class_to_idx could be updated',
                  'you should retrain this model again.'
                  )
        print('rebuild model finished!!')  # message
        return model

    # cv2 resize to keep aspect ratio
    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def img_preprocess_for_model(self, np_img_or_filepath, resize=256,
                                 center_crop_size=224, plot_img=False):
        '''image preprocess for pytorch model'''
        # load img from np.array or from file
        try:
            img_cv = cv2.imread(np_img_or_filepath)

        except TypeError:
            img_cv = np_img_or_filepath

        # resize shorter edge to 256
        if img_cv.shape[0] <= img_cv.shape[1]:
            img_resize = self.image_resize(img_cv, height=resize, inter=cv2.INTER_AREA)
        else:
            img_resize = self.image_resize(img_cv, width=resize, inter=cv2.INTER_AREA)

        # plot original size img and crop img size
        if plot_img:
            plt.figure(figsize=[20, 30])
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            plt.title("original size:{}".format(img_cv.shape[:2]))
            plt.subplot(122)
            plt.imshow(cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB))
            plt.title("crop size:{}".format(img_resize.shape[:2]))

        # center crop to img_size (224, 224)
        y_min = (img_resize.shape[0] - center_crop_size) // 2
        x_min = (img_resize.shape[1] - center_crop_size) // 2
        y_max = (img_resize.shape[0] - center_crop_size) // 2 + center_crop_size
        x_max = (img_resize.shape[1] - center_crop_size) // 2 + center_crop_size
        img_center_crop = np.zeros((center_crop_size, center_crop_size, 3), dtype='uint8')
        img_center_crop = img_resize[y_min:y_max, x_min:x_max, :]

        # convet color chanel from BGR to RGB
        img_RGB = cv2.cvtColor(img_center_crop, cv2.COLOR_BGR2RGB)

        # Nomalization with means and stderr
        means = np.array([0.485, 0.456, 0.406])
        stderr = np.array([0.229, 0.224, 0.225])
        nor_img = (img_RGB / 255 - means) / stderr

        if plot_img:
            plt.figure(figsize=[20, 30])
            plt.subplot(121)
            plt.imshow(img_RGB)
            plt.title("Befor Nomarlization size:{}".format(img_RGB.shape[:2]))
            plt.subplot(122)
            # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
            # image = np.clip(image, 0, 1)
            plt.imshow(np.clip(nor_img, 0, 1))
            plt.title("After Nomarlization size:{}".format(nor_img.shape[:2]))

        # transpose color chanel to first dimension for pytorch tensor
        img_trans = nor_img.transpose((2, 1, 0))

        return img_trans

    def model_predict(self, image_path, model, device=torch.device("cpu"), topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

        # TODO: Implement the code to predict the class from an image file

        # image preprocessing by function process_image(), it return a numpy_image
        np_imgs = self.img_preprocess_for_model(image_path)
        # transfer numpy_image to tensor_image (FloatTensor >> same datatype as model weight)
        tensor_imgs = torch.from_numpy(np_imgs).type(torch.FloatTensor)
        # incread to 4 dimemtion tensor using unsqueeze function
        if device == torch.device("cpu"):
            tensor_imgs_1 = tensor_imgs.unsqueeze(0)
        elif device == torch.device("cuda"):
            tensor_imgs_0 = tensor_imgs.unsqueeze(0).cpu()
            tensor_imgs_1 = tensor_imgs_0.type(torch.cuda.FloatTensor)

        model.to(device)
        model.eval()
        # disable autograd for prediction (like validation and test)
        with torch.set_grad_enabled(False):
            output = model.forward(tensor_imgs_1)

        # convert output to probability by exponential func
        probs = torch.exp(output)
        # choosing top5 probs and idxs with topk function
        probs_tp5, idx_tp5 = torch.topk(probs, topk)

        if device == torch.device("cuda"):
            probs_tp5_1 = probs_tp5.cpu()
            # probs_tp5_1 = probs_tp5_cpu.type(torch.cuda.FloatTensor)
            idx_tp5_1 = idx_tp5.cpu()
            # idx_tp5_1 = idx_tp5_cpu.type(torch.cuda.FloatTensor)
        else:
            probs_tp5_1 = probs_tp5
            idx_tp5_1 = idx_tp5

        # reload mapping dictionary from model attribute
        class_to_idx = model.class_to_idx
        # reverse class_to_idx to idx_to_class
        idx_to_class = {str(value): key for key, value in class_to_idx.items()}

        # conver top5 idx to top5 class
        class_tp5 = [idx_to_class[str(i)] for i in idx_tp5_1[0].numpy()]

        # return top5 probability and class idx

        # reverse top5 class index to top5 name
        top5_name = [self.cat_to_name[i] for i in class_tp5]

        return probs_tp5_1[0].numpy(), top5_name

    # classification
    def object_classification(self, cap, model):
        if cap.isOpened:
            print("Camera is opened")
        else:
            print("Can't Open Camera")

        ref, frame = cap.read()
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        frame0 = frame.copy()
        try:
            # find bounding box
            boxs = self.bounding_box_pre(frame0, save_path='', show_binary=True,
                                         can1=250, can2=255, can3=3,
                                         dilate0=1, erode1=0,
                                         dilate1=0, erode2=0,
                                         dilate2=0, choose=0)
            # filter boxs
            boxs_f0 = self.box_filter(boxs)
            boxs_f1 = self.remove_box_in_box(boxs_f0)
            self.plot_boxs(frame0, boxs_f1)
        except IndexError:
            print("No object")

        result_list, result_img = self.crop_img_with_box_list(frame, boxs_f1, model, prediction=True)
        return result_list, result_img

    def object_boundingbox(self, cap):
        # if cap.isOpened:
        #     print("Camera is opened")
        # else:
        #     print("Can't Open Camera")

        ref, frame = cap.read()
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        try:
            # find bounding box
            boxs = self.bounding_box_pre(frame, save_path='', show_binary=True,
                                         can1=250, can2=255, can3=3,
                                         dilate0=1, erode1=0,
                                         dilate1=0, erode2=0,
                                         dilate2=0, choose=0)
            # filter boxs
            boxs_f0 = self.box_filter(boxs)
            boxs_f1 = self.remove_box_in_box(boxs_f0)
            self.plot_boxs(frame, boxs_f1)
        except IndexError:
            print("No object")

        return frame

    def cellchange(self,row,col):
        print('6')
        item = self.table.item(row,col)
        self.txt = item.text()
        txtfont = QFont()
        txtfont.setPointSize(20)
        self.txt.setFont(txtfont)
    #     self.settext('第%s行，第%s列 , 数据改变为:%s'%(row,col,txt))
    #
    # def settext(self,txt):
    #     font = QFont('微软雅黑',30)
    #     self.txt.setFont(font)
    #     self.txt.setText(txt)

    def button_checkout_click(self):
        if self.timer_camera.isActive() == False:
            msg = QtWidgets.QMessageBox.warning(self, u"警告", u"請先點選'打開相機'選項。",
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.timer_checkout.start(30)

    def show_checkout(self):
        result_list, result_img = self.object_classification(self.cap, model=self.model)
        result_name = result_list[0]
        show = cv2.resize(result_img, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_picture.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.timer_checkout.stop()

        self.removelines = []
        for line in self.lines:
            row = self.table.rowCount()
            for x in range(row, 0, -1):
                if line[0] == self.table.item(x - 1, 0).text():
                    self.table.removeRow(x - 1)
                    self.removelines.append(line)
        for line in self.removelines:
            self.lines.remove(line)

        self.table.cellChanged.disconnect()
        for name in result_name:
            row = self.table.rowCount()
            self.table.setRowCount(row+1)
            name_list = {"Black_Tea": "紅茶", "Cheers": "Cheers礦泉水", "Chrunchoco": "巧克力脆片",
                         "Coffee_Milk": "咖啡牛奶", "Family_Water": "全家礦泉水", "Green_Milk_Tea": "奶綠",
                         "Lays": "洋芋片", "LP33": "優酪乳", "Lemon_Tea": "檸檬紅茶", "Lotte": "小熊餅乾",
                         "LS_SoyMilk": "低糖豆漿", "Oats_Drink": "喝的燕麥", "Ocean_Spray": "蔓越莓",
                         "Oolong_Tea": "烏龍奶茶", "Oreo": "Oreo餅乾", "Puff": "小泡芙","Purple": "紫米燕麥米漿",
                         "Soy_Oats": "豆漿燕麥", "Soymilk": "豆漿", "With_Kernel": "顆粒燕麥"}
            price_list = {"Black_Tea": "25", "Cheers": "40", "Chrunchoco": "60", "Coffee_Milk": "20",
                          "Family_Water": "15", "Green_Milk_Tea": "25", "Lays": "30", "LP33": "30",
                          "Lemon_Tea": "20", "Lotte": "40", "LS_SoyMilk": "20", "Oats_Drink": "40",
                          "Ocean_Spray": "25", "Oolong_Tea": "20", "Oreo": "50", "Puff": "30",
                          "Purple": "20", "Soy_Oats": "40", "Soymilk": "20", "With_Kernel": "40"}
            commodity_name = name_list[name]
            price = price_list[name]
            count = '1'
            self.table.setItem(row, 0, QTableWidgetItem(commodity_name))
            self.table.setItem(row, 1, QTableWidgetItem(count))
            self.table.setItem(row, 2, QTableWidgetItem(price))
            self.lines.append([commodity_name, count, price])

        msg = QtWidgets.QMessageBox.warning(self, u"商品結帳", u"商品結帳已完成。",
                                            buttons=QtWidgets.QMessageBox.Ok,
                                            defaultButton=QtWidgets.QMessageBox.Ok)
        print('self.lines:', self.lines)
        # 調用Django程序操作model
        pathname = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, pathname)
        sys.path.insert(0, os.path.abspath(os.path.join(pathname, '..')))
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
        django.setup()
        from app.models import Category, Record
        from django.contrib.auth.models import User
        username = self.signin_name
        user = User.objects.get(last_name=username)
        id = user.id
        for word in self.lines:
            name = word[0]
            count = word[1]
            price = word[2]
            category = Category(category=name, user_id=id)
            unix = time.time()
            date = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d'))
            record = Record(date=date, category=name, cash=price, cnt=count, user_id=id)
            category.save()
            record.save()

        self.table.cellChanged.connect(self.cellchange)

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
        image = self.object_boundingbox(self.cap)
        # face = self.face_detect.align(self.image)
        # if face:
        #     pass
        show = cv2.resize(image, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        # print(show.shape[1], show.shape[0])
        # show.shape[1] = 640, show.shape[0] = 480
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # self.x += 1
        # self.label_move.move(self.x,100)

        # if self.x ==320:
        #     self.label_show_camera.raise_()

    def button_detail_click(self):
        detail = DetailBrowser(self)
        detail.show()

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
    mainWindow = CheckOutWidget()
    mainWindow.show()
    sys.exit(app.exec_())