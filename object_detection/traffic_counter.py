# -*- coding:utf-8 -*-
#@Time : 2020/3/3 下午1:10
#@Author: kkkkibj@163.com
#@File : traffic_counter.py
#video vehicle counter

import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/hongyanma/gitspace/python/python/data/视频源/源视频0.avi')

# bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=True)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# bg_subtractor = cv2.createBackgroundSubtractorKNN()

se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # cv2.imwrite('/Users/hongyanma/gitspace/python/python/data/视频源/bg0.jpg', backgimage)
        break
    # mog_sub_mask = bg_subtractor.apply(frame)

    bgimg = cv2.imread("/Users/hongyanma/gitspace/python/python/data/视频源/bg0.jpg")
    bgimg_gray = cv2.cvtColor(bgimg, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th = cv2.subtract(frame_gray, bgimg_gray)
    # th = cv2.cvtColor(th, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(th, 64, 255, cv2.THRESH_BINARY)[1]

    # th = cv2.threshold(mog_sub_mask.copy(), 128, 255, cv2.THRESH_BINARY)[1]
    # th = cv2.threshold(mog_sub_mask.copy(),  128, 255, cv2.THRESH_BINARY_INV)[1] #对图像取反
    # th = cv2.threshold(mog_sub_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    th = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)

    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    #th = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    # th = cv2.morphologyEx(th, cv2.MORPH_OPEN, se)

    # contours, hier = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hier = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # th = cv2.morphologyEx(th, cv2.MORPH_OPEN, se)
    # backgimage = bg_subtractor.getBackgroundImage()

    for c in contours:
        # 获取矩形框边界坐标
        x, y, w, h = cv2.boundingRect(c)
        # 计算矩形框的面积
        # area = cv2.contourArea(c)
        # if 5000 < area < 15000:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # cv2.imwrite('/Users/hongyanma/gitspace/python/python/data/视频源/bg0.jpg',backgimage)
        break