# -*- coding:utf-8 -*-
#@Time : 2020/2/26 上午11:55
#@Author: kkkkibj@163.com
#@File : background_modeling.py
#背景帧建模
import numpy as np
import cv2
cap = cv2.VideoCapture('/Users/hongyanma/gitspace/python/python/data/视频源/源视频0.avi')
#fgbg(foreground background)
fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret,frame = cap.read()
    # fgmask forgeground前景掩模,通过apply方法得到前景图像
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()