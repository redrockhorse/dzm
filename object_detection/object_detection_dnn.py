# -*- coding:utf-8 -*-
# @Time : 2020/2/26 下午1:18
# @Author: kkkkibj@163.com
# @File : target_tracking.py
# object detection opencv-dnn

import cv2
import numpy as np


def makeLableMap():
    labelfiledir = "/Users/hongyanma/gitspace/python/python/test/data/mscoco_label_map.pbtxt"
    with open(labelfiledir, 'r') as labelfile:
        labelMap = ['nul']
        i = 0
        while True:
            line = labelfile.readline()
            if line:
                if i % 5 == 3:
                    labelMap.append(line.split(":")[1].replace('"', '').replace('\r', '').replace('\n', ''))
                i += 1
            else:
                break
        return labelMap


labelMap = makeLableMap()
modelConfiguration = "/Users/hongyanma/gitspace/python/python/test/ssd_mobilenet_v1_coco_2018_01_28/graph.pbtxt"
modelBinary = "/Users/hongyanma/gitspace/python/python/test/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb"
net = cv2.dnn.readNetFromTensorflow(modelBinary, modelConfiguration)

# cap = cv2.VideoCapture('/Users/hongyanma/gitspace/python/python/data/视频源/源视频0.avi')
cap = cv2.VideoCapture('/Users/hongyanma/Downloads/trafficvideo/sxjgl_sdsdgs_370000011011001001265.flv')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
while cap.isOpened():
    ret, image_np = cap.read()
    np.expand_dims(image_np, axis=0)
    net.setInput(cv2.dnn.blobFromImage(image_np, size=(300, 300), swapRB=True, crop=False))
    cvOut = net.forward()
    for detection in cvOut[0, 0, :, :]:
        score = float('%.2f' % detection[2])
        labelName = labelMap[int(detection[1])]
        if score > 0.3:
            left = detection[3] * size[0]
            top = detection[4] * size[1]
            right = detection[5] * size[0]
            bottom = detection[6] * size[1]
            cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            cv2.putText(image_np, labelName, (int(right), int(bottom)), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 230, 210), 2)
    cv2.imshow('img', image_np)
    cv2.waitKey(1)
