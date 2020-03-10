# -*- coding:utf-8 -*-
#@Time : 2020/2/26 下午1:18
#@Author: kkkkibj@163.com
#@File : target_tracking.py
#tracking

import cv2
import numpy as np
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
tracker = cv2.TrackerKCF_create()

def makeLableMap():
   labelfiledir = "/Users/hongyanma/gitspace/python/python/test/data/mscoco_label_map.pbtxt"
   with open(labelfiledir,'r') as labelfile:
       labelMap = ['nul']
       i = 0
       while True:
           line = labelfile.readline()
           if line:
               if i%5==3:
                  labelMap.append(line.split(":")[1].replace('"','').replace('\r','').replace('\n',''))
               i+=1
           else:
               break
       return labelMap

labelMap = makeLableMap()

modelConfiguration = "/Users/hongyanma/gitspace/python/python/test/ssd_mobilenet_v1_coco_2018_01_28/graph.pbtxt"
modelBinary = "/Users/hongyanma/gitspace/python/python/test/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb"
net = cv2.dnn.readNetFromTensorflow(modelBinary, modelConfiguration)


# cap = cv2.VideoCapture('/Users/hongyanma/gitspace/python/python/data/视频源/源视频0.avi')
cap = cv2.VideoCapture('/Users/hongyanma/Downloads/trafficvideo/sxjgl_sdsdgs_370000011011001001265.flv')

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(size)
initBB = None
waitTime = 1000
while cap.isOpened():
    ret, image_np = cap.read()
    np.expand_dims(image_np, axis=0)

    if initBB is not None:
        waitTime = 100
        success, box = tracker.update(image_np)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(image_np, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
        else:
            initBB = None

    cv2.imshow('img', image_np)
    key = cv2.waitKey(waitTime) & 0xFF

    if key == ord("s"):
        initBB = cv2.selectROI("Frame", image_np, fromCenter=False, showCrosshair=True)
        print(initBB)
        tracker = cv2.TrackerKCF_create()
        tracker.init(image_np, initBB)
    elif key == ord("q"):
        break