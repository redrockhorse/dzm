# -*- coding:utf-8 -*-
# @Time : 2020/2/26 下午1:18
# @Author: kkkkibj@163.com
# @File : target_tracking.py
# tracking

import cv2
import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
tracker = cv2.TrackerKCF_create()




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

cap = cv2.VideoCapture('/Users/hongyanma/gitspace/python/python/data/视频源/源视频0.avi')
# cap = cv2.VideoCapture('/Users/hongyanma/Downloads/trafficvideo/sxjgl_sdsdgs_370000011011001001265.flv')

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(size)

# ShiTomasi 角点检测参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=100)

# lucas kanade光流法参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2)

# 创建随机颜色
color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# 创建一个蒙版用来画轨迹
mask = np.zeros_like(old_frame)

while cap.isOpened():
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    img = frame
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
    else:
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        print(p0)

    cv2.imshow('frame', img)
    key = cv2.waitKey(100) & 0xFF
    if key == ord("q"):
        break

    # 更新上一帧的图像和追踪点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
