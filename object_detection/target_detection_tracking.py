# -*- coding:utf-8 -*-
# @Time : 2020/2/26 下午1:18
# @Author: kkkkibj@163.com
# @File : target_tracking.py
# tracking

import cv2
import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
# tracker = cv2.TrackerKCF_create()
tracker = cv2.MultiTracker_create()


def bbOverlap(box1, box2):
    if box1[0] > box2[0] + box2[2]:
        return 0.0
    if box1[1] > box2[1] + box2[3]:
        return 0.0
    if box1[0] + box1[2] < box2[0]:
        return 0.0
    if box1[1] + box1[3] < box2[1]:
        return 0.0
    colInt = min(box1[0] + box1[2], box2[0] + box2[2]) - max(box1[0], box2[0])
    rowInt = min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1])
    intersection = colInt * rowInt
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    return intersection / (area1 + area2 - intersection)
    # return intersection / min(area1,area2)


def perceptualHashAlgorithm(img1):
    dim = (8, 8)
    img1_shrink = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img1_shrink_gray = cv2.cvtColor(img1_shrink, cv2.COLOR_BGR2GRAY)
    arr = []
    for i in range(8):
        for j in range(8):
            arr.append(img1_shrink_gray[i][j])
    avg = np.mean(arr)
    binary_ret = 0
    for n in range(64):
        if arr[n] >= avg:
            binary_ret += 1 << n
    return binary_ret


def perceptualHashDif(h1, h2):
    t = h1 ^ h2
    return bin(t).count('1')


detector = cv2.ORB_create()


def getDesOrb(img):
    kp1 = detector.detect(img, None)
    kp1, des1 = detector.compute(img, kp1)
    return des1


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)


def orb_match(des1, des2):
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask += 1
    return matchesMask / len(matches)


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
initBB = None
waitTime = 1000
trackerList = []
while cap.isOpened():
    ret, image_np = cap.read()
    np.expand_dims(image_np, axis=0)
    net.setInput(cv2.dnn.blobFromImage(image_np, size=(300, 300), swapRB=True, crop=False))
    cvOut = net.forward()
    print('-----------------------')
    for detection in cvOut[0, 0, :, :]:
        score = float('%.2f' % detection[2])
        labelName = labelMap[int(detection[1])]
        if score > 0.3:
            left = detection[3] * size[0] if detection[3] * size[0] > 0 else 0
            top = detection[4] * size[1] if detection[4] * size[1] > 0 else 0
            right = detection[5] * size[0]
            bottom = detection[6] * size[1]
            w = int(right - left)
            h = int(bottom - top)
            if w > 0.4 * size[0] or h > 0.4 * size[1] or w < 0.1 * size[0] or h < 0.1 * size[1]:
                continue
            initBB = (int(left), int(top), int(right - left), int(bottom - top))
            objectImg = image_np[int(top):int(bottom), int(left):int(right)]
            # cv2.imshow('object', objectImg)
            # key = cv2.waitKey(100) & 0xFF
            des_n = getDesOrb(objectImg)
            track_flag = True
            for des_o in trackerList:
                n = orb_match(des_n, des_o)
                print(n)
                if n > 0:
                    #trackerList.remove(des_o)
                    track_flag = False
                    continue

            if track_flag:
                # ok = tracker.add(cv2.TrackerKCF_create(), image_np, initBB)
                trackerList.append(des_n)

    waitTime = 10
    ok, boxes = tracker.update(image_np)
    # print('tracker number: ', len(tracker.getObjects()))
    # trackerList = []
    if ok:
        # print('box length: ',len(boxes))
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            # print(x, y, w, h)
            # trackerList.append((x, y, w, h))
            cv2.rectangle(image_np, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
    else:
        tracker = cv2.MultiTracker_create()

    cv2.imshow('img', image_np)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
