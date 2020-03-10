# -*- coding:utf-8 -*-
# @Time : 2020/3/6 下午4:57
# @Author: kkkkibj@163.com
# @File : object_bsm_blob.py
import cv2
print(cv2.__version__)

cap = cv2.VideoCapture('/Users/hongyanma/gitspace/python/python/data/视频源/源视频0.avi')
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
trackers = cv2.MultiTracker_create()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    sample_frame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    mog_sub_mask = bg_subtractor.apply(sample_frame)

    th = cv2.threshold(mog_sub_mask.copy(), 128, 255, cv2.THRESH_BINARY)[1]
    lbimg = cv2.medianBlur(th, 3)

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, se)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, se)
    # th = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    # th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    image, contours, hier = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # 获取矩形框边界坐标
        x, y, w, h = cv2.boundingRect(c)
        # 计算矩形框的面积
        area = cv2.contourArea(c)
        if 3000 < area:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # // track
    # cvLabel( & IplImage(fgMaskMOG2), labelImg.get(), blobs);
    # cvFilterByArea(blobs, 64, 10000);
    # cvUpdateTracks(blobs, tracks, 10, 90, 30);
    # cvRenderTracks(tracks, & IplImage(frame), & IplImage(frame));

    backgimage = bg_subtractor.getBackgroundImage()
    cv2.imshow('img', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
