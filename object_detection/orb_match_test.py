# -*- coding:utf-8 -*-
# @Time : 2020/3/6 下午2:14
# @Author: kkkkibj@163.com
# @File : orb_match_test.py
import cv2
import numpy as np

img1 = cv2.imread("/Users/hongyanma/Desktop/wood.jpeg")
img2 = cv2.imread("/Users/hongyanma/Desktop/target.jpeg")

#ORB搜索特征
detector = cv2.ORB_create()
kp1 = detector.detect(img1, None)
kp2 = detector.detect(img2, None)
kp1, des1 = detector.compute(img1, kp1)
kp2, des2 = detector.compute(img2, kp2)

# 暴力匹配
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# knn匹配
# des1 = cv2.cvtColor(des1, cv2.COLOR_BGR2GRAY)
# des2 = cv2.cvtColor(des2, cv2.COLOR_BGR2GRAY)
des1 = np.float32(des1)
des2 = np.float32(des2)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
print(len(des1))
print(len(des2))
print(len(matches))
matchesMask = 0
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask += 1
print(matchesMask/len(matches))
print(matchesMask)
print(matches)
