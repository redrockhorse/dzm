# -*- coding:utf-8 -*-
#@Time : 2020/3/5 下午5:29
#@Author: kkkkibj@163.com
#@File : sift_test.py
import cv2
print(cv2.__version__)
sift = cv2.xfeatures2d.SIFT_create()

# FLANN 参数设计
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

img1 = cv2.imread("/Users/hongyanma/Desktop/wood.jpeg")
img2 = cv2.imread("/Users/hongyanma/Desktop/target.jpeg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
kp1, des1 = sift.detectAndCompute(img1,None)#des是描述子

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = sift.detectAndCompute(img2,None)

matches = flann.knnMatch(des1,des2,k=2)
matchesMask = [[0,0] for i in range(len(matches))]
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(good)
