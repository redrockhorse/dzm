# -*- coding:utf-8 -*-
#@Time : 2020/3/6 上午9:53
#@Author: kkkkibj@163.com
#@File : shi-tomasi.py

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/hongyanma/Desktop/wood.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    print(corners)
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)



plt.imshow(img),plt.show()