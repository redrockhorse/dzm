# -*- coding:utf-8 -*-
#@Time : 2020/3/5 下午2:33
#@Author: kkkkibj@163.com
#@File : perceptualHashAlgorithmtest.py
import cv2
import numpy as np

#求图像的感知hash
#1.缩小尺寸：将图像缩小到8*8的尺寸，总共64个像素。这一步的作用是去除图像的细节，只保留结构/明暗等基本信息，摒弃不同尺寸/比例带来的图像差异；
# 2.简化色彩：将缩小后的图像，转为64级灰度，即所有像素点总共只有64种颜色；
# 3.计算平均值：计算所有64个像素的灰度平均值；
# 4.比较像素的灰度：将每个像素的灰度，与平均值进行比较，大于或等于平均值记为1，小于平均值记为0；
# 5.计算哈希值：将上一步的比较结果，组合在一起，就构成了一个64位的整数，这就是这张图像的指纹。组合的次序并不重要，只要保证所有图像都采用同样次序就行了；
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
        # arr[n] = 1 if arr[n] >= avg else 0
        if arr[n] >= avg:
            binary_ret += 1 << n
    # print(binary_ret)
    return binary_ret
    # print(bin(binary_ret))          # 转化为二进制显示
    # print(bin(binary_ret)[2:])  # 切片，去掉前面的：0b
    # print(bin(binary_ret)[2:])

#求感知hash的差异
def perceptualHashDif(h1,h2):
    t = h1^h2
    return bin(t).count('1')
    # print(bin(h1)[2:])
    # print(bin(h2)[2:])
    # print(t)


if __name__ == '__main__':
    img = cv2.imread("/Users/hongyanma/Desktop/wood.jpeg")
    img1 = cv2.imread("/Users/hongyanma/Desktop/target.jpeg")
    i1 = perceptualHashAlgorithm(img)
    i2 = perceptualHashAlgorithm(img1)
    print(perceptualHashDif(i1, i2))
