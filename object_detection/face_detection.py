# -*- coding:utf-8 -*-
#@Time : 2020/3/7 下午12:11
#@Author: kkkkibj@163.com
#@File : face_detection.py

import cv2
import os
import numpy as np


face_cade = cv2.CascadeClassifier('/Users/hongyanma/gitspace/c++/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#



def detect_face(img):
    # 将测试图像转换为灰度图像，因为opencv人脸检测器需要灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 加载OpenCV人脸检测分类器Haar
    # face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    # 检测多尺度图像，返回值是一张脸部区域信息的列表（x,y,宽,高）
    faces = face_cade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # 如果未检测到面部，则返回原始图像
    if (len(faces) == 0):
        return None
    return faces

    # 目前假设只有一张脸，xy为左上角坐标，wh为矩形的宽高
    # (x, y, w, h) = faces[0]

    # 返回图像的正面部分
    # return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data(data_folder_path):
    # 获取数据文件夹中的目录（每个主题的一个目录）
    dirs = os.listdir(data_folder_path)

    # 两个列表分别保存所有的脸部和标签
    faces = []
    labels = []

    # 浏览每个目录并访问其中的图像
    for dir_name in dirs:
        if dir_name == '.DS_Store':
            continue
        print(dir_name)
        # dir_name(str类型)即标签
        label = int(dir_name)
        # 建立包含当前主题主题图像的目录路径
        subject_dir_path = data_folder_path + "/" + dir_name
        # 获取给定主题目录内的图像名称
        subject_images_names = os.listdir(subject_dir_path)

        # 浏览每张图片并检测脸部，然后将脸部信息添加到脸部列表faces[]
        for image_name in subject_images_names:
            if image_name == '.DS_Store':
                continue
            print('image_name', image_name)
            # 建立图像路径
            image_path = subject_dir_path + "/" + image_name
            # 读取图像
            image = cv2.imread(image_path)
            # 显示图像0.1s
            cv2.imshow("Training on image...", image)
            cv2.waitKey(1)

            # 检测脸部
            faces_finded = detect_face(image)
            # 我们忽略未检测到的脸部
            if faces_finded is not None:
                # 将脸添加到脸部列表并添加相应的标签
                for f in faces_finded:
                    (x, y, w, h) = f
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray[y:y + w, x:x + h],(200,200),0,0,cv2.INTER_AREA)
                    faces.append(gray)
                    labels.append(label)
                    cv2.imshow("Training on image...", gray[y:y + w, x:x + h])
                    cv2.waitKey(1000)

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    # 最终返回值为人脸和标签列表
    return faces, labels


# 根据给定的（x，y）坐标和宽度高度在图像上绘制矩形
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 0), 2)


# 根据给定的（x，y）坐标标识出人名
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)


# 建立标签与人名的映射列表（标签只能为整数）
subjects = ["liudehua", "wuyanzu"]
def trainning():
    # 调用prepare_training_data（）函数
    faces, labels = prepare_training_data("/Users/hongyanma/gitspace/python/python/data/person/")
    # 创建LBPH识别器并开始训练，当然也可以选择Eigen或者Fisher识别器
    # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    cv2.face.FisherFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save('liu-wu.xml')

# 滑动轴一定要个函数
def nothing(x):
    pass

# 创建一个滑动轴来控制置信度
# cv2.createTrackbar('confindence', 'Frame', 0, 200, nothing)
# 此函数识别传递的图像中的人物并在检测到的脸部周围绘制一个矩形及其名称
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read('liu-wu.xml')
def predict(test_img):
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.read('liu-wu.xml')
    # 生成图像的副本，这样就能保留原始图像
    img = test_img.copy()
    # 检测人脸
    faces = detect_face(img)
    # print(faces)
    if faces is not None:
        for f in faces:
            (x, y, w, h) = faces[0]
            rect = (x, y, w, h)
            face = img[y:y + w, x:x + h]
            # 预测人脸
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200), 0, 0, cv2.INTER_AREA)
            label = face_recognizer.predict(gray)
            print(label)
            confidence = ":{0}".format(label[1])
            # con_threhold = cv2.getTrackbarPos('confindence', 'Frame')
            con_threhold = 100000
            print(con_threhold)
            if label[1] < con_threhold:
                # 获取由人脸识别器返回的相应标签的名称
                label_text = subjects[label[0]]
                # 在检测到的脸部周围画一个矩形
                draw_rectangle(img, rect)
                # 标出预测的名字
                draw_text(img, label_text, rect[0], rect[1] - 5)
            # 返回预测的图像
    return img


def videotest():
    video_url = '/Users/hongyanma/Desktop/liu-wu.mp4'
    cap =cv2.VideoCapture(video_url)
    while cap.isOpened():
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # fa = face_cade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
        # for (x, y, w, h) in fa:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255.0), 2)
        timg = predict(frame)
        cv2.imshow('face', timg)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

if __name__ == '__main__':
    # trainning()
    videotest()


