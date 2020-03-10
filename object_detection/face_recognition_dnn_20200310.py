# -*- coding:utf-8 -*-
# @Time : 2020/3/8 上午10:29
# @Author: kkkkibj@163.com
# @File : face_recognition_dnn.py

import cv2
import numpy as np

# https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py
# proto = 'https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector/deploy.prototxt';
# weights = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel';
# recognModel = 'https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7';

proto = '/Users/hongyanma/gitspace/c++/opencv/samples/dnn/face_detector/deploy.prototxt';
weights = '/Users/hongyanma/gitspace/python/python/data/res10_300x300_ssd_iter_140000.caffemodel';
# proto = '/Users/hongyanma/gitspace/python/python/Ultra-Light-Fast-Generic-Face-Detector-1MB/caffe/model/RFB-320/RFB-320.prototxt';
# weights = '/Users/hongyanma/gitspace/python/python/Ultra-Light-Fast-Generic-Face-Detector-1MB/caffe/model/RFB-320/RFB-320.caffemodel';
#recognModel = '/Users/hongyanma/gitspace/python/python/data/openface.nn4.small2.v1.t7';
recognModel = '/Users/hongyanma/gitspace/python/python/facenet_opencv_dnn/models/graph_final.pb'

face_detector_net = cv2.dnn.readNetFromCaffe(proto, weights)

face_recognize_net = cv2.dnn.readNetFromTensorflow(recognModel)
from sklearn.svm import SVC

from sklearn.externals import joblib

# svm_model = cv2.ml.SVM_load('/Users/hongyanma/gitspace/python/python/facenet_facerecognition/20180402-114759/my_classifier.pkl')
svm_model = joblib.load('/Users/hongyanma/gitspace/python/python/facenet_facerecognition/20180402-114759/my_classifier.pkl')
print(svm_model)

# image = cv2.imread('/Users/hongyanma/gitspace/python/python/data/person/0/1583556761024.jpg')
#
# (h, w) = image.shape[:2]
# blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
#                              (300, 300), (104.0, 177.0, 123.0))
# face_detector_net.setInput(blob)
# detections = face_detector_net.forward()
# for i in range(0, detections.shape[2]):
#     confidence = detections[0, 0, i, 2]
#     if confidence > 0.9:
#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (startX, startY, endX, endY) = box.astype("int")
#
#
#         text = "{:.2f}%".format(confidence * 100)
#         y = startY - 10 if startY - 10 > 10 else startY + 10
#         cv2.rectangle(image, (startX, startY), (endX, endY),
#                       (0, 0, 255), 2)
#         cv2.putText(image, text, (startX, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#
# cv2.imshow('test', image);
# key = cv2.waitKey(0)
# if key == ord("q"):
#     pass

label = ['00000','111111','unknow']
def detect_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_detector_net.setInput(blob)
    detections = face_detector_net.forward()
    # print(detections)
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # print(confidence)
        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            label_index = recognize(image[int(startY):int(endY), int(startX):int(endX)])
            text = label[label_index]
            color = (0, 0,255)
            if label_index == 0:
                color = (255, 255, 0)
            elif label_index == 1:
                color = (0, 255, 0)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          color, 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    return image

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def recognize(face_image):
    im = face_image.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(im, (160, 160), interpolation=cv2.INTER_LINEAR)
    prewhitened = prewhiten(resized)
    # HWC -> CHW
    input_face_img = prewhitened.transpose([2, 0, 1])
    # CHW -> NCHW
    input_face_img = np.expand_dims(input_face_img, axis=0)
    face_recognize_net.setInput(input_face_img)
    cvOut = face_recognize_net.forward()#[[0.04103242 0.95896758]]
    predictions =svm_model[0].predict_proba(cvOut)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    if best_class_probabilities[0] < 0.8:
        return 2
    else:
        return best_class_indices[0]



def trainning():
    print('tring!!')

def imagetest():
    im0=cv2.imread('/Users/hongyanma/gitspace/python/python/data/person/0/1583556657332.jpg')
    # im0 = cv2.imread('/Users/hongyanma/gitspace/python/python/data/person/1/1583557091926.jpg')
    # im0 = cv2.imread('/Users/hongyanma/Desktop/2/1583557418667.jpg')
    detect_face(im0)

def videotest():
    video_url = '/Users/hongyanma/Desktop/liu-wu.mp4'
    # video_url = 0
    cap = cv2.VideoCapture(video_url)
    while cap.isOpened():
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # fa = face_cade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
        # for (x, y, w, h) in fa:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255.0), 2)
        # timg = detect_face(frame)
        timg = detect_face(frame)
        cv2.imshow('face', timg)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == '__main__':
    videotest()
    # imagetest()
