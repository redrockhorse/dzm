# -*- coding:utf-8 -*-
# @Time : 2020/4/10 下午5:29
# @Author: kkkkibj@163.com
# @File : traffic_lighticon_detector.py
# Load names of classes
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import UnexpectedAlertPresentException
import math

chromedriver = '/Users/hongyanma/Downloads/chromedriver'
# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

xu = 6370996.81
Sp = [1.289059486E7, 8362377.87, 5591021, 3481989.83, 1678043.12, 0]
Hj = [75, 60, 45, 30, 15, 0]

Au = [[1.410526172116255e-8, 0.00000898305509648872, -1.9939833816331, 200.9824383106796, -187.2403703815547,
       91.6087516669843, -23.38765649603339, 2.57121317296198, -0.03801003308653, 17337981.2],
      [-7.435856389565537e-9, 0.000008983055097726239, -0.78625201886289, 96.32687599759846, -1.85204757529826,
       -59.36935905485877, 47.40033549296737, -16.50741931063887, 2.28786674699375, 10260144.86],
      [-3.030883460898826e-8, 0.00000898305509983578, 0.30071316287616, 59.74293618442277, 7.357984074871,
       -25.38371002664745, 13.45380521110908, -3.29883767235584, 0.32710905363475, 6856817.37],
      [-1.981981304930552e-8, 0.000008983055099779535, 0.03278182852591, 40.31678527705744, 0.65659298677277,
       -4.44255534477492, 0.85341911805263, 0.12923347998204, -0.04625736007561, 4482777.06],
      [3.09191371068437e-9, 0.000008983055096812155, 0.00006995724062, 23.10934304144901, -0.00023663490511,
       -0.6321817810242, -0.00663494467273, 0.03430082397953, -0.00466043876332, 2555164.4],
      [2.890871144776878e-9, 0.000008983055095805407, -3.068298e-8, 7.47137025468032, -0.00000353937994,
       -0.02145144861037, -0.00001234426596, 0.00010322952773, -0.00000323890364, 826088.5]]
Qp = [[-0.0015702102444, 111320.7020616939, 1704480524535203, -10338987376042340, 26112667856603880, -35149669176653700,
       26595700718403920, -10725012454188240, 1800819912950474, 82.5],
      [0.0008277824516172526, 111320.7020463578, 647795574.6671607, -4082003173.641316, 10774905663.51142,
       -15171875531.51559, 12053065338.62167, -5124939663.577472, 913311935.9512032, 67.5],
      [0.00337398766765, 111320.7020202162, 4481351.045890365, -23393751.19931662, 79682215.47186455,
       -115964993.2797253, 97236711.15602145, -43661946.33752821, 8477230.501135234, 52.5],
      [0.00220636496208, 111320.7020209128, 51751.86112841131, 3796837.749470245, 992013.7397791013, -1221952.21711287,
       1340652.697009075, -620943.6990984312, 144416.9293806241, 37.5],
      [-0.0003441963504368392, 111320.7020576856, 278.2353980772752, 2485758.690035394, 6070.750963243378,
       54821.18345352118, 9540.606633304236, -2710.55326746645, 1405.483844121726, 22.5],
      [-0.0003218135878613132, 111320.7020701615, 0.00369383431289, 823725.6402795718, 0.46104986909093,
       2351.343141331292, 1.58060784298199, 8.77738589078284, 0.37238884252424, 7.45]]


def Yr(x, y, b):
    if b is not None:
        c = b[0] + b[1] * abs(x)
        d = abs(y) / b[9]
        d = b[2] + b[3] * d + b[4] * d * d + b[5] * d * d * d + b[6] * d * d * d * d + b[7] * d * d * d * d * d + b[
            8] * d * d * d * d * d * d
        lon = c * (-1 if 0 > x else 1)
        lat = d * (-1 if 0 > y else 1)
        return round(lon, 6), round(lat, 6)
    else:
        return None


def Mercator2BD09(lng, lat):
    x, y = abs(lng), abs(lat)
    for d in range(len(Sp)):
        if y >= Sp[d]:
            c = Au[d]
            break
    return Yr(x, y, c)


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, frame):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, mapBounds):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    icon_centers = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                icon_centers.append([left + int(left + width / 2), top + int(top + height / 2)])
                if len(mapBounds) == 4:
                    print(mapBounds)
                    x = mapBounds[0] + (mapBounds[2] - mapBounds[0]) * detection[0]
                    y = mapBounds[1] + (mapBounds[3] - mapBounds[1]) * detection[1]
                    print(x, y)
                    # lon, lat = UTMtoGeog(x, y)
                    lon, lat = Mercator2BD09(x, y)
                    with open("/Users/hongyanma/Desktop/mapcompare/trafficlight.txt", 'a') as f:
                        f.write(str(lon) + "," + str(lat) + '\n')
                    print(lon, lat)
    print(frameWidth, frameHeight, icon_centers)
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    # indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # for i in indices:
    #     i = i[0]
    #     box = boxes[i]
    #     left = box[0]
    #     top = box[1]
    #     width = box[2]
    #     height = box[3]
    #     drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)


classesFile = "/Users/hongyanma/gitspace/python/python/darknet/x64/Release/data/obj.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "/Users/hongyanma/gitspace/python/python/darknet/cfg/yolov3-voc.cfg";
modelWeights = "/Users/hongyanma/gitspace/python/python/darknet/backup/yolov3-voc_900.weights";

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findAndLocation(mapBounds):
    frame = cv2.imread('/Users/hongyanma/Desktop/mapcompare/baidumap.png')
    cv2.imwrite('/Users/hongyanma/Desktop/mapcompare/baidumap.jpg', frame)
    frame = cv2.imread('/Users/hongyanma/Desktop/mapcompare/baidumap.jpg')
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers

    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, mapBounds)

    # Put efficiency information. The function getPerfProfile returns the
    # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    # t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    # cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
    # cv2.imshow('testwindow', frame)
    # cv2.waitKey(0)


def savaBaiduMapAsPng(lon, lat, zoom, waittime, suffix):
    # lon_int_str = str(int(round(float(lon), 6) * 100000))
    # lat_int_str = str(int(round(float(lat), 6) * 100000))
    lon_int_str = str(lon)
    lat_int_str = str(lat)
    zoom = str(int(zoom) + 1)  # 百度的zoom要在搜狗的基础上+1
    driver = webdriver.Chrome(chromedriver)
    # driver.maximize_window()
    driver.set_window_size(1000, 623)
    driver.implicitly_wait(100)
    # driver.get(
    #     'https://map.baidu.com/@' + lon_int_str + ',' + lat_int_str + ',' + zoom + 'z/maplayer%3Dtrafficrealtime')
    driver.get('https://map.baidu.com/@' + lon_int_str + ',' + lat_int_str + ',' + zoom + 'z')
    time.sleep(int(waittime))

    # dragger = driver.find_elements_by_id("maps")[0]
    # action = ActionChains(driver)
    # action.click_and_hold(dragger).perform()
    # try:
    #     action.drag_and_drop_by_offset(dragger, 500, 0).perform()
    # except UnexpectedAlertPresentException:
    #     print("faild")
    # time.sleep(int(60))
    mapBounds = []
    try:
        EC.presence_of_element_located((By.XPATH, '//canvas'))
        with open('/Users/hongyanma/gitspace/python/python/palmgo3.5/cleanbaidumap.js', 'r', encoding='utf8') as fr:
            js_str = fr.read()
            driver.execute_script(js_str)
            tt = driver.execute_script('return map.getBounds()')
            mapBounds = [tt['minX'], tt['minY'], tt['maxX'], tt['maxY']]  # Mercator 坐标矩形框 250 * 125
            # "maxX": 12951222,
            # "maxY": 4829317.5,
            # "minX": 12950972,
            # "minY": 4829192.5,
            # print(tt)
        driver.save_screenshot('/Users/hongyanma/Desktop/mapcompare/baidumap.png')
    finally:
        driver.quit()
    return mapBounds


if __name__ == '__main__':
    # 惠州市墨卡托范围
    minX = 12736399.587945374
    minY = 2627940.154266209
    maxX = 12739182.439338444
    maxY = 2629385.9482605834
    center = [minX + 125, minY + 125 / 2]
    stepX = 250
    stepY = 125
    # mapBounds = savaBaiduMapAsPng(129.51097657118486, 48.29255044118175, 19, 10, '03')
    while center[1] < maxY:
        mapBounds = savaBaiduMapAsPng(center[0], center[1], 19, 5, '03')
        findAndLocation(mapBounds)
        while center[0] < maxX:
            center[0] = center[0] + 250
            mapBounds = savaBaiduMapAsPng(center[0], center[1], 19, 5, '03')
            findAndLocation(mapBounds)
        center[1] = center[1] + 125
        center[0] = minX + 125
