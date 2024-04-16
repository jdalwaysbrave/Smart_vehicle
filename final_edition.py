import time
from socket import *
from tkinter.messagebox import NO

import cv2
from sklearn.externals import joblib
import numpy as np
from skimage import feature as ft
from threading import Thread

img_label = {"Speed_limit_15": 0, "Speed_limit_30": 1, "Speed_limit_60": 2, "Speed_limit_80": 3, "No straight": 4,
             "Turn left": 5, "Turn right": 6, "background": 7}
cls_names = ["Speed_limit_15", "Speed_limit_30", "Speed_limit_60", "Speed_limit_80", "No straight", "Turn left",
             "Turn right", "background"]
'''
通过颜色阈值分割选出蓝色和红色对应的区域得到二值化图像。
'''


def preprocess_img(imgBGR):
    # 将图像由RGB模型转化成HSV模型
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_RGB2HSV)
    Bmin = np.array([110, 43, 46])
    Bmax = np.array([124, 255, 255])
    # 使用inrange(HSV,lower,upper)设置阈值去除背景颜色
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)
    Rmin2 = np.array([165, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_bin = np.maximum(img_Bbin, img_Rbin)
    return img_bin


'''
提取轮廓,返回轮廓矩形框
'''


def contour_detect(img_bin, min_area=0, max_area=-1, wh_ratio=2.0):
    rects = []
    # 检测轮廓，其中cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_NONE 存储所有的边界点
    # findContours返回三个值:第一个值返回img，第二个值返回轮廓信息，第三个返回相应轮廓的关系
    contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects
    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects


'''
返回带有矩形框的img
'''


def draw_rects_on_img(img, rects):
    img_copy = img.copy()
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy


def hog_extra_and_svm_class(proposal, clf, resize=(64, 64)):
    # 对图片进行分类
    img = cv2.cvtColor(proposal, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, resize)
    bins = 9
    cell_size = (8, 8)
    cpb = (2, 2)
    norm = "L2"
    features = ft.hog(img, orientations=bins, pixels_per_cell=cell_size,
                      cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
    features = np.reshape(features, (1, -1))
    cls_prop = clf.predict_proba(features)
    cls_prop = cls_prop[0]
    return cls_prop


class handle(Thread):
    def __init__(self, threadID, name, counter, *args) -> None:
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.args = args
    
    def run(self):
        num = self.args[0]
        if num == 0:
            s.send(b'\xff\x02\x01\x40\xff')
            s.send(b'\xff\x02\x02\x40\xff')
            print('Speed:15\n')
            time.sleep(1)
        if num == 1:
            s.send(b'\xff\x02\x01\x45\xff')
            s.send(b'\xff\x02\x02\x47\xff')
            print('Speed:30\n')
            time.sleep(1)
        if num == 2:
            s.send(b'\xff\x02\x01\x30\xff')
            s.send(b'\xff\x02\x02\x30\xff')
            print('Speed:60\n')
            time.sleep(1)
        if num == 3:
            s.send(b'\xff\x02\x01\x60\xff')
            s.send(b'\xff\x02\x02\x60\xff')
            print('Speed:80\n')
            time.sleep(1)
        # if num == 0:
        #     s.send(b'\xff\x02\x01\x20\xff')
        #     s.send(b'\xff\x02\x02\x20\xff')
        #     print('Speed:15\n')
        #     time.sleep(1)
        # if num == 1:
        #     s.send(b'\xff\x02\x01\x35\xff')
        #     s.send(b'\xff\x02\x02\x35\xff')
        #     print('Speed:30\n')
        #     time.sleep(1)
        # if num == 2:
        #     s.send(b'\xff\x02\x01\x40\xff')
        #     s.send(b'\xff\x02\x02\x40\xff')
        #     print('Speed:60\n')
        #     time.sleep(1)
        # if num == 3:
        #     s.send(b'\xff\x02\x01\x70\xff')
        #     s.send(b'\xff\x02\x02\x72\xff')
        #     print('Speed:80\n')
        #     time.sleep(1)
        if num == 4:
            s.send(b'\xff\x02\x01\x00\xff')
            s.send(b'\xff\x02\x02\x00\xff')
            print('停止\n')
            time.sleep(1)
        if num == 5:
            # s.send(b'\xff\x00\x06\x00\xff')
            # s.send(b'\xff\x02\x02\x45\xff')
            # s.send(b'\xff\x02\x01\x00\xff')
            # print('右前转\n')
            # time.sleep(3)
            # s.send(b'\xff\x02\x01\x40\xff')
            # s.send(b'\xff\x02\x02\x48\xff')
            s.send(b'\xff\x02\x01\x45\xff')
            s.send(b'\xff\x02\x02\x00\xff')
            print('右前转\n')
            time.sleep(3)
            s.send(b'\xff\x02\x01\x40\xff')
            s.send(b'\xff\x02\x02\x48\xff')
        if num == 6:
            # s.send(b'\xff\x02\x01\x45\xff')
            # s.send(b'\xff\x02\x02\x00\xff')
            # print('左前转\n')
            # time.sleep(3)
            # s.send(b'\xff\x02\x01\x40\xff')
            # s.send(b'\xff\x02\x02\x48\xff')
            s.send(b'\xff\x02\x02\x45\xff')
            s.send(b'\xff\x02\x01\x00\xff')
            print('左前转\n')
            time.sleep(3)
            s.send(b'\xff\x02\x01\x40\xff')
            s.send(b'\xff\x02\x02\x48\xff')
class ThreadedCamera(object):
    def __init__(self,source=0) -> None:
        self.capture = cv2.VideoCapture(source)
        self.thread = Thread(target=self.update,args=())
        self.thread.daemon=True
        self.thread.start()

        self.status = False
        self.frame = None
    
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status,self.frame)=self.capture.read()

    def grab_frame(self):
        if self.status:
            return self.frame
        return None



if __name__ == "__main__":
    s = socket(AF_INET, SOCK_STREAM)
    s.connect(('192.168.1.1', 2001))
    print("Socket connect successfully")
    s.send(b'\xff\x00\x01\x00\xff')
    s.send(b'\xff\x02\x01\x40\xff')
    s.send(b'\xff\x02\x02\x48\xff')
    print('Speed:15\n')

    streamlink="http://192.168.1.1:8080/?action=stream"
    streamer = ThreadedCamera(streamlink)

    cv2.namedWindow('camera')
    # cv2.resizeWindow("camera", 640, 480)
    cols = int(streamer.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(streamer.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fp = open('./svm_model.pkl', 'rb+')
    clf = joblib.load(fp)
    i = 0
    cls_me=-1
    while 1:
        i += 1
        img = streamer.grab_frame()
        img_bin = preprocess_img(img)
        cv2.imshow("bin",img_bin)
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area=min_area)
        if rects:
            Max_X = 0
            Max_Y = 0
            Max_W = 0
            Max_H = 0
            for r in rects:
                if r[2] * r[3] >= Max_W * Max_H:
                    Max_X, Max_Y, Max_W, Max_H = r
            proposal = img[Max_Y:(Max_Y + Max_H), Max_X:(Max_X + Max_W)]
            # 用Numpy数组对图像像素进行访问时，应该先写图像高度所对应的坐标(y,row)，再写图像宽度对应的坐标(x,col)。
            cv2.rectangle(img, (Max_X, Max_Y), (Max_X + Max_W, Max_Y + Max_H), (0, 255, 0), 2)
            cv2.imshow("proposal", proposal)
            cls_prop = hog_extra_and_svm_class(proposal, clf)
            cls_prop = np.round(cls_prop, 2)
            cls_num = np.argmax(cls_prop)  # 找到最大相似度的索引
            if cls_names[cls_num] != "background" and cls_num != cls_me :
                cls_me = cls_num
                print(cls_names[cls_num])
                handleThread=handle(i,'handle-1'.format(i),1,cls_num)
                handleThread.start()
        cv2.imshow('camera', img)
        cv2.waitKey(40)
