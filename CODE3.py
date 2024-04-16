import os
import numpy as np 

import cv2
from skimage import feature as ft 

from sklearn.externals import joblib
import json
import socket
import sys
import binascii
import time


cls_names = ["Straight", "Turn left", "Turn right", "stop", "background"]
img_label = {"Straight": 0, "Turn left": 1, "Turn right": 2, "stop": 3, "background": 4}
'''
通过颜色阈值分割选出蓝色和红色对应的区域得到二值化图像。
'''
def preprocess_img(imgBGR):
    	##将图像由RGB模型转化成HSV模型
	imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
	Bmin = np.array([110, 43, 46])
	Bmax = np.array([124, 255, 255])
    	##使用inrange(HSV,lower,upper)设置阈值去除背景颜色
	img_Bbin = cv2.inRange(imgHSV,Bmin, Bmax)
	Rmin2 = np.array([165, 43, 46])
	Rmax2 = np.array([180, 255, 255])
	img_Rbin = cv2.inRange(imgHSV,Rmin2, Rmax2)
	img_bin = np.maximum(img_Bbin, img_Rbin)
	return img_bin

'''
提取轮廓,返回轮廓矩形框
'''
def contour_detect(img_bin, min_area=0, max_area=-1, wh_ratio=2.0):
	rects = []
	##检测轮廓，其中cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_NONE 存储所有的边界点
	##findContours返回三个值:第一个值返回img，第二个值返回轮廓信息，第三个返回相应轮廓的关系
	contours, hierarchy= cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if len(contours) == 0:
		return rects
	max_area = img_bin.shape[0]*img_bin.shape[1] if max_area<0 else max_area
	for contour in contours:
		area = cv2.contourArea(contour)
		if area >= min_area and area <= max_area:
			x, y, w, h = cv2.boundingRect(contour)
			if 1.0*w/h < wh_ratio and 1.0*h/w < wh_ratio:
				rects.append([x,y,w,h])
	return rects
'''
返回带有矩形框的img
'''
def draw_rects_on_img(img, rects):
	img_copy = img.copy()
	for rect in rects:
		x, y, w, h = rect
		cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0,255,0), 2)
	return img_copy

def hog_extra_and_svm_class(proposal, clf, resize = (64, 64)):
    ##对图片进行分类
	img = cv2.cvtColor(proposal, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, resize)
	bins = 9
	cell_size = (8, 8)
	cpb = (2, 2)
	norm = "L2"
	features = ft.hog(img, orientations=bins, pixels_per_cell=cell_size, 
		cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
	features = np.reshape(features, (1,-1))
	cls_prop = clf.predict_proba(features)
	cls_prop = cls_prop[0]
	return cls_prop
'''
视频识别
'''
if __name__ == "__main__":

	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect(("192.168.1.1", 2001))
	print(1)

	streamlink="http://192.168.1.1:8080/?action=stream"

	cap = cv2.VideoCapture(streamlink)
	
	cv2.namedWindow('camera')
	cv2.resizeWindow("camera",640,480)
	cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	clf = joblib.load("E:\smartcar\svm_model.pkl")
	i=0
	cls_me=-1
	while (1):
		i+=1
		ret, img = cap.read()
		if ret ==  False:
			continue
		img_bin = preprocess_img(img)
		min_area = img_bin.shape[0]*img.shape[1]/(25*25)
		rects = contour_detect(img_bin, min_area=min_area)
		if rects:
			Max_X=0
			Max_Y=0
			Max_W=0
			Max_H=0
			for r in rects:
				if r[2]*r[3]>=Max_W*Max_H:
					Max_X,Max_Y,Max_W,Max_H=r
			proposal = img[Max_Y:(Max_Y+Max_H),Max_X:(Max_X+Max_W)]##用Numpy数组对图像像素进行访问时，应该先写图像高度所对应的坐标(y,row)，再写图像宽度对应的坐标(x,col)。
			cv2.rectangle(img,(Max_X,Max_Y), (Max_X+Max_W,Max_Y+Max_H), (0,255,0), 2)
			cv2.imshow("proposal", proposal)
			cls_prop = hog_extra_and_svm_class(proposal, clf)
			cls_prop = np.round(cls_prop, 2)
			cls_num = np.argmax(cls_prop)##找到最大相似度的索引
			
			if cls_names[cls_num] != "background" and cls_num != cls_me :
				cls_me = cls_num
                # print(cls_names[cls_num])

	#		if cls_names[cls_num] is not "background": 
	#			print(cls_names[cls_num])


			


				s.send(b'\xff\x02\x01\x40\xff') #右边履带
				# time.sleep(1)
				print(2)
				s.send(b'\xff\x02\x02\x48\xff') #左边履带
				# time.sleep(1)
				print(3)
				if cls_names[cls_num] is  "Straight":
					print("直行")
					s.send(b'\xff\x00\x01\x00\xff')
					# time.sleep(1)
				elif cls_names[cls_num] is  "Turn left":
					# print("左转")
					# s.send(b'\xff\x00\x03\x00\xff')
					# print('左前转')
					# time.sleep(1)
					# s.send(b'\xff\x00\x01\x00\xff')

					print("左转")
					s.send(b'\xff\x00\x04\x00\xff')
					time.sleep(1)
					s.send(b'\xff\x00\x01\x00\xff')

				elif cls_names[cls_num] is  "Turn right":
					# print("右转")
					# s.send(b'\xff\x00\x04\x00\xff')
					# time.sleep(2)
					# s.send(b'\xff\x00\x01\x00\xff')

					print("右转")
					s.send(b'\xff\x00\x03\x00\xff')
					print('右前转')
					time.sleep(1)
					s.send(b'\xff\x00\x01\x00\xff')
				
				elif cls_names[cls_num] is  "stop":
					print("停止")
					s.send(b'\xff\x02\x01\x00\xff')
					s.send(b'\xff\x02\x02\x00\xff') 
					time.sleep(1)
				
				else:
					print("N/A")

	

		cv2.imshow('camera',img)
		cv2.waitKey(40)
	cv2.destroyAllWindows()
	cap.release()