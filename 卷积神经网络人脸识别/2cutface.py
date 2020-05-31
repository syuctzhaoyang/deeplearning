# _*_ coding: utf-8 _*_
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
#利用opencv中haarcascade方法，缺点图片中人脸位置
#对象实例化opencv分类器实例
face_cascade = cv.CascadeClassifier(r'C:\opencv4\build\etc\haarcascades\haarcascade_frontalface_default.xml')
from PIL import Image
#方法作用：从网络上下载的人物图片通常含有背景、其他人物且图片尺寸大小不一
#首先使用opencv中利用opencv中haarcascade方法确定人脸位置，将人脸截取出来
# 然后将人脸图片统一设置成64*64大小的图片
#最后将其存储到dest_dir处
#filename要处理的图片的名称, src_dir处理的图片的目录, dest_dir处理后，图片的输出目录
def extractFace(filename, src_dir, dest_dir):
    img = cv.imread(src_dir + filename)
    #确定图像中的人脸，1.3为相似度, 5个框确定一个人脸，faces为图片人脸的列表
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    im = Image.open(src_dir + filename)
    #遍历人脸的列表中的每张脸，
    for (x,y,w,h) in faces:
        #设置人脸范围
        box = (x, y, x+w, y+h)
        #扣取图像中人脸，并将人脸尺寸统一调整为64*64
        crpim = im.crop(box).resize((64,64))
        #将人脸存储到目标文件夹中，并使用原有图片名称
        crpim.save(dest_dir + filename)

#依次对idol1、idol2、idol3中的图片处理
#idol1--范冰冰、idol2--迪丽热巴、idol3--林志玲
for filename in os.listdir('idol3'):
    try:
        extractFace(filename, 'idol3/', 'idol3_test/')
    except:
        print(filename)
