# _*_ coding: utf-8 _*_
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

face_cascade = cv.CascadeClassifier(r'C:\opencv4\build\etc\haarcascades\haarcascade_frontalface_default.xml')
from PIL import Image
def extractFace(filename, src_dir, dest_dir):
    img = cv.imread(src_dir + filename)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    im = Image.open(src_dir + filename)
    for (x,y,w,h) in faces:
        box = (x, y, x+w, y+h)
        crpim = im.crop(box).resize((64,64))
        crpim.save(dest_dir + filename)


for filename in os.listdir('idol3'):
    try:
        extractFace(filename, 'idol3/', 'idol3_test/')
    except:
        print(filename)