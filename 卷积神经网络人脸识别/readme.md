# 利用卷积神经网络对人物脸部的提取与识别
### 1.运行1downloadimage.py，范冰冰、迪丽热巴、林志玲的图片分别下载到idol1,idol2,idol3文件夹中
### 2.先建好idol1_test,idol2_test,idol3_test文件夹，再运行2cutface.py文件，三个人的脸的图片就分别放入idol1_test,idol2_test,idol3_test文件夹中
### 3.新建trainset文件夹，将idol1_test、idol2_test、idol3_test文件夹拷贝到trainset文件夹中，trainset文件夹下idol1_test、idol2_test、idol3_test保留大部分图片，做为训练集；新建testset文件夹，将idol1_test、idol2_test、idol3_test文件夹拷贝到testset文件夹中，trainset文件夹下idol1_test、idol2_test、idol3_test保留少部分图片，做为验证集。#trainset文件夹下idol1_test、idol2_test、idol3_test与trainset文件夹下idol1_test、idol2_test、idol3_test中的图片尽量不同
### 4. 新建prediction_dataset文件夹，分别从idol1,idol2,idol3文件夹中选取一张照片，并将其命名为i1.jpg,i2.jpg,i3.jpg。运行4predict.py文件。运行结果见示例.jpg
