# _*_ coding: utf-8 _*_
#%% md
### 预测单张图片
#%%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
face_cascade = cv.CascadeClassifier(r'C:\opencv4\build\etc\haarcascades\haarcascade_frontalface_default.xml')
from PIL import Image
from tensorflow.keras.models import load_model
#加载训练好的模型及参数
classifier = load_model('model.h5')
#测试图片地址
filename = r'prediction_dataset/i3.jpg'
#提取人脸后的照片的地址
file_d_name = r'prediction_dataset/i3_d.jpg'

img = cv.imread(filename)
#提取人脸
faces = face_cascade.detectMultiScale(img, 1.3, 5)
im = Image.open(filename)

x,y,w,h = faces[0]

box = (x, y, x+w, y+h)
crpim = im.crop(box).resize((64,64))
crpim.save(file_d_name)
#%%
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(file_d_name, target_size= (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
flag = classifier.predict_classes(test_image)[0];

#显示原有图片
plt.imshow(im)
plt.show()
#输入判断结果
if(flag == 0):
    print('范冰冰')
elif (flag == 1):
    print('迪丽热巴')
elif(flag == 2):
    print('林志玲')
else:
    print('我不认识这个人')

