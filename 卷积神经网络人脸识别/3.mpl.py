# _*_ coding: utf-8 _*_

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Initialising the CNN
classifier = Sequential()
# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64,3), activation = 'relu'))
# Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Convolution
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Flattening
classifier.add(Flatten())

# Fully Connected
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

classifier.compile(optimizer = 'adam',
                        loss ='categorical_crossentropy',
                     metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,     #x坐标保持不变，而对应的y坐标按比例发生平移
                                   zoom_range = 0.2,      #可以让图片在长或宽的方向进行放大
                                   horizontal_flip = True #水平翻转操作
                                  )
#%%
test_datagen = ImageDataGenerator(rescale = 1./255)
#%% md
### 建立训练与测试数据集
#%%
training_set = train_datagen.flow_from_directory(
    'trainset/', target_size = (64, 64),
     batch_size = 30,
     class_mode = 'categorical')
#%%
# training_set.class_indices
#%%

#%%
test_set = test_datagen.flow_from_directory(
    'testset/', target_size = (64, 64),
    batch_size = 30,
    class_mode = 'categorical')
#%% md
### 训练神经网路
#%%
history = classifier.fit_generator(training_set,
                         epochs = 100,
                         verbose = 1,
                         validation_data = test_set)
#%%
classifier.save("model.h5")
print("Saved model to disk")
