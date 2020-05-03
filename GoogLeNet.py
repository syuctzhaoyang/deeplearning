##尽管35行处batchsize已经设的很小了，仍然无法解决内存溢出问题

import numpy as np
import tensorflow




def normalize(X_train, X_test):
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    return X_train, X_test


def preprocess(x, y):
    x = tensorflow.cast(x, tensorflow.float32)
    y = tensorflow.cast(y, tensorflow.int32)
    y = tensorflow.squeeze(y, axis=1)
    y = tensorflow.one_hot(y, depth=10)

    return x, y

(x_train,y_train),(x_test,y_test) = tensorflow.keras.datasets.cifar10.load_data()
x_train, x_test = normalize(x_train, x_test)


train_db = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.shuffle(500).batch(1).map(preprocess)

test_db = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(100).batch(1).map(preprocess)

class ConvBNRelu(tensorflow.keras.Model):
    '''
    扩展基本卷积层Conv+BN+Relu
    '''

    def __init__(self, filters, kernelsize=3, strides=1, padding='same'):
        super().__init__()
        self.model = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.Conv2D(filters=filters,
                                kernel_size=kernelsize,
                                strides=strides,
                                padding=padding),
            tensorflow.keras.layers.BatchNormalization(),
            tensorflow.keras.layers.ReLU()
        ])
    ## 此处调用call方法，不是__call__()方法
    def call(self, x, training=None):
        x = self.model(x, training=training)
        return x


class Inception_v2(tensorflow.keras.Model):
    '''
    构造Inception_v2模块
    '''

    def __init__(self, filters, strides=1):
        super().__init__()
        self.conv1_1 = ConvBNRelu(filters=filters, kernelsize=1, strides=1)
        self.conv1_2 = ConvBNRelu(filters=filters, kernelsize=3, strides=1)
        self.conv1_3 = ConvBNRelu(filters=filters, kernelsize=3, strides=strides)

        self.conv2_1 = ConvBNRelu(filters=filters, kernelsize=1, strides=1)
        self.conv2_2 = ConvBNRelu(filters=filters, kernelsize=3, strides=strides)

        self.pool = tensorflow.keras.layers.MaxPooling2D(pool_size=3, strides=strides, padding='same')
    ## 此处调用call方法，不是__call__()方法
    def call(self, x, training=None):
        x1_1 = self.conv1_1(x, training=training)
        x1_2 = self.conv1_2(x1_1, training=training)
        x1_3 = self.conv1_3(x1_2, training=training)

        x2_1 = self.conv2_1(x, training=training)
        x2_2 = self.conv2_2(x2_1, training=training)

        x3 = self.pool(x)

        x = tensorflow.concat([x1_3, x2_2, x3], axis=3)

        return x


class GoogLeNet(tensorflow.keras.Model):
    def __init__(self, num_blocks, num_classes, filters=16):
        '''
        构造GoogLeNet模型
        :param num_blocks: 包含具有相同filters的n个Inception_v2模块的block数量
        :param num_classes: 分类数量
        :type filters: 卷积核的个数
        '''

        super().__init__()
        self.filters = filters
        # 第一个卷积层
        self.conv1 = ConvBNRelu(filters=filters)
        # 购置动态数量的Inception_v2组合模块
        self.blocks = tensorflow.keras.models.Sequential()

        for block_id in range(num_blocks):
            for Inception_id in range(2):
                if Inception_id == 0:
                    block = Inception_v2(self.filters, strides=2)
                else:
                    block = Inception_v2(self.filters, strides=1)
                self.blocks.add(block)
            self.filters *= 2
        self.avg_pool = tensorflow.keras.layers.GlobalAveragePooling2D()
        self.fc = tensorflow.keras.layers.Dense(num_classes, activation='softmax')
    ## 此处调用call方法，不是__call__()方法
    def call(self, x, training=None):
        out = self.conv1(x, training=training)
        out = self.blocks(out, training=training)
        out = self.avg_pool(out)
        out = self.fc(out)
        return out


model = GoogLeNet(2,10)
model.build(input_shape=(None, 32, 32, 3))
print(model.summary())

model.compile(optimizer=tensorflow.keras.optimizers.Adam(0.001),
              loss=tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
history = model.fit(train_db,epochs=20)
