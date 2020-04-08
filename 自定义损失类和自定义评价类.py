# _*_ coding: utf-8 _*_
'''
自定义损失类和自定义评价类
class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = tf.one_hot(tf.cast(tf.squeeze(y_true,axis=-1),tf.int32),depth=10)
        #tf.squeeze   降维
        #tf.cast      类型转换    tf.float32-->tf.int32，原因同class SparseCategoricalAccuracy(tf.keras.metrics.Metric)
        #tf.one_hot   转换成独热编码
        return tf.reduce_mean(tf.square(y_pred - y_true))

class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
         # tf.cast(y_true,tf.int32),axis=-1
         # y_true形同[             \   tf.argmax(y_pred, axis=-1, output_type=tf.int32)形同[
         #            [1.],        \                                                        1，
         #            [2.],        \                                                        3,
         #            [5.],        \                                                        5,
         #            ...          \                                                       ...
         #       ]                 \                                                        ]
         #
         #数据类型不同:y_true---tf.float32, y_pred----tf.int32
         #维度不同，双方要想比较，y_true需要降维
        values = tf.cast(tf.equal(tf.squeeze(tf.cast(y_true,tf.int32),axis=-1), tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total
'''
import tensorflow as tf
import numpy as np

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]
# 1第一种  继承方式
# class MLP(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
#         self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
#         self.dense2 = tf.keras.layers.Dense(units=10)
#
#     def call(self, inputs):         # [batch_size, 28, 28, 1]
#         x = self.flatten(inputs)    # [batch_size, 784]
#         x = self.dense1(x)          # [batch_size, 100]
#         x = self.dense2(x)          # [batch_size, 10]
#         output = tf.nn.softmax(x)
#         return output
# model = MLP()

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_weight(name='w',
            shape=[input_shape[-1], self.units], initializer=tf.initializers.GlorotNormal())
        self.b = self.add_weight(name='b',
            shape=[self.units], initializer=tf.initializers.GlorotNormal())

    def call(self, inputs):
        y_pred = tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
        return y_pred

# 第二种  Keras Sequential模式建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(100, activation=tf.nn.relu),
    LinearLayer(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])


# 第三种  Keras Functional API 模式建立模型
# inputs = tf.keras.Input(shape=(28, 28, 1))
# x = tf.keras.layers.Flatten()(inputs)
# x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
# x = tf.keras.layers.Dense(units=10)(x)
# outputs = tf.keras.layers.Softmax()(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = tf.one_hot(tf.cast(tf.squeeze(y_true,axis=-1),tf.int32),depth=10)
        return tf.reduce_mean(tf.square(y_pred - y_true))

class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(tf.squeeze(tf.cast(y_true,tf.int32),axis=-1), tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=MeanSquaredError(),#tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[SparseCategoricalAccuracy()]#[tf.keras.metrics.sparse_categorical_accuracy]
)

num_epochs = 50
batch_size = 50
learning_rate = 0.001

data_loader = MNISTLoader()

model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)

print(model.evaluate(data_loader.test_data, data_loader.test_label))