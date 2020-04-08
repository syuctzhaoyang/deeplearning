# _*_ coding: utf-8 _*_
'''
使用3种keras方法定义神经网络
第一种  继承方式 class MLP(tf.keras.Model): 继承tf.keras.Model基础类，
         重写 __init__(self):方法，在该方法中设置网络中的构成组件
         重写 call(self, inputs):方法，在该方法中设置网络中各个组件的顺序和关系
         inputs 为待传入的训练数据

第二种  Keras Sequential模式建立模型
        Keras 的 Sequential API 通过向 tf.keras.models.Sequential()
        提供一个层的列表快速地建立一个 tf.keras.Model 模型
        缺点：这种层叠结构并不能表示任意的神经网络结构。

第三种  Keras Functional API 模式建立模型
        这种方式帮助我们建立更为复杂的模型，例如多输入 / 输出或存在参数共享的模型。
        其使用方法是将层作为可调用的对象并返回张量（这点与之前章节的使用方法一致），
        并将输入向量和输出向量提供给 tf.keras.Model 的 inputs 和 outputs 参数，

推荐使用第三种方式建立神经网络

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
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):         # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output
model = MLP()

# 第二种  Keras Sequential模式建立模型
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(100, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10),
#     tf.keras.layers.Softmax()
# ])
# model = MLP()

# 第三种  Keras Functional API 模式建立模型
# inputs = tf.keras.Input(shape=(28, 28, 1))
# x = tf.keras.layers.Flatten()(inputs)
# x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
# x = tf.keras.layers.Dense(units=10)(x)
# outputs = tf.keras.layers.Softmax()(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
# model = MLP()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

num_epochs = 5
batch_size = 50
learning_rate = 0.001

data_loader = MNISTLoader()

model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)

print(model.evaluate(data_loader.test_data, data_loader.test_label))
