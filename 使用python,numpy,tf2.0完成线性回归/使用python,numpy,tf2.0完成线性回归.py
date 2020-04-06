# 1.使用纯python语言完成最小二乘法线性回归

# a = 0
# b = 0
#
# def f(x):
#     y_pred = a * x + b
#     return y_pred
#
# def loss(x, y):
#     l = (a * x + b - y) ** 2
#     return l
#
# def gradient_loss(x, y):
#     g_a = 2 * (a * x + b - y) * x
#     g_b = 2 * (a * x + b - y)
#     return g_a, g_b
#
# X_raw = [2013, 2014, 2015, 2016, 2017]
# Y_raw = [12000, 14000, 15000, 16500, 17500]
# x_pred_raw = 2018
# X = [(x - min(X_raw)) / (max(X_raw) - min(X_raw)) for x in X_raw]
# Y = [(y - min(Y_raw)) / (max(Y_raw) - min(Y_raw)) for y in Y_raw]
#
# num_epoch = 10000
# learning_rate = 1e-3
# for e in range(num_epoch):
#     for i in range(len(X)):
#         x, y = X[i], Y[i]
#         g_a, g_b = gradient_loss(x, y)
#         a = a - learning_rate * g_a
#         b = b - learning_rate * g_b
# print(a, b)
# for i in range(len(X)):
#     x, y = X[i], Y[i]
#     print(f(x), y)
# x_pred = (x_pred_raw - min(X_raw)) / (max(X_raw) - min(X_raw))
# y_pred = f(x_pred)
# y_pred_raw = y_pred * (max(Y_raw) - min(Y_raw)) + min(Y_raw)
# print(x_pred_raw, y_pred_raw)



# 2.使用numpy 完成完成最小二乘法线性回归
# import numpy as np
#
# X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
# y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
#
# X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
# y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
#
# a, b = 0, 0
#
# num_epoch = 10000
# learning_rate = 1e-3
# for e in range(num_epoch):
#     # 手动计算损失函数关于自变量（模型参数）的梯度
#     y_pred = a * X + b
#     grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()
#
#     # 更新参数
#     a, b = a - learning_rate * grad_a, b - learning_rate * grad_b
#
# print(a, b)

# 3.使用tensorflow 2.0完成完成最小二乘法线性回归
# _*_ coding: utf-8 _*_
import numpy as np
import tensorflow as tf

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17000], dtype=np.float32)
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
print(X)
print(y)

X = tf.constant(X)
y = tf.constant(y)

w0 = tf.Variable(initial_value=0.)
w1 = tf.Variable(initial_value=0.)
variables = [w0, w1]

num_epoch = 1000
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = 0
        for i in range(len(variables)):
            y_pred += pow(X,i) * variables[i]
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))


x_test = 2019
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = 0
for i in range(len(variables)):
    y += pow(X, i) * variables[i]

y = (y_raw.max() - y_raw.min()) * y + y_raw.min()
print(y)
import matplotlib.pyplot as plt

plt.scatter(X_raw,y_raw)
plt.plot(X_raw,y,c='r')
plt.show()
