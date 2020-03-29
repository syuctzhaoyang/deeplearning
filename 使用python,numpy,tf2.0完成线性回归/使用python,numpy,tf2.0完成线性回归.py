x = [137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
     106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21]
y = [145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
     62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30]
x_test = [128.15, 45.00, 141.43, 106.27, 99.00, 53.84, 85.36, 70.00]
# ---------------------------------------------------------
# 1.使用纯python代码完成线性回归
# meanX = sum(x) / len(x)
# meanY = sum(y) / len(y)
#
# sumXY = 0.0
# sumX = 0.0
#
# for i in range(len(x)):
#     sumXY += (x[i] - meanX) * (y[i] - meanY)
#     sumX += (x[i] - meanX) * (x[i] - meanX)
#
# w = sumXY / sumX
# b = meanY - w * meanX
#
# print('w = ', w)
# print('b=', b)
#
# for i in range(len(x_test)):
#     print(x_test[i], '\t', w * x_test[i] + b)

# -------------------------------------------------------
# 2.使用numpy 完成线性回归

# import numpy as np
# x = np.array(x)
# y = np.array(y)
# x_test = np.array(x_test)
#
# sumXY = np.sum((x - np.mean(x)) * (y - np.mean(y)))
# sumX = np.sum((x - np.mean(x)) * (x - np.mean(x)))
#
# w = sumXY / sumX
# b = np.mean(y) - w * np.mean(x)
#
# print('w = ', w)
# print('b=', b)
#
# y_pred = w * x_test + b
# for i in range(len(x_test)):
#     print(x_test[i], '\t', np.round(y_pred[i],2))

# 3. 使用tensorflow完成线性回归
# -----------------------------------------------
import tensorflow as tf

tf.enable_eager_execution()
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)

x_test = tf.convert_to_tensor(x_test)

meanX = tf.reduce_mean(x)
meanY = tf.reduce_mean(y)

sumXY = tf.reduce_sum((x - meanX) * (y - meanY))
sumX = tf.reduce_sum((x - meanX) * (x - meanX))

w = sumXY / sumX
b = meanY - w * meanX

print('w = ', w.numpy())
print('b=', b.numpy())

y_pred = w * x_test + b
for i in range(len(x_test)):
    print(x_test.numpy()[i], '\t', y_pred.numpy()[i])
