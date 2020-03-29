# _*_ coding: utf-8 _*_


x1 = [137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
     106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21]
x2 = [3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2]
y = [145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
     62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30]
# 1.使用numpy完成多元线性回归
# import numpy as np
# x1 = np.array(x1)
# x2 = np.array(x2)
# y = np.array(y)
#
# x0 = np.ones(len(x1))
# X = np.stack((x0,x1,x2),axis=1)
# Y = y.reshape(-1,1)
#
# Xt = np.transpose(X)
# XtX_1 = np.linalg.inv(np.matmul(Xt,X))
# XtX_1_Xt = np.matmul(XtX_1,Xt)
# W =np.matmul(XtX_1_Xt,Y)
#
# print(W.reshape(-1))
#
# print('请输入房屋面积和房间数，预测房屋销售价格：')
# x1_test = float(input('商品房面积：'))
# x2_test = int(input('房间数：'))
#
# y_pred = W[1] * x1_test + W[2] * x2_test + W[0]
# print('预测价格：', np.round(y_pred, 2), '万元')

# 1.使用tensorflow完成多元线性回归
import tensorflow as tf
tf.enable_eager_execution()
x1 = tf.convert_to_tensor(x1)
x2 = tf.convert_to_tensor(x2,tf.float32)
y = tf.convert_to_tensor(y,tf.float32)


x0 = tf.ones(tf.shape(x1))
print(x0.numpy())
print(x1.numpy())
print(x2.numpy())
X = tf.stack((x0,x1,x2),axis=1)
Y = tf.reshape(y,[-1,1])

Xt = tf.transpose(X)
XtX_1 = tf.linalg.inv(tf.matmul(Xt,X))
XtX_1_Xt = tf.matmul(XtX_1,Xt)
W =tf.matmul(XtX_1_Xt,Y)

print(tf.reshape(W,[-1]))

print('请输入房屋面积和房间数，预测房屋销售价格：')
x1_test = float(input('商品房面积：'))
x2_test = int(input('房间数：'))

y_pred = W[1] * x1_test + W[2] * x2_test + W[0]
y_pred = y_pred.numpy()[0]
print('预测价格：', round(y_pred, 2), '万元')