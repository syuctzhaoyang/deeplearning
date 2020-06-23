# 使用python语言完成深度学习

## 机器学习和深度学习本质都是最优化

## 一个最优化问题通常有两个最基本的因素

   1.目标函数，也就是你希望什么东西的什么指标达到最好；机器学习和深度学习里通常指代价函数，即（预测值-真实值）的平方和。有时代价函数要乘以0.5，是为了对代价函数求导时，弥合求导下来的2。要是在除以样本数m，是为了求出样本的平均误差，本质上没什么用。
   
   2. 优化对象，你期望通过改变哪些因素来使你的目标函数达到最优。机器学习和深度学习里通常对代价函数进行梯度求导，再乘以学习率（步长），修正w,b等参数值，以达到代价函数值的作用。当代价函数值缩小时，称迭代是收敛的。当代价函数值小于给定的值时，迭代结束。

## 函数的书写规范
   
      def threshold_function(x: float) -> int:
         y = x > 0
         return y.astype(int)

## 字典类型的两种添加方法

      network = {}
      network['w1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
      network2={'w1':  np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])}
      
## 均方误差(mse)是计算回归问题的loss，交叉熵(cross entropy)是计算分类问题的loss

##  方差，标准差，均方误差计算方法，其中方差，标准差是计算数据距平均数的离散程度，均方误差计算数据距真实值的距离

      import numpy as np

      x = [100,80,90,70,60]
      # 1.方差(varance)
      fangcha = np.var(x)
      print(fangcha)

      #2.标准差(std)
      biaozhuncha = np.std(x)
      print(biaozhuncha)    

      #3.平均平方误差(mse)
      print((0.5 * np.sum((x - np.array([90,80,70,80,80])) ** 2)))
      
      
##    交叉熵(cross entropy)公式

      def cross_entropy_err(y_hat: list, y: list) -> float:
         delta = 1e-8
         return -np.sum(y * np.log(y_hat + delta))
      # y_hat 为预测值，y为真实值，即target
###    直接计算交叉熵，值太大，不利于比较，通常计算交叉熵之前使用softmax函数，在保证关系不变的情况下缩小计算数值

##    softmax函数

      def softmax_function(x: float) -> float:
         return np.exp(x) / np.sum(np.exp(x))
         
##    softmax函数与交叉熵完整示例

      x = np.array([90,80,80,80,60])   # x 为计算出来的值
     y = np.array([1,0,1,0,0])   # y 为真实值，通常交叉熵(cross entropy)是计算分类问题，y通常为one-hot类型
      
      def softmax_function(x: float) -> float:
         return np.exp(x) / np.sum(np.exp(x))  
         
      def cross_entropy_err(y_hat: list, y: list) -> float:
         delta = 1e-8
         return -np.sum(y * np.log(y_hat + delta))
         
       print(cross_entropy_err(softmax_function(x), y))
      
      import tensorflow as tf

##  sparse_categorical_crossentropy计算稀疏分类交叉熵说明
      #凡是有sparse字样的，y_true为标量，系统会自动转换成one-hot编码，y_pred为向量组合
      # 0 <----> [0.9, 0.05, 0.05],前面的0表示序号为0的分类，后面数字是序号的softmax值
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          y_true=tf.constant([0, 1, 2]),
          y_pred=tf.constant([[0.9, 0.05, 0.05], [0.05, 0.89, 0.06], [0.05, 0.01, 0.94]]))
      print('Loss: ', loss.numpy())
