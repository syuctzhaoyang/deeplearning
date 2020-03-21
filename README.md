# 使用python语言完成深度学习

## 函数的书写规范
   
      def threshold_function(x: float) -> int:
         y = x > 0
         return y.astype(int)

## 字典类型的两种添加方法

      network = {}
      network['w1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
      network2={'w1':  np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])}
      
## 均方差(mse)是计算回归问题的loss，交叉熵(cross entropy)是计算分类问题的loss
