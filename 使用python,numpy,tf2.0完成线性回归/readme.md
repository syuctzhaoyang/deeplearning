## 分别使用纯python语言、numpy、tensorflow 2.0完成线性回归

## 数据类型转换
    x = [137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
     106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21]
### python语言<------>numpy
    x = np.array(x)
    x = list(x)
### python语言,numpy<------>tensorflow 2.0
    x = tf.convert_to_tensor(x)
    x = x.numpy()
    
