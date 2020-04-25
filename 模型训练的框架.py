# 导入数据集，参考https://devdocs.io/tensorflow~python/tf/keras/datasets/mnist/load_data
# 拷贝返回值和函数体
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 构建tf.data.Dataset，好处是能将图片与标签配对，以后交换训练顺序时，两两配对组不串位置
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# 创建映射函数，函数中将图片像素值归一化，将标签值做成one-hot编码
def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.one_hot(y, depth=10)
    return (x, y)
# 完成数据集的映射
train_db = train_db.map(preprocess)
# 设置批处理大小，每次一组处理的样本数
# 批处理的好处是在训练过程中每次取一组多个值进行训练
# 计算误差时，以这一组值的平均值变化来调节下次训练的参数值
# 有效避免单个样本出现突变值影响训练效果
train_db = train_db.batch(128)
# 训练多轮，每训练完一轮，可将图片顺序打乱或是调整每张图片的角度，
# 以此来增强训练模型的健壮性
for epoch in range(5):
    # 一次训练样本为128
    for idx,(x,y) in enumerate(train_db):
        # 训练，此处可加入神经网络或是卷积神经网络
        pass
    # 打乱样本训练顺序
    train_db = train_db.shuffle(100000)
    # 此处可打印每次迭代的误差，以此观察训练模型是否收敛
    print(epoch)
