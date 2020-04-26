# 导入数据集，参考https://devdocs.io/tensorflow~python/tf/keras/datasets/mnist/load_data
# 拷贝返回值和函数体
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 构建tf.data.Dataset，好处是能将图片与标签配对，以后交换训练顺序时，两两配对组不串位置
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# 创建映射函数，函数中将图片像素值归一化，将标签值做成one-hot编码
def preprocess(x,y):
    x = tf.reshape(x,[-1,28* 28])
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.one_hot(y, depth=10)
    return (x, y)
# 完成数据集的映射
train_db = train_db.map(preprocess)


from  tensorflow import keras
from  tensorflow.keras import layers, optimizers, datasets
# 此处可加入神经网络或是卷积神经网络的模型结构
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])
# 为模型设置优化方式为随机梯度下降
# 设置学习率，即更新参数时梯度所乘的参数，为0.001
optimizer = optimizers.SGD(learning_rate=0.001)


# 设置批处理大小，每次一组处理的样本数
# 批处理的好处是在训练过程中每次取一组多个值进行训练
# 计算误差时，以这一组值的平均值变化来调节下次训练的参数值
# 有效避免单个样本出现突变值影响训练效果
train_db = train_db.batch(100)
# 训练多轮，每训练完一轮，可将图片顺序打乱或是调整每张图片的角度，
# 以此来增强训练模型的健壮性
# 训练为5轮
for epoch in range(5):
    # 一次训练样本为100
    print(f'---------------------epoch:{epoch}---------------------------')
    for step,(x,y) in enumerate(train_db):
        # 训练，
        # 设置录像带
        with tf.GradientTape() as tape:
            # 利用模型和初始化参数求出第一组预测值
            out = model(x)
            # 计算预测值与标签值直接的差距
            # 此处使用预测值与标签值差的平方和，再除以一个batch个数，
            # 即求预测值与标签值差的平方和的平均值作为误差
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # 误差对w1, w2, w3, b1, b2, b3分别求偏导值
        # model.trainable_variables中包含了w1, w2, w3, b1, b2, b3等参数值
        grads = tape.gradient(loss, model.trainable_variables)
        # 更新模型参数
        # w' = w - lr * grad
        # 即w1' = w1 - lr * grad(w1)
        # b1' = b1 - lr * grad(b1)
        # w2' = w2 - lr * grad(w2)
        # b2' = b2 - lr * grad(b2)
        # w3' = w3 - lr * grad(w3)
        # b3' = b3 - lr * grad(b3)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(step, 'loss:', loss.numpy())
        # 打乱样本训练顺序
    train_db = train_db.shuffle(100000)
