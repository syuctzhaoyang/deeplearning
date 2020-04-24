# _*_ coding: utf-8 _*_
import tensorflow as tf
import cv2
# 利用opencv读取图片，img为numpy.ndarray类型数据
img = cv2.imread(r'd:\img\lena.jpg')
#读取的图片shape为（512,512,3）
# tensorflow卷积时要求图片的shape为（图片数，图片行数，图片列数，图片通道数）
# 因此需要为图片增加一个维度，即图片数。此处图片数为1，增加维度时，默认值就为1.
# axis = 0，即在图片的最外围增加一个维度
x = tf.expand_dims(img,axis = 0)
# x中每个数据单元的类型为tf.int32,
# tf.keras.layers.Conv2D中的卷积核数据类型为tf.float32
# tensorflow卷积过程中要求图片与卷积核数据类型一致
#因此此处需转换数据类型为tf.float32
x = tf.cast(x, tf.float32)
# 2维卷积，输出通道为6个，即此处有6个卷积核。卷积核的值为tensorflow随机给定。
# kernel_size=[3,3]为每个卷积核大小设定为3*3.即卷积核为3行3列。
# tensorflow 也支持行、列数量不等的卷积核。
# 例如 kernel_size=[3,4]，即卷积核为3行3列。
# 卷积的对象实例化，实例名为layer
layer = tf.keras.layers.Conv2D(6, kernel_size=[3,3])
# 对图片x进行卷积
out = layer(x)
# 卷积的结果为out。
# 源图片尺寸512 * 512 ，卷积核尺寸为3 * 3,步长为1，因此输出图片尺寸为510 * 510
# 由于程序中使用了6个卷积核，即6个输出通道
# 此处out的shape为[1,510,510,6]
# 使用tf.squeeze方法去掉维数为1，使输出图片out的shape变为[510,510,6]
out = tf.squeeze(out)
# 循环6个通道，将每个通道视为一张灰度图片作为展示
for i in range(6):
    #循环遍历每个通道
    img_out0 = out[:,:,i]
    # 卷积后的图片中包含大量的负数值，将其转换为整数
    img_out0 = tf.abs(img_out0)
    # 将图片灰度值类型由tf.float32转换为tf.uint8，适于opencv展示
    img_out0 = tf.cast(img_out0,tf.uint8)
    cv2.imshow(f"img{i}",img_out0.numpy())
cv2.waitKey(0)


