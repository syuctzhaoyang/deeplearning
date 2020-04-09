# _*_ coding: utf-8 _*_

import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np

# 第一种，像素值为0~1之间的浮点数
# x = [np.random.random()for i in range(400)]
# x = np.reshape(x,[20,20,-1])
# plt.imshow(x[:,:,0])
# plt.show()

# 第二种，像素值为0~255之间的整数
# x = [np.random.randint(0,255,[1])for i in range(400)]
# x = np.reshape(x,[20,20,-1])
# plt.imshow(x[:,:,0])
# plt.show()


# 第3种，像素值为0~1之间的浮点数
x = [np.random.uniform(0,1,[3])for i in range(400)]
x = np.reshape(x,[20,20,-1])
plt.imshow(x[:,:,2])
plt.show()

# # 第4.1种，像素值为0~255之间的整数，每个元素由3个元组值构成
# x = [np.random.randint(0,255,[3])for i in range(400)]
# x = np.reshape(x,[20,20,-1])
# plt.imshow(x)
# plt.show()

# 第4.2种，像素值为0~255之间的整数，每个元素由3个元组值构成
# x = [np.random.uniform(0,255,3)for i in range(400)]
# x = np.reshape(x,[20,20,-1]).astype(np.int32)
# plt.imshow(x)
# plt.show()