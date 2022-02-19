import numpy as np
import tensorflow as tf
from icecream import ic

x = tf.constant([[1, 2], [3, 4]])
y = tf.constant([[0, 10], [20, 30]])

## [TensorFlow] matrix element-wise product
ic(tf.math.multiply(x, y))
ic(x * y)
## [numpy] matrix element-wise produce
ic(np.array(x) * np.array(y))
## matrix multiplication using tf.matmul()
ic(tf.matmul(x, y))
## [numpy] matrix multiplication using np.dot()
ic(np.dot(x, y))
## [numpy] matrix multiplication using np.matmul()
ic(np.matmul(x, y))
## casting tensor to numpy's array
ic(tf.matmul(x, y).numpy())

x = tf.constant([[1, 2], [3, 4]])
y = tf.constant([[10], [20]])

## [TensorFlow] matrix element-wise product
ic(tf.math.multiply(x, y))
ic(x * y)
ic(tf.matmul(x, y))

# matmul은 a의 기준에서 맨 마지막 axis에 대해서 b에 곱해줌. 이는 transpose_b=True와 동일함.
a = tf.constant(tf.random.uniform(shape=(1, 6, 10, 300)))
b = tf.constant(tf.random.uniform(shape=(1, 6, 10, 300)))

b_t = tf.transpose(b, perm=[0, 1, 3, 2])  # (1, 6, 300, 10)
ic(tf.linalg.matmul(a, b_t))  # (1, 6, 10, 10)
ic(tf.linalg.matmul(a, b, transpose_b=True))  # (1, 6, 10, 10)
