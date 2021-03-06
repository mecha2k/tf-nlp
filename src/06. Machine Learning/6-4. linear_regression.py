import tensorflow as tf

print(tf.__version__)

w = tf.Variable(2.0)


def f(w):
    y = w ** 2
    z = 2 * y + 5
    return z


with tf.GradientTape() as tape:
    z = f(w)

gradients = tape.gradient(z, [w])
print(gradients)

# 2. 자동 미분을 이용한 선형 회귀 구현

# 학습될 가중치 변수를 선언
W = tf.Variable(4.0)
b = tf.Variable(1.0)


@tf.function
def hypothesis(x):
    return W * x + b


x_test = [3.5, 5, 5.5, 6]
print(hypothesis(x_test).numpy())


@tf.function
def mse_loss(y_pred, y):
    # 두 개의 차이값을 제곱을 해서 평균을 취한다.
    return tf.reduce_mean(tf.square(y_pred - y))


x = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95]  # 각 공부하는 시간에 맵핑되는 성적

optimizer = tf.optimizers.SGD(0.01)

for i in range(301):
    with tf.GradientTape() as tape:
        # 현재 파라미터에 기반한 입력 x에 대한 예측값을 y_pred
        y_pred = hypothesis(x)

        # 평균 제곱 오차를 계산
        cost = mse_loss(y_pred, y)

    # 손실 함수에 대한 파라미터의 미분값 계산
    gradients = tape.gradient(cost, [W, b])

    # 파라미터 업데이트
    optimizer.apply_gradients(zip(gradients, [W, b]))

    if i % 10 == 0:
        print(
            "epoch : {:3} | W의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(
                i, W.numpy(), b.numpy(), cost
            )
        )

x_test = [3.5, 5, 5.5, 6]
print(hypothesis(x_test).numpy())

# 3. 케라스로 구현하는 선형 회귀

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

x = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95]  # 각 공부하는 시간에 맵핑되는 성적

model = Sequential()

# 입력 x의 차원은 1, 출력 y의 차원도 1. 선형 회귀이므로 activation은 'linear'
model.add(Dense(1, input_dim=1, activation="linear"))

# sgd는 경사 하강법을 의미. 학습률(learning rate, lr)은 0.01.
sgd = optimizers.SGD(learning_rate=0.01)

# 손실 함수(Loss function)은 평균제곱오차 mse를 사용합니다.
model.compile(optimizer=sgd, loss="mse", metrics=["mse"])

# 주어진 x와 y데이터에 대해서 오차를 최소화하는 작업을 300번 시도합니다.
model.fit(x, y, epochs=300, verbose=0)

plt.plot(x, model.predict(x), "b", x, y, "k.")
