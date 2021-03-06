import tensorflow as tf

print(tf.__version__)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 1. 다중 선형 회귀

# 중간 고사, 기말 고사, 가산점 점수
X = np.array([[70, 85, 11], [71, 89, 18], [50, 80, 20], [99, 20, 10], [50, 10, 10]])
y = np.array([73, 82, 72, 57, 34])  # 최종 성적

model = Sequential()
model.add(Dense(1, input_dim=3, activation="linear"))

sgd = optimizers.SGD(learning_rate=0.0001)
model.compile(optimizer=sgd, loss="mse", metrics=["mse"])
model.fit(X, y, epochs=2000, verbose=0)
print(model.predict(X))

X_test = np.array([[20, 99, 10], [40, 50, 20]])  # 각각 58점과 56점을 예측해야 합니다.
print(model.predict(X_test))

# 2. 다중 로지스틱 회귀

X = np.array([[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]])
y = np.array([0, 0, 0, 1, 1, 1])

model = Sequential()
model.add(Dense(1, input_dim=2, activation="sigmoid"))
model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["binary_accuracy"])

model.fit(X, y, epochs=2000, verbose=0)
print(model.predict(X))
