import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 숫자 10부터 1

model = Sequential()
model.add(Dense(1, input_dim=1, activation="sigmoid"))

sgd = optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["binary_accuracy"])

model.fit(x, y, epochs=200)

plt.plot(x, model.predict(x), "b", x, y, "k.")
plt.savefig("images/06-01")

print(model.predict([1, 2, 3, 4, 4.5]))
print(model.predict([11, 21, 31, 41, 500]))
