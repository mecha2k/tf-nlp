import numpy as np
import matplotlib.pyplot as plt


def step(x):
    return np.array(x > 0, dtype=int)


# -5.0부터 5.0까지 0.1 간격 생성
x = np.arange(-5.0, 5.0, 0.1)
y = step(x)
plt.title("Step Function")
plt.plot(x, y)
plt.savefig("images/step")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)

# 가운데 점선 추가
plt.plot([0, 0], [1.0, 0.0], ":")
plt.title("Sigmoid Function")
plt.savefig("images/sigmoid")

# -5.0부터 5.0까지 0.1 간격 생성
x = np.arange(-5.0, 5.0, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.plot([0, 0], [1.0, -1.0], ":")
plt.axhline(y=0, color="orange", linestyle="--")
plt.title("Tanh Function")
plt.savefig("images/tanh")


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.plot([0, 0], [5.0, 0.0], ":")
plt.title("Relu Function")
plt.savefig("images/relu")

a = 0.1


def leaky_relu(x):
    return np.maximum(a * x, x)


x = np.arange(-5.0, 5.0, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.plot([0, 0], [5.0, 0.0], ":")
plt.title("Leaky ReLU Function")
plt.savefig("images/leaky_relu")

x = np.arange(-5.0, 5.0, 0.1)  # -5.0부터 5.0까지 0.1 간격 생성
y = np.exp(x) / np.sum(np.exp(x))

plt.plot(x, y)
plt.title("Softmax Function")
plt.savefig("images/softmax")
