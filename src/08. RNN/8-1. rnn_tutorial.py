import tensorflow as tf

# 1. 케라스(Keras)로 RNN 구현하기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN

model = Sequential()
model.add(SimpleRNN(3, input_shape=(2, 10)))
# model.add(SimpleRNN(3, input_length=2, input_dim=10))와 동일
model.summary()

model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8, 2, 10)))
model.summary()

model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8, 2, 10), return_sequences=True))
model.summary()

# 2. 파이썬으로 RNN 구현하기

import numpy as np

timesteps = 10  # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_dim = 4  # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8  # 은닉 상태의 크기. 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_dim))  # 입력에 해당되는 2D 텐서

# 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.
hidden_state_t = np.zeros((hidden_size,))  # 초기 은닉 상태는 0(벡터)로 초기화

# 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.# 8의 크기를 가지는 은닉 상태. 현재는 초기 은닉 상태로 모든 차원이 0의 값을 가짐.
print(hidden_state_t)

Wx = np.random.random((hidden_size, input_dim))  # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치.
Wh = np.random.random((hidden_size, hidden_size))  # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치.
b = np.random.random((hidden_size,))  # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias).

print(np.shape(Wx))
print(np.shape(Wh))
print(np.shape(b))

total_hidden_states = []

# 메모리 셀 동작
for input_t in inputs:  # 각 시점에 따라서 입력값이 입력됨.
    output_t = np.tanh(
        np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b
    )  # Wx * Xt + Wh * Ht-1 + b(bias)
    total_hidden_states.append(list(output_t))  # 각 시점의 은닉 상태의 값을 계속해서 축적
    print(np.shape(total_hidden_states))  # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis=0)
# 출력 시 값을 깔끔하게 해준다.

print(total_hidden_states)  # (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.

# 3. 깊은 순환 신경망(Deep Recurrent Neural Network)


model = Sequential()
model.add(SimpleRNN(hidden_size, input_length=10, input_dim=5, return_sequences=True))
model.add(SimpleRNN(hidden_size, return_sequences=True))
model.summary()

# 4. 양방향 순환 신경망(Bidirectional Recurrent Neural Network)

from tensorflow.keras.layers import Bidirectional

timesteps = 10
input_dim = 5

model = Sequential()
model.add(
    Bidirectional(SimpleRNN(hidden_size, return_sequences=True), input_shape=(timesteps, input_dim))
)
model.summary()

model = Sequential()
model.add(
    Bidirectional(SimpleRNN(hidden_size, return_sequences=True), input_shape=(timesteps, input_dim))
)
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True)))
model.summary()
