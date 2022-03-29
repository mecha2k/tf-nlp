import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import os

from konlpy.tag import Mecab
from collections import Counter
from wordcloud import WordCloud
import squarify


np.random.seed(42)
plt.style.use("seaborn")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


time_steps = 10
input_features = 16
output_features = 32

inputs = np.random.random(size=(time_steps, input_features))
state_t = np.zeros(shape=(output_features,))

W = np.random.random(size=(output_features, input_features))
U = np.random.random(size=(output_features, output_features))
b = np.random.random(size=(output_features,))

outputs = []
for input_t in inputs:
    output = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    outputs.append(output)
    state_t = output

output_array = np.array(outputs)
output_sequences = np.stack(outputs, axis=0)
print(output_array.shape)
print(output_sequences.shape)
print(np.allclose(output_sequences, output_array))


from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

num_words = 2000
max_len = 100
batch_size = 256

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

x_train = pad_sequences(x_train, maxlen=max_len, padding="post")
x_test = pad_sequences(x_test, maxlen=max_len, padding="post")
print(x_train.shape)
print(x_test.shape)


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adam

inputs = Input(shape=(max_len,))
x = Embedding(input_dim=num_words, output_dim=32, input_length=max_len)(inputs)
x = LSTM(256, return_sequences=True)(x)
x = LSTM(256, return_sequences=True)(x)
x = LSTM(256)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)
plot_model(model, to_file="images/08-rnn.png", show_shapes=True)
model.summary()

model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

model.fit(x=x_train, y=y_train, batch_size=256, epochs=1, validation_split=0.2, verbose=1)
