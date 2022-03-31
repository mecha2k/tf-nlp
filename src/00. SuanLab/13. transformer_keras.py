import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os, re

from tensorflow.keras.layers import Dense, Dropout, Embedding
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt


np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)
plt.style.use("seaborn")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


okt = Okt()
pattern = re.compile(pattern="([~.,!?\"':;()])")


def apply_morphs(lines):
    return [" ".join(okt.morphs(line.replace(" ", ""))) for line in lines]


df = pd.read_csv("../data/ChatBotData.csv")
print(len(df))

question = apply_morphs(df["Q"].to_numpy())
answer = apply_morphs(df["A"].to_numpy())
lines = np.concatenate([question, answer])
print(lines.shape)

words = []
for line in lines:
    line = re.sub(pattern=pattern, repl="", string=line)
    for word in line.split():
        words.append(word)
words = list(set([word for word in words if word]))
words[:0] = [PAD, STA, END, UNK]

vocab_size = len(words)
cha2idx = dict([(word, idx) for idx, word in enumerate(words)])
idx2cha = dict([(idx, word) for idx, word in enumerate(words)])

x_train, x_test, y_train, y_test = train_test_split(
    df["Q"].to_numpy(), df["A"].to_numpy(), test_size=0.2, random_state=42
)
print(x_train.shape)

max_len = 25
embed_dim = 128
ffn_dim = 128
num_heads = 8
num_layers = 2

epochs = 2
batch_size = 256
learning_rate = 0.001
xavier_initializer = True

train_input, train_input_len = encoder_preprocessing(apply_morphs(x_train), cha2idx)
train_output, train_output_len = decoder_output_preprocessing(apply_morphs(y_train), cha2idx)
train_target = decoder_target_preprocessing(apply_morphs(y_train), cha2idx)

valid_input, valid_input_len = encoder_preprocessing(apply_morphs(x_test), cha2idx)
valid_output, valid_output_len = decoder_output_preprocessing(apply_morphs(y_test), cha2idx)
valid_target = decoder_target_preprocessing(apply_morphs(y_test), cha2idx)
for line in valid_target[:10]:
    text = [idx2cha[x] + " " for x in line if idx2cha[x] != PAD]
    print("".join(text))


def make_train_ds():
    train_ds = tf.data.Dataset.from_tensor_slices((train_input, train_output, train_target))
    train_ds = (
        train_ds.shuffle(10000)
        .batch(batch_size)
        .map(lambda x, y, z: ({"inputs": x, "outputs": y}, z), num_parallel_calls=4)
        .repeat()
    )
    return train_ds
