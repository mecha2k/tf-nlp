import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from icecream import ic


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    @staticmethod
    def get_angles(position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )

        # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def __call__(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]


# sample_pos_encoding = PositionalEncoding(50, 128)
# print(sample_pos_encoding)
#
# aa = sample_pos_encoding.pos_encoding.numpy()
# print(aa)

position = 8
d_model = 4
vocab_size = 100

pos = tf.range(position, dtype=tf.float32)
ic(pos.shape)
ic(tf.reshape(pos, (-1, 1)))
pos = pos[:, tf.newaxis]
ic(pos.shape)
i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
ic(i.shape)
d_model = d_model

ic(i // 2)
ic((2 * (i // 2)) / tf.cast(d_model, tf.float32))

angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
angle_rads = pos * angles

ic(pos)
ic(angles)
ic(angle_rads)

# 배열의 짝수 인덱스(2i)에는 사인 함수 적용
sines = tf.math.sin(angle_rads[:, 0::2])
ic(sines)

# 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
cosines = tf.math.cos(angle_rads[:, 1::2])
ic(cosines)

angle_rads = np.zeros(angle_rads.shape)
angle_rads[:, 0::2] = sines
angle_rads[:, 1::2] = cosines
pos_encoding = tf.constant(angle_rads)[tf.newaxis, ...]
pos_encoding = tf.cast(pos_encoding, tf.float32)
ic(pos_encoding)
ic(pos_encoding.numpy()[0].shape)

inputs = tf.keras.Input(shape=(None,), name="inputs")
ic(inputs.shape)

# 인코더는 패딩 마스크 사용
padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

# 포지셔널 인코딩 + 드롭아웃
embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
ic(embeddings.shape)
ic(embeddings + pos_encoding)
embeddings += pos_encoding[:, : tf.shape(inputs)[1], :]
ic(embeddings.shape)
ic(tf.shape(inputs)[0])
ic(tf.shape(inputs)[1])
