import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

np.random.seed(42)
plt.style.use("seaborn")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def positional_encoding(max_len, embed_dim):
    encoded_vec = np.array(
        [
            pos / np.power(10000, 2 * i / embed_dim)
            for pos in range(max_len)
            for i in range(embed_dim)
        ]
    )
    encoded_vec[0::2] = np.sin(encoded_vec[0::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.constant(encoded_vec.reshape(shape=(max_len, embed_dim)), dtype=tf.float32)


def layer_normalization(inputs, eps=1e-6):
    feature_shape = inputs.get_shape()[-1:]
    mean = tf.keras.backend.mean(x=inputs, axis=-1, keepdims=True)
    std = tf.keras.backend.std(x=inputs, axis=-1, keepdims=True)
    beta = tf.Variable(tf.zeros(shape=feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(shape=feature_shape), trainable=False)
    return beta + gamma * (inputs - mean) / (std + eps)


def layer_connection(inputs, layer, dropout=0.2):
    return layer_normalization(inputs + tf.keras.layers.Dropout(dropout)(layer))
