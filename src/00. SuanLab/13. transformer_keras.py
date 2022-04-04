import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.layers import Dense, Dropout, Embedding


np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)
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
    return tf.constant(encoded_vec.reshape((max_len, embed_dim)), dtype=tf.float32)


def scaled_dot_product_attention(query, key, value, mask=False):
    # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, key의 문장 길이)

    key_dim_size = float(key.get_shape().as_list()[-1])
    outputs = tf.matmul(query, key, transpose_b=True) / tf.sqrt(key_dim_size)

    if mask:
        diag_values = tf.ones_like(outputs[0, :, :])
        triangular = tf.linalg.LinearOperatorLowerTriangular(diag_values).to_dense()
        masks = tf.tile(tf.expand_dims(triangular, 0), multiples=[tf.shape(outputs)[0], 1, 1])
        paddings = tf.ones_like(masks) * (-(2 ** 30))
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    attention_map = tf.nn.softmax(outputs)

    return tf.matmul(attention_map, value)


query = tf.constant([[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 10, 10]], dtype=tf.float32)
key = tf.constant([[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 0, 10]], dtype=tf.float32)
value = tf.constant([[1, 0], [10, 0], [100, 5], [200, 15]], dtype=tf.float32)
print(scaled_dot_product_attention(query, key, value, mask=False))


def multi_head_attention(query, key, value, embed_dim, num_heads, mask=False):
    query = Dense(embed_dim, activation=tf.nn.relu)(query)
    key = Dense(embed_dim, activation=tf.nn.relu)(key)
    value = Dense(embed_dim, activation=tf.nn.relu)(value)

    query = tf.concat(tf.split(query, num_heads, axis=-1), axis=0)
    key = tf.concat(tf.split(key, num_heads, axis=-1), axis=0)
    value = tf.concat(tf.split(value, num_heads, axis=-1), axis=0)

    attention_map = scaled_dot_product_attention(query, key, value, mask)
    outputs = tf.concat(tf.split(attention_map, num_heads, axis=0), axis=-1)
    outputs = Dense(embed_dim, activation=tf.nn.relu)(outputs)

    return outputs


query = tf.random.normal(shape=(1, 2, 32), mean=0.0, stddev=1.0)
key = tf.random.normal(shape=(1, 2, 32), mean=0.0, stddev=1.0)
value = tf.random.normal(shape=(1, 2, 32), mean=0.0, stddev=1.0)
outputs = multi_head_attention(query, key, value, embed_dim=64, num_heads=8)
print(outputs.shape)


from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
import re

okt = Okt()

(PAD, STA, END, UNK) = ("<PAD>", "<STA>", "<END>", "<UNK>")
(PAD_IDX, STA_IDX, END_IDX, UNK_IDX) = (0, 1, 2, 3)

pattern = re.compile(pattern="([~.,!?\"':;()])")


def apply_morphs(lines):
    return [" ".join(okt.morphs(line.replace(" ", ""))) for line in lines]


def index2string(lines, dictionary):
    sentence = []
    finished = False
    for line in lines:
        sentence = [dictionary[idx] for idx in line["indexs"]]
    answer = ""
    for word in sentence:
        if word == END:
            finished = True
            break
        if word != PAD:
            answer += word + " "
    return answer, finished


# ---------------------------------------------------------------------------------------------
max_len = 25
embed_dim = 128
ffn_dim = 128
num_heads = 8
num_layers = 2

epochs = 1
batch_size = 256
dropout = 0.2
learning_rate = 0.001
xavier_initializer = True


cha2idx, idx2cha = np.load(file="../data/tf_dict.npy", allow_pickle=True)
train_input, train_output, train_target = np.load(file="../data/tf_train.npy", allow_pickle=True)
valid_input, valid_output, valid_target = np.load(file="../data/tf_valid.npy", allow_pickle=True)
vocab_size = len(cha2idx)
print(vocab_size)
for line in valid_output[:5]:
    text = [idx2cha[x] + " " for x in line if idx2cha[x] != PAD]
    print("".join(text))
for line in valid_target[:5]:
    text = [idx2cha[x] + " " for x in line if idx2cha[x] != PAD]
    print("".join(text))


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

# embed_dim: word embedding size, encoder/decoder input/output size
def encoders(inputs, embed_dim, num_heads, ffn_dim, num_layers):
    # inputs = Input(shape=(None, embed_dim))
    # mask = Input(shape=(1, 1, None))
    positional_encode = positional_encoding(max_len, embed_dim)

    if xavier_initializer:
        embedding_initializer = tf.keras.initializers.glorot_normal(seed=42)
    else:
        embedding_initializer = "uniform"

    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        embeddings_initializer=embedding_initializer,
    )

    outputs = embedding(inputs) + positional_encode
    outputs = Dropout(rate=dropout)(outputs)

    for i in range(num_layers):
        attentions = multi_head_attention(outputs, outputs, outputs, embed_dim, num_heads)
        attentions = Dropout(rate=dropout)(attentions)
        attentions = LayerNormalization(epsilon=1e-6)(attentions + outputs)

        outputs = Dense(ffn_dim, activation="relu")(attentions)
        outputs = Dense(embed_dim)(outputs)

        outputs = Dropout(rate=dropout)(outputs)
        outputs = LayerNormalization(epsilon=1e-6)(attentions + outputs)

    return outputs


def decoders(inputs, encoder_outputs, embed_dim, num_heads, ffn_dim, num_layers):
    positional_encode = positional_encoding(max_len, embed_dim)

    if xavier_initializer:
        embedding_initializer = tf.keras.initializers.glorot_normal(seed=42)
    else:
        embedding_initializer = "uniform"

    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        embeddings_initializer=embedding_initializer,
    )

    outputs = embedding(inputs) + positional_encode
    outputs = Dropout(rate=dropout)(outputs)

    outputs = inputs
    for i in range(num_layers):
        masked_attentions = multi_head_attention(
            inputs, inputs, inputs, embed_dim, num_heads, mask=True
        )
        masked_attentions = Dropout(rate=dropout)(masked_attentions)
        masked_attentions = LayerNormalization(epsilon=1e-6)(masked_attentions + outputs)

        self_attentions = multi_head_attention(
            masked_attentions, encoder_outputs, encoder_outputs, embed_dim, num_heads, mask=True
        )
        self_attentions = Dropout(rate=dropout)(self_attentions)
        self_attentions = LayerNormalization(epsilon=1e-6)(masked_attentions + self_attentions)

        outputs = Dense(ffn_dim, activation="relu")(self_attentions)
        outputs = Dense(embed_dim)(outputs)

        outputs = Dropout(rate=dropout)(outputs)
        outputs = LayerNormalization(epsilon=1e-6)(self_attentions + outputs)

    return outputs


inputs = Input(shape=(None,))
outputs = encoders(inputs, embed_dim, num_heads, ffn_dim, num_layers)
encoder_model = Model(inputs=inputs, outputs=outputs)
plot_model(model=encoder_model, to_file="images/13-encoder.png", show_shapes=True)


# def make_model(features, labels, mode, params):
#     TRAIN = mode == tf.estimator.ModeKeys.TRAIN
#     EVAL = mode == tf.estimator.ModeKeys.EVAL
#     PREDICT = mode == tf.estimator.ModeKeys.PREDICT
#
#     positional_encode = positional_encoding(params["max_len"], params["embed_dim"])
#     if params["xavier_initializer"]:
#         embedding_initializer = tf.keras.initializers.glorot_normal(seed=42)
#     else:
#         embedding_initializer = "uniform"
#
#     embedding = Embedding(
#         input_dim=params["vocab_size"],
#         output_dim=params["embed_dim"],
#         embeddings_initializer=embedding_initializer,
#     )
#
#     x_embedded_matrix = embedding(features["inputs"]) + positional_encode
#     y_embedded_matrix = embedding(features["outputs"]) + positional_encode
#
#     encoder_outputs = encoders(
#         x_embedded_matrix,
#         params["embed_dim"],
#         params["num_heads"],
#         params["ffn_dim"],
#         params["num_layers"],
#     )
#     decoder_outputs = decoders(
#         y_embedded_matrix,
#         encoder_outputs,
#         params["embed_dim"],
#         params["num_heads"],
#         params["ffn_dim"],
#         params["num_layers"],
#     )
#
#     logits = Dense(params["vocab_size"])(decoder_outputs)
#     predictions = tf.argmax(logits, axis=2)
#
#     if PREDICT:
#         predictions = {"indexs": predictions, "logits": logits}
#         return tf.estimator.EstimatorSpec(mode, predictions=predictions)
#
#     labels_ = tf.one_hot(labels, depth=params["vocab_size"])
#     loss = tf.reduce_mean(
#         tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_)
#     )
#     accuracy = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions)
#
#     metrics = {"accuracy": accuracy}
#     tf.summary.scalar("accuracy", accuracy[1])
#
#     if EVAL:
#         return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
#     assert TRAIN
#
#     optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params["learning_rate"])
#     train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
#
#     return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# transformer = tf.estimator.Estimator(
#     model_fn=make_model,
#     model_dir="../data/transformer",
#     params={
#         "vocab_size": vocab_size,
#         "embed_dim": embed_dim,
#         "num_heads": num_heads,
#         "ffn_dim": ffn_dim,
#         "num_layers": num_layers,
#         "max_len": max_len,
#         "learning_rate": learning_rate,
#         "xavier_initializer": xavier_initializer,
#     },
# )
#
# transformer.train(input_fn=lambda: make_train_ds(), steps=epochs)
# results = transformer.evaluate(input_fn=lambda: make_valid_ds())
# print(results)
