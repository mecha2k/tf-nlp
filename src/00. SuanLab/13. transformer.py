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


def layer_normalization(inputs, eps=1e-6):
    feature_shape = inputs.get_shape()[-1:]
    mean = tf.keras.backend.mean(x=inputs, axis=-1, keepdims=True)
    std = tf.keras.backend.std(x=inputs, axis=-1, keepdims=True)
    beta = tf.Variable(tf.zeros(shape=feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(shape=feature_shape), trainable=False)
    return beta + gamma * (inputs - mean) / (std + eps)


def layer_norm_dropout(inputs, layer, dropout=0.2):
    return layer_normalization(inputs + tf.keras.layers.Dropout(dropout)(layer))


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


query = tf.random.normal(shape=(1, 1, 64), mean=0.0, stddev=1.0)
key = tf.random.normal(shape=(1, 1, 64), mean=0.0, stddev=1.0)
value = tf.random.normal(shape=(1, 1, 64), mean=0.0, stddev=1.0)
outputs = multi_head_attention(query, key, value, embed_dim=64, num_heads=8)
print(outputs.shape)


def feed_forward_network(inputs, embed_dim):
    feature_shape = inputs.get_shape()[-1]
    x = Dense(embed_dim, activation=tf.nn.relu)(inputs)
    outputs = Dense(feature_shape)(x)
    return outputs


def encoder_module(inputs, embed_dim, num_heads, ffn_dim):
    attentions = layer_norm_dropout(
        inputs, multi_head_attention(inputs, inputs, inputs, embed_dim, num_heads)
    )
    outputs = layer_norm_dropout(attentions, feed_forward_network(attentions, ffn_dim))
    return outputs


def encoders(inputs, embed_dim, num_heads, ffn_dim, num_layers):
    outputs = inputs
    for i in range(num_layers):
        outputs = encoder_module(outputs, embed_dim, num_heads, ffn_dim)
    return outputs


def decoder_module(inputs, encoder_outputs, embed_dim, num_heads, ffn_dim):
    masked_self_attention = layer_norm_dropout(
        inputs, multi_head_attention(inputs, inputs, inputs, embed_dim, num_heads, mask=True)
    )
    self_attention = layer_norm_dropout(
        masked_self_attention,
        multi_head_attention(
            masked_self_attention, encoder_outputs, encoder_outputs, embed_dim, num_heads
        ),
    )
    return layer_norm_dropout(self_attention, feed_forward_network(self_attention, ffn_dim))


def decoders(inputs, encoder_outputs, embed_dim, num_heads, ffn_dim, num_layers):
    outputs = inputs
    for i in range(num_layers):
        outputs = decoder_module(outputs, encoder_outputs, embed_dim, num_heads, ffn_dim)
    return outputs


from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
import re

okt = Okt()

(PAD, STA, END, UNK) = ("<PAD>", "<STA>", "<END>", "<UNK>")
(PAD_IDX, STA_IDX, END_IDX, UNK_IDX) = (0, 1, 2, 3)

pattern = re.compile(pattern="([~.,!?\"':;()])")


def apply_morphs(lines):
    return [" ".join(okt.morphs(line.replace(" ", ""))) for line in lines]


def encoder_preprocessing(lines, dictionary):
    sentences, sent_len = [], []
    for line in lines:
        line = re.sub(pattern=pattern, repl="", string=line)
        sentence = []
        for word in line.split():
            if dictionary.get(word) is not None:
                sentence.extend([dictionary[word]])
            else:
                sentence.extend([dictionary[UNK]])
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        sent_len.append(len(sentence))
        sentence += (max_len - len(sentence)) * [dictionary[PAD]]
        sentences.append(sentence)
    return np.array(sentences), sent_len


def decoder_output_preprocessing(lines, dictionary):
    sentences, sent_len = [], []
    for line in lines:
        line = re.sub(pattern=pattern, repl="", string=line)
        sentence = [dictionary[STA]] + [dictionary[word] for word in line.split()]
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        sent_len.append(len(sentence))
        sentence += (max_len - len(sentence)) * [dictionary[PAD]]
        sentences.append(sentence)
    return np.array(sentences), sent_len


def decoder_target_preprocessing(lines, dictionary):
    sentences = []
    for line in lines:
        line = re.sub(pattern=pattern, repl="", string=line)
        sentence = [dictionary[word] for word in line.split()]
        if len(sentence) >= max_len:
            sentence = sentence[: max_len - 1] + [dictionary[END]]
        else:
            sentence += [dictionary[END]]
        sentence += (max_len - len(sentence)) * [dictionary[PAD]]
        sentences.append(sentence)
    return np.array(sentences)


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


def make_model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    positional_encode = positional_encoding(params["max_len"], params["embed_dim"])
    if params["xavier_initializer"]:
        embedding_initializer = tf.keras.initializers.glorot_normal(seed=42)
    else:
        embedding_initializer = "uniform"

    embedding = Embedding(
        input_dim=params["vocab_size"],
        output_dim=params["embed_dim"],
        embeddings_initializer=embedding_initializer,
    )

    x_embedded_matrix = embedding(features["inputs"]) + positional_encode
    y_embedded_matrix = embedding(features["outputs"]) + positional_encode

    encoder_outputs = encoders(
        x_embedded_matrix,
        params["embed_dim"],
        params["num_heads"],
        params["ffn_dim"],
        params["num_layers"],
    )
    decoder_outputs = decoders(
        y_embedded_matrix,
        encoder_outputs,
        params["embed_dim"],
        params["num_heads"],
        params["ffn_dim"],
        params["num_layers"],
    )

    logits = Dense(params["vocab_size"])(decoder_outputs)
    predictions = tf.argmax(logits, axis=2)

    if PREDICT:
        predictions = {"indexs": predictions, "logits": logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    labels_ = tf.one_hot(labels, depth=params["vocab_size"])
    loss = tf.reduce_mean(
        tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_)
    )
    accuracy = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions)

    metrics = {"accuracy": accuracy}
    tf.summary.scalar("accuracy", accuracy[1])

    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    assert TRAIN

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


df = pd.read_csv("../data/ChatBotData.csv")
print(df)

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

epochs = 1
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


def make_valid_ds():
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_input, valid_output, valid_target))
    valid_ds = (
        valid_ds.shuffle(10000)
        .batch(batch_size)
        .map(lambda x, y, z: ({"inputs": x, "outputs": y}, z), num_parallel_calls=4)
    )
    return valid_ds


def make_datasets(inputs, outputs, targets, batch_size):
    datasets = tf.data.Dataset.from_tensor_slices((inputs, outputs, targets))
    datasets = (
        datasets.shuffle(10000)
        .batch(batch_size)
        .map(lambda x, y, z: ({"inputs": x, "outputs": y}, z), num_parallel_calls=4)
    )
    return datasets


# embed_dim: word embedding size, encoder/decoder input/output size
transformer = tf.estimator.Estimator(
    model_fn=make_model,
    model_dir="../data/transformer",
    params={
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "num_layers": num_layers,
        "max_len": max_len,
        "learning_rate": learning_rate,
        "xavier_initializer": xavier_initializer,
    },
)

transformer.train(input_fn=lambda: make_train_ds(), steps=epochs)
results = transformer.evaluate(input_fn=lambda: make_valid_ds())
print(results)


def chatbot(sentence):
    inputs, _ = encoder_preprocessing([sentence], cha2idx)
    outputs, _ = decoder_output_preprocessing([""], cha2idx)
    targets = decoder_target_preprocessing([""], cha2idx)

    answer = ""
    for i in range(max_len):
        if i > 0:
            outputs, _ = decoder_output_preprocessing([answer], cha2idx)
            targets = decoder_target_preprocessing([answer], cha2idx)

        predictions = transformer.predict(
            input_fn=lambda: make_datasets(inputs, outputs, targets, batch_size)
        )
        answer, finished = index2string(predictions, idx2cha)
        if finished:
            break

    return answer


# print(chatbot("안녕?"))
