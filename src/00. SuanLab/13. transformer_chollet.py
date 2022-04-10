import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    Layer,
    Dropout,
    Dense,
    Embedding,
    MultiHeadAttention,
    LayerNormalization,
    GlobalMaxPooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)
plt.style.use("seaborn")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


max_len = 25
embed_dim = 128
ffn_dim = 512
num_heads = 8
num_layers = 4

epochs = 100
batch_size = 256
dropout = 0.2
learning_rate = 0.001
xavier_initializer = True


from konlpy.tag import Okt
import re

okt = Okt()

(PAD, STA, END, UNK) = ("<PAD>", "<STA>", "<END>", "<UNK>")
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


train_input, train_input_len = encoder_preprocessing(apply_morphs(x_train), cha2idx)
train_output, train_output_len = decoder_output_preprocessing(apply_morphs(y_train), cha2idx)
train_target = decoder_target_preprocessing(apply_morphs(y_train), cha2idx)
print(train_input.shape)

valid_input, valid_input_len = encoder_preprocessing(apply_morphs(x_test), cha2idx)
valid_output, valid_output_len = decoder_output_preprocessing(apply_morphs(y_test), cha2idx)
valid_target = decoder_target_preprocessing(apply_morphs(y_test), cha2idx)
print(valid_input.shape)


# cha2idx, idx2cha = np.load(file="../data/tf_dict.npy", allow_pickle=True)
# train_input, train_output, train_target = np.load(file="../data/tf_train.npy", allow_pickle=True)
# valid_input, valid_output, valid_target = np.load(file="../data/tf_valid.npy", allow_pickle=True)
# vocab_size = len(cha2idx)
# print(vocab_size)
# for line in valid_output[:5]:
#     text = [idx2cha[x] + " " for x in line if idx2cha[x] != PAD]
#     print("".join(text))
# for line in valid_target[:5]:
#     text = [idx2cha[x] + " " for x in line if idx2cha[x] != PAD]
#     print("".join(text))
# print(train_input.shape)
# print(valid_input.shape)


train_ds = tf.data.Dataset.from_tensor_slices((train_input, train_output, train_target))
train_ds = (
    train_ds.shuffle(10000)
    .batch(batch_size)
    .map(lambda x, y, z: ({"encoder": x, "decoder": y}, z), num_parallel_calls=4)
)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_input, valid_output, valid_target))
valid_ds = (
    valid_ds.shuffle(10000)
    .batch(batch_size)
    .map(lambda x, y, z: ({"encoder": x, "decoder": y}, z), num_parallel_calls=4)
)


class TransformerEncoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = Sequential(
            [
                Dense(dense_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
            }
        )
        return config


class TransformerDecoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = Sequential(
            [
                Dense(dense_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.layernorm_3 = LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
            }
        )
        return config

    @staticmethod
    def get_causal_attention_mask(inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0
        )
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        padding_mask = None
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)


class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    @staticmethod
    def compute_mask(inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "sequence_length": self.sequence_length,
                "input_dim": self.input_dim,
            }
        )
        return config


encoder_inputs = Input(shape=(None,), dtype="int64", name="encoder")
x = PositionalEmbedding(max_len, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, ffn_dim, num_heads)(x)

decoder_inputs = Input(shape=(None,), dtype="int64", name="decoder")
x = PositionalEmbedding(max_len, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, ffn_dim, num_heads)(x, encoder_outputs)
x = Dropout(0.5)(x)
decoder_outputs = Dense(vocab_size, activation="softmax")(x)
transformer = Model(
    inputs={"encoder": encoder_inputs, "decoder": decoder_inputs}, outputs=decoder_outputs
)
optimizer = Adam(learning_rate=learning_rate)
transformer.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
plot_model(transformer, "images/13-seq2seq_transformer.png", show_shapes=True)
transformer.summary()

callbacks = [
    EarlyStopping(monitor="val_accuracy", min_delta=0.0001, patience=5),
    ModelCheckpoint("../data/seq2seq_transformer.keras", save_best_only=True),
]
history = transformer.fit(
    train_ds, epochs=epochs, batch_size=batch_size, validation_data=valid_ds, callbacks=callbacks
)
transformer.save_weights("../data/transformer_chollet_weight.h5")


names = ["loss", "accuracy"]
plt.figure(figsize=(10, 5))
for i, name in enumerate(names):
    plt.subplot(1, 2, i + 1)
    plt.plot(history.history[name])
    plt.plot(history.history[f"val_{name}"])
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.legend([name, f"val_{name}"])
plt.savefig("images/13-transformer_keras", dpi=300)


def index2string(lines, dictionary):
    sentence = []
    finished = False
    sentence = [dictionary[idx] for idx in lines]
    answer = ""
    for word in sentence:
        if word == END:
            finished = True
            break
        if word != PAD:
            answer += word + " "
    return answer, finished


def chatbot(sentence):
    inputs, _ = encoder_preprocessing([sentence], cha2idx)
    outputs, _ = decoder_output_preprocessing([""], cha2idx)

    answer = ""
    for i in range(max_len):
        if i > 0:
            outputs, _ = decoder_output_preprocessing([answer], cha2idx)
        predictions = transformer([inputs, outputs])
        predictions = [tf.argmax(predictions[0, i, :]).numpy()]
        # predictions = tf.argmax(predictions, axis=2).numpy()
        answer, finished = index2string(predictions, idx2cha)
        if finished:
            break

    print("=" * 100)
    print("Question: ", sentence)
    print("Answer: ", answer)


transformer.load_weights("../data/transformer_chollet_weight.h5")

chatbot("안녕?")
chatbot("놀고 싶다.")
