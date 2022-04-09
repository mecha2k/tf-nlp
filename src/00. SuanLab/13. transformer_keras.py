import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

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

epochs = 1
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
print(train_input.shape)
print(valid_input.shape)

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
        paddings = tf.ones_like(masks) * (-(2**30))
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

    for i in range(num_layers):
        masked_attentions = multi_head_attention(
            outputs, outputs, outputs, embed_dim, num_heads, mask=True
        )
        masked_attentions = Dropout(rate=dropout)(masked_attentions)
        masked_attentions = LayerNormalization(epsilon=1e-6)(masked_attentions + outputs)

        attentions = multi_head_attention(
            query=masked_attentions,
            key=encoder_outputs,
            value=encoder_outputs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mask=False,
        )
        attentions = Dropout(rate=dropout)(attentions)
        attentions = LayerNormalization(epsilon=1e-6)(masked_attentions + attentions)

        outputs = Dense(ffn_dim, activation="relu")(attentions)
        outputs = Dense(embed_dim)(outputs)
        outputs = Dropout(rate=dropout)(outputs)
        outputs = LayerNormalization(epsilon=1e-6)(attentions + outputs)

    return outputs


def transformers(vocab_size, embed_dim, num_heads, ffn_dim, num_layers):
    encoder_inputs = Input(shape=(None,))
    encoder_outputs = encoders(encoder_inputs, embed_dim, num_heads, ffn_dim, num_layers)
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_outputs)
    plot_model(model=encoder_model, to_file="images/13-encoder.png", show_shapes=True)

    decoder_inputs = Input(shape=(None,))
    decoder_outputs = decoders(
        decoder_inputs, encoder_outputs, embed_dim, num_heads, ffn_dim, num_layers
    )
    decoder_model = Model(inputs=[decoder_inputs, encoder_outputs], outputs=decoder_outputs)
    plot_model(model=decoder_model, to_file="images/13-decoder.png", show_shapes=True)

    enc_outputs = encoder_model(inputs=encoder_inputs)
    dec_outputs = decoder_model(inputs=[decoder_inputs, enc_outputs])
    outputs = Dense(units=vocab_size)(dec_outputs)

    return Model(inputs={"encoder": encoder_inputs, "decoder": decoder_inputs}, outputs=outputs)


model = transformers(vocab_size, embed_dim, num_heads, ffn_dim, num_layers)
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
plot_model(model=model, to_file="images/13-transformer.png", show_shapes=True)

callbacks = [
    EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=2),
    ModelCheckpoint(
        "../data/transformer_ex.keras",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    ),
]


history = model.fit(
    train_ds,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=valid_ds,
    verbose=1,
    callbacks=callbacks,
)

# history = model.fit(
#     x={"encoder": train_input, "decoder": train_output},
#     y=train_target,
#     batch_size=batch_size,
#     epochs=epochs,
#     verbose=1,
#     callbacks=callbacks,
#     validation_data=[{"encoder": valid_input, "decoder": valid_output}, valid_target],
# )

results = model.evaluate(
    x={"encoder": valid_input, "decoder": valid_output},
    y=valid_target,
    batch_size=batch_size,
    verbose=1,
)
model.save_weights("../data/transformer_weight.h5")
print(results)


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
    for line in lines:
        sentence = [dictionary[idx] for idx in line]
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
        predictions = model.predict(x=[inputs, outputs])
        predictions = tf.argmax(predictions, axis=2).numpy()
        answer, finished = index2string(predictions, idx2cha)
        if finished:
            break

    print("=" * 100)
    print("Question: ", sentence)
    print("Answer: ", answer)


model.load_weights("../data/transformer_weight.h5")

chatbot("안녕?")
chatbot("놀고 싶다.")
