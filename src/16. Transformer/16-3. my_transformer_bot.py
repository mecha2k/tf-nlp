import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import text_dataset_from_directory, plot_model
from tensorflow.keras.layers import (
    Layer,
    TextVectorization,
    Dropout,
    Dense,
    Embedding,
    MultiHeadAttention,
    LayerNormalization,
    GlobalMaxPooling1D,
)

import re


class TransformerEncoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
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
        attention_output = self.attention(query=inputs, value=inputs, attention_mask=mask)
        print("inputs: ", inputs.shape)
        print("attention: ", attention_output.shape)
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
        print("tokens: ", embedded_tokens.shape)
        print("positions: ", embedded_positions.shape)
        return embedded_tokens + embedded_positions

    @staticmethod
    def compute_mask(inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "sequence_length": self.sequence_length,
                "input_dim": self.input_dim,
            }
        )
        return config


train_data = pd.read_csv("../data/ChatBotData.csv")
print(train_data.head())
print("챗봇 샘플의 개수 :", len(train_data))
print(train_data.isna().sum())

questions = []
for sentence in train_data["Q"]:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data["A"]:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)

print(questions[:5])
print(answers[:5])


#
# batch_size = 32
# train_ds = text_dataset_from_directory("../data/aclImdb/train", batch_size=batch_size)
# val_ds = text_dataset_from_directory("../data/aclImdb/val", batch_size=batch_size)
# test_ds = text_dataset_from_directory("../data/aclImdb/test", batch_size=batch_size)
# text_only_train_ds = train_ds.map(lambda x, y: x)
#
#
# max_length = 600
# max_tokens = 20000
# sequence_length = max_length
# vocab_size = max_tokens
#
#
# text_vectorization = TextVectorization(
#     max_tokens=max_tokens,
#     output_mode="int",
#     output_sequence_length=max_length,
# )
# text_vectorization.adapt(text_only_train_ds)
# vocabulary = text_vectorization.get_vocabulary()
# inverse_vocab = dict(enumerate(vocabulary))
#
# int_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
# int_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
# int_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
#
# for inputs, targets in int_train_ds:
#     print(inputs[0])
#     print(inputs.shape)
#     print(targets.shape)
#     print(vocabulary[:10])
#     decoded_sentence = " ".join(inverse_vocab[int(idx)] for idx in inputs[0])
#     print(decoded_sentence)
#     break
#
# embed_dim = 256
# num_heads = 2
# dense_dim = 32
#
#
# inputs = Input(shape=(None,), dtype="int64")
# # x = Embedding(vocab_size, embed_dim)(inputs)
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
# x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
# x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
# x = GlobalMaxPooling1D()(x)
# x = Dropout(0.5)(x)
# outputs = Dense(1, activation="sigmoid")(x)
# model = Model(inputs, outputs)
# model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
# plot_model(model, "images/transformer.png", show_shapes=True)
# model.summary()
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("../data/full_transformer_encoder.keras", save_best_only=True)
# ]
# model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)
#
# model = keras.models.load_model(
#     "../data/full_transformer_encoder.keras",
#     custom_objects={
#         "TransformerEncoder": TransformerEncoder,
#         "PositionalEmbedding": PositionalEmbedding,
#     },
# )
# print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
