import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    Layer,
    TextVectorization,
    Dropout,
    Dense,
    Embedding,
    MultiHeadAttention,
    LayerNormalization,
)
from tensorflow.keras.utils import plot_model
import string, re, random
import numpy as np


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


#
# text_file = "../data/spa-eng/spa.txt"
# with open(text_file, encoding="utf-8") as f:
#     lines = f.read().split("\n")[:-1]
# text_pairs = []
# for line in lines:
#     english, spanish = line.split("\t")
#     spanish = "[start] " + spanish + " [end]"
#     text_pairs.append((english, spanish))
# print(random.choice(text_pairs))
#
# random.shuffle(text_pairs)
# num_val_samples = int(0.15 * len(text_pairs))
# num_train_samples = len(text_pairs) - 2 * num_val_samples
# train_pairs = text_pairs[:num_train_samples]
# val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
# test_pairs = text_pairs[num_train_samples + num_val_samples :]
#
# strip_chars = string.punctuation + "¿"
# strip_chars = strip_chars.replace("[", "")
# strip_chars = strip_chars.replace("]", "")
#
#
# def custom_standardization(input_string):
#     lowercase = tf.strings.lower(input_string)
#     return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")
#
#
# vocab_size = 15000
# sequence_length = 20
#
# source_vectorization = TextVectorization(
#     max_tokens=vocab_size,
#     output_mode="int",
#     output_sequence_length=sequence_length,
# )
# target_vectorization = TextVectorization(
#     max_tokens=vocab_size,
#     output_mode="int",
#     output_sequence_length=sequence_length + 1,
#     standardize=custom_standardization,
# )
# train_english_texts = [pair[0] for pair in train_pairs]
# train_spanish_texts = [pair[1] for pair in train_pairs]
# source_vectorization.adapt(train_english_texts)
# target_vectorization.adapt(train_spanish_texts)
#
# batch_size = 64
#
#
# def format_dataset(eng, spa):
#     eng = source_vectorization(eng)
#     spa = target_vectorization(spa)
#     return (
#         {
#             "english": eng,
#             "spanish": spa[:, :-1],
#         },
#         spa[:, 1:],
#     )
#
#
# def make_dataset(pairs):
#     eng_texts, spa_texts = zip(*pairs)
#     eng_texts = list(eng_texts)
#     spa_texts = list(spa_texts)
#     dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.map(format_dataset, num_parallel_calls=4)
#     return dataset.shuffle(2048).prefetch(16).cache()
#
#
# train_ds = make_dataset(train_pairs)
# val_ds = make_dataset(val_pairs)
#
# for inputs, targets in train_ds.take(1):
#     print(f"inputs['english'].shape: {inputs['english'].shape}")
#     print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
#     print(f"targets.shape: {targets.shape}")
#
#
# embed_dim = 256
# dense_dim = 2048
# num_heads = 8
#
# encoder_inputs = Input(shape=(None,), dtype="int64", name="english")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
# encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
#
# decoder_inputs = Input(shape=(None,), dtype="int64", name="spanish")
# x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
# x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
# x = Dropout(0.5)(x)
# decoder_outputs = Dense(vocab_size, activation="softmax")(x)
# transformer = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# transformer.compile(
#     optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# plot_model(transformer, "images/my_transformer_bot.png", show_shapes=True)
# transformer.summary()
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("../data/seq2seq_transformer.keras", save_best_only=True)
# ]
# # transformer.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=callbacks)
#
# transformer = keras.models.load_model(
#     "../data/seq2seq_transformer.keras",
#     custom_objects={
#         "PositionalEmbedding": PositionalEmbedding,
#         "TransformerEncoder": TransformerEncoder,
#         "TransformerDecoder": TransformerDecoder,
#     },
# )
#
# spa_vocab = target_vectorization.get_vocabulary()
# spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
# max_decoded_sentence_length = 20
#
#
# def decode_sequence1(input_sentence):
#     tokenized_input_sentence = source_vectorization([input_sentence])
#     decoded_sentence = "[start]"
#     for i in range(max_decoded_sentence_length):
#         tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
#         predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
#         sampled_token_index = np.argmax(predictions[0, i, :])
#         sampled_token = spa_index_lookup[sampled_token_index]
#         decoded_sentence += " " + sampled_token
#         if sampled_token == "[end]":
#             break
#     return decoded_sentence
#
#
# test_eng_texts = [(pair[0], pair[1]) for pair in test_pairs]
# for _ in range(20):
#     input_sentence = random.choice(test_eng_texts)
#     print("-")
#     print(input_sentence[0])
#     print(decode_sequence1(input_sentence[0]))
#     print(decode_sequence1(input_sentence[1]))
