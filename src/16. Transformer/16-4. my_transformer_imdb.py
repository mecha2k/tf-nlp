import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model
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


max_len = 500
vocab_size = 10000
embed_dim = 256
num_heads = 2
dense_dim = 32

epochs = 1
batch_size = 256
buffer_size = 100000


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

x_train_len = len(x_train)
buffer_size = x_train_len if buffer_size > x_train_len else buffer_size
print(max([len(sequence) for sequence in x_train]))
print(buffer_size)

x_train = pad_sequences(x_train, maxlen=max_len, padding="post")
x_test = pad_sequences(x_test, maxlen=max_len, padding="post")

word_to_idx = imdb.get_word_index()
idx_to_word = dict([(value, key) for (key, value) in word_to_idx.items()])
# 0, 1, and 2 are reserved indices for <pad>, <sos>, and <unk>
decoded_sentence = " ".join(idx_to_word.get(i - 3, "?") for i in x_train[0])
print(decoded_sentence)

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=buffer_size, seed=42)
    .batch(batch_size=batch_size)
)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size=batch_size)


inputs = Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(max_len, vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
plot_model(model, "images/transformer_imdb.png", show_shapes=True)
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("../data/full_transformer_imdb.keras", save_best_only=True)
]
model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=callbacks)

model = keras.models.load_model(
    "../data/full_transformer_imdb.keras",
    custom_objects={
        "TransformerEncoder": TransformerEncoder,
        "PositionalEmbedding": PositionalEmbedding,
    },
)
print(f"Test acc: {model.evaluate(test_ds)[1]:.3f}")
