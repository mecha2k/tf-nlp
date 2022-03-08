import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model


class BahdanauAttention:
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units=units)
        self.W2 = Dense(units=units)
        self.V = Dense(units=1)

    def __call__(self, values, query):  # 단, key와 value는 같음
        # query shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
        hidden_with_time_axis = tf.expand_dims(input=query, axis=1)
        assert query.shape[1] == embedding_dim

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights_ = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector_ = attention_weights_ * values
        context_vector_ = tf.reduce_sum(context_vector_, axis=1)

        return context_vector_, attention_weights_


max_len = 500
vocab_size = 10000
embedding_dim = 128

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

for input_, label_ in train_ds.take(1):
    print(input_.numpy()[0][:10])
    print(label_.numpy()[0])


inputs = Input(shape=(max_len,), dtype=tf.int32)
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=True)(
    inputs
)
x = Bidirectional(LSTM(units=64, dropout=0.5, return_sequences=True))(x)
x = Bidirectional(LSTM(units=64, dropout=0.5, return_sequences=True, return_state=True))(x)

lstm, forward_h, forward_c, backward_h, backward_c = x
print(lstm.shape, forward_h.shape, forward_c.shape, backward_h.shape, backward_c.shape)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

context_vector, attention_weights = BahdanauAttention(units=64)(lstm, state_h)

x = Dense(20, activation="relu")(context_vector)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
plot_model(model, "bilstm_attention.png", show_shapes=True)
model.summary()

callbacks = [keras.callbacks.ModelCheckpoint("bilstm_attention.keras", save_best_only=True)]
history = model.fit(
    train_ds,
    epochs=1,
    batch_size=batch_size,
    validation_data=test_ds,
    callbacks=callbacks,
    verbose=1,
)

model = keras.models.load_model("bilstm_attention.keras")
print("\n 테스트 정확도: %.4f" % (model.evaluate(test_ds)[1]))
