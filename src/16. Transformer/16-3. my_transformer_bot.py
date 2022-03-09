import tensorflow as tf
import re
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.utils import plot_model

from my_transformer import PositionalEmbedding, TransformerEncoder, TransformerDecoder
from my_sent_tokenizer import (
    MAX_LENGTH,
    VOCAB_SIZE,
    BUFFER_SIZE,
    BATCH_SIZE,
    tokenizer,
    START_TOKEN,
    END_TOKEN,
)


max_len = MAX_LENGTH - 1
vocab_size = VOCAB_SIZE
embed_dim = 256
num_heads = 8
dense_dim = 32

epochs = 50
batch_size = BATCH_SIZE
buffer_size = BUFFER_SIZE


encoder_inputs = Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(max_len, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

decoder_inputs = Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(max_len, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = Dropout(0.5)(x)
decoder_outputs = Dense(vocab_size, activation="softmax")(x)
transformer = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
plot_model(transformer, "images/my_transformer_bot.png", show_shapes=True)
transformer.summary()


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(embed_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")(
        y_true, y_pred
    )
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)


# transformer.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
transformer.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

callbacks = [keras.callbacks.ModelCheckpoint("../data/my_transformer_bot.keras")]
# transformer.fit(dataset, epochs=epochs, callbacks=callbacks)

transformer = keras.models.load_model(
    "../data/my_transformer_bot.keras",
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerEncoder": TransformerEncoder,
        "TransformerDecoder": TransformerDecoder,
    },
)


def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence


def evaluate(sentence):
    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
    output = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
        predictions = transformer(inputs=[sentence, output], training=False)
        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN[0]):
            break
        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)
    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
    print("Input: {}".format(sentence))
    print("Output: {}".format(predicted_sentence))
    return predicted_sentence


predict("영화 볼래?")
predict("고민이 있어")
predict("너무 화가나")
predict("게임하고싶은데 할래?")
predict("나 너 좋아하는 것 같아")
predict("딥 러닝 자연어 처리를 잘 하고 싶어")
