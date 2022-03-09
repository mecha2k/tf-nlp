import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import re

from tensorflow import keras
from my_transformer import PositionalEmbedding, TransformerEncoder, TransformerDecoder


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

# 서브워드텍스트인코더를 사용하여 질문과 답변을 모두 포함한 단어 집합(Vocabulary) 생성
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13
)

# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
VOCAB_SIZE = tokenizer.vocab_size + 2

print("시작 토큰 번호 :", START_TOKEN)
print("종료 토큰 번호 :", END_TOKEN)
print("단어 집합의 크기 :", VOCAB_SIZE)

# 서브워드텍스트인코더 토크나이저의 .encode()를 사용하여 텍스트 시퀀스를 정수 시퀀스로 변환.
print("Tokenized sample question: {}".format(tokenizer.encode(questions[20])))

# 서브워드텍스트인코더 토크나이저의 .encode()와 decode() 테스트해보기
# 임의의 입력 문장을 sample_string에 저장
sample_string = questions[20]

# encode() : 텍스트 시퀀스 --> 정수 시퀀스
tokenized_string = tokenizer.encode(sample_string)
print("정수 인코딩 후의 문장 {}".format(tokenized_string))

# decode() : 정수 시퀀스 --> 텍스트 시퀀스
original_string = tokenizer.decode(tokenized_string)
print("기존 문장: {}".format(original_string))

# 각 정수는 각 단어와 어떻게 mapping되는지 병렬로 출력
# 서브워드텍스트인코더는 의미있는 단위의 서브워드로 토크나이징한다. 띄어쓰기 단위 X 형태소 분석 단위 X
for ts in tokenized_string:
    print("{} ----> {}".format(ts, tokenizer.decode([ts])))

# 최대 길이를 40으로 정의
MAX_LENGTH = 40


# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)

    # 패딩
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding="post"
    )
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding="post"
    )

    return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_and_filter(questions, answers)

print("질문 데이터의 크기(shape) :", questions.shape)
print("답변 데이터의 크기(shape) :", answers.shape)

# 0번째 샘플을 임의로 출력
print(questions[0])
print(answers[0])

print("단어 집합의 크기(Vocab size): {}".format(VOCAB_SIZE))
print("전체 샘플의 수(Number of samples): {}".format(len(questions)))

# 텐서플로우 dataset을 이용하여 셔플(shuffle)을 수행하되, 배치 크기로 데이터를 묶는다.
# 또한 이 과정에서 교사 강요(teacher forcing)을 사용하기 위해서 디코더의 입력과 실제값 시퀀스를 구성한다.
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# 디코더의 실제값 시퀀스에서는 시작 토큰을 제거해야 한다.
dataset = tf.data.Dataset.from_tensor_slices(
    (
        (questions[:, :-1], answers[:, :-1]),
        answers[:, 1:]
        # {"input_1": questions[:, :-1], "input_2": answers[:, :-1]},  # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
        # {"outputs": answers[:, 1:]},  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
    )
)

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 임의의 샘플에 대해서 [:, :-1]과 [:, 1:]이 어떤 의미를 가지는지 테스트해본다.
print(answers[0])  # 기존 샘플
print(answers[:1][:, :-1])  # 마지막 패딩 토큰 제거하면서 길이가 39가 된다.
print(answers[:1][:, 1:])  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다. 길이는 역시 39가 된다.

tf.keras.backend.clear_session()

max_len = MAX_LENGTH - 1
vocab_size = VOCAB_SIZE
embed_dim = 256
num_heads = 8
dense_dim = 32

epochs = 50
batch_size = BATCH_SIZE
buffer_size = BUFFER_SIZE

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.utils import plot_model

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
