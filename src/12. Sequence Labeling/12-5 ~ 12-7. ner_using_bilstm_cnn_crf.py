import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data = pd.read_csv("../data/ner_dataset.csv", encoding="latin1")
print(data)

print("데이터프레임 행의 개수 : {}".format(len(data)))
print("데이터에 Null 값이 있는지 유무 : " + str(data.isna().values.any()))
print("어떤 열에 Null값이 있는지 출력")
print("==============================")
print(data.isna().sum())

print("sentence # 열의 중복을 제거한 값의 개수 : {}".format(data["Sentence #"].nunique()))
print("Word 열의 중복을 제거한 값의 개수 : {}".format(data.Word.nunique()))
print("Tag 열의 중복을 제거한 값의 개수 : {}".format(data.Tag.nunique()))

print("Tag 열의 각각의 값의 개수 카운트")
print("================================")
print(data.groupby("Tag").size().reset_index(name="count"))

data = data.fillna(method="ffill")
print(data.tail())
print("데이터에 Null 값이 있는지 유무 : " + str(data.isna().values.any()))

data["Word"] = data["Word"].str.lower()
print("Word 열의 중복을 제거한 값의 개수 : {}".format(data.Word.nunique()))
print(data)

func = lambda temp: [
    (w, t) for w, t in zip(temp["Word"].values.tolist(), temp["Tag"].values.tolist())
]
tagged_sentences = [t for t in data.groupby("Sentence #").apply(func)]
print("전체 샘플 개수: {}".format(len(tagged_sentences)))
print(tagged_sentences[0])  # 첫번째 샘플 출력

sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences:  # 47,959개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence)  # 각 샘플에서 단어들은 sentence에 개체명 태깅 정보들은 tag_info에 저장.
    sentences.append(list(sentence))  # 각 샘플에서 단어 정보만 저장한다.
    ner_tags.append(list(tag_info))  # 각 샘플에서 개체명 태깅 정보만 저장한다.

print(sentences[0])
print(ner_tags[0])

print(sentences[98])
print(ner_tags[98])

print("샘플의 최대 길이 : %d" % max(len(l) for l in sentences))
print("샘플의 평균 길이 : %f" % (sum(map(len, sentences)) / len(sentences)))
plt.hist([len(s) for s in sentences], bins=50)
plt.xlabel("length of samples")
plt.ylabel("number of samples")
plt.savefig("images/05-01", dpi=300)

src_tokenizer = Tokenizer(oov_token="OOV")  # 모든 단어를 사용하지만 인덱스 1에는 단어 'OOV'를 할당한다.
src_tokenizer.fit_on_texts(sentences)
tar_tokenizer = Tokenizer(lower=False)  # 태깅 정보들은 내부적으로 대문자를 유지한채로 저장
tar_tokenizer.fit_on_texts(ner_tags)

vocab_size = len(src_tokenizer.word_index) + 1
tag_size = len(tar_tokenizer.word_index) + 1
print("단어 집합의 크기 : {}".format(vocab_size))
print("개체명 태깅 정보 집합의 크기 : {}".format(tag_size))
print("단어 OOV의 인덱스 : {}".format(src_tokenizer.word_index["OOV"]))

X_data = src_tokenizer.texts_to_sequences(sentences)
y_data = tar_tokenizer.texts_to_sequences(ner_tags)
print(X_data[0])
print(y_data[0])

word_to_index = src_tokenizer.word_index
index_to_word = src_tokenizer.index_word
ner_to_index = tar_tokenizer.word_index
index_to_ner = tar_tokenizer.index_word
index_to_ner[0] = "PAD"
print(index_to_ner)

decoded = []
for index in X_data[0]:  # 첫번째 샘플 안의 인덱스들에 대해서
    decoded.append(index_to_word[index])  # 다시 단어로 변환
print("기존의 문장 : {}".format(sentences[0]))
print("디코딩 문장 : {}".format(decoded))

max_len = 70
X_data = pad_sequences(X_data, padding="post", maxlen=max_len)
y_data = pad_sequences(y_data, padding="post", maxlen=max_len)
print(X_data.shape)
print(y_data.shape)

X_train, X_test, y_train_int, y_test_int = train_test_split(
    X_data, y_data, test_size=0.2, random_state=777
)
y_train = to_categorical(y_train_int, num_classes=tag_size)
y_test = to_categorical(y_test_int, num_classes=tag_size)

print("훈련 샘플 문장의 크기 : {}".format(X_train.shape))
print("훈련 샘플 레이블(정수 인코딩)의 크기 : {}".format(y_train_int.shape))
print("훈련 샘플 레이블(원-핫 인코딩)의 크기 : {}".format(y_train.shape))
print("테스트 샘플 문장의 크기 : {}".format(X_test.shape))
print("테스트 샘플 레이블(정수 인코딩)의 크기 : {}".format(y_test_int.shape))
print("테스트 샘플 레이블(원-핫 인코딩)의 크기 : {}".format(y_test.shape))

# char 정보를 사용하기 위한 추가 전처리

# char_vocab 만들기
words = list(set(data["Word"].values))
chars = set([w_i for w in words for w_i in w])
chars = sorted(list(chars))
print(chars)

char_to_index = {c: i + 2 for i, c in enumerate(chars)}
char_to_index["OOV"] = 1
char_to_index["PAD"] = 0

index_to_char = {}
for key, value in char_to_index.items():
    index_to_char[value] = key
print(sentences[0])

max_len_char = 15


def padding_char_indice(char_indice, max_len_char):
    return pad_sequences(char_indice, maxlen=max_len_char, padding="post", value=0)


def integer_coding(sentences):
    char_data = []
    for ts in sentences:
        # word_indice = [word_to_index[t] for t in ts]
        char_indice = [[char_to_index[char] for char in t] for t in ts]
        char_indice = padding_char_indice(char_indice, max_len_char)

        for chars_of_token in char_indice:
            if len(chars_of_token) > max_len_char:
                continue
        char_data.append(char_indice)
    return char_data


X_char_data = integer_coding(sentences)

# 정수 인코딩 이전의 기존 문장
print(sentences[0])
print(X_data[0])
print(X_char_data[0])

X_char_data = pad_sequences(X_char_data, maxlen=max_len, padding="post", value=0)

X_char_train, X_char_test, _, _ = train_test_split(
    X_char_data, y_data, test_size=0.2, random_state=777
)

X_char_train = np.array(X_char_train)
X_char_test = np.array(X_char_test)

print(X_train[0])
print(index_to_word[150])
print(X_char_train[0])
print(X_char_train[0][0])
print(" ".join([index_to_char[index] for index in X_char_train[0][0]]))

print("훈련 샘플 문장의 크기 : {}".format(X_train.shape))
print("훈련 샘플 레이블의 크기 : {}".format(y_train.shape))
print("훈련 샘플 char 데이터의 크기 : {}".format(X_char_train.shape))
print("테스트 샘플 문장의 크기 : {}".format(X_test.shape))
print("테스트 샘플 레이블의 크기 : {}".format(y_test.shape))

# BiLSTM을 이용한 개체명 인식
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    InputLayer,
    Bidirectional,
    TimeDistributed,
    Embedding,
)
from tensorflow.keras.optimizers import Adam

embedding_dim = 128
hidden_units = 256

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
model.add(TimeDistributed(Dense(tag_size, activation="softmax")))
model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train, batch_size=128, epochs=6, validation_split=0.1)
model.save("../data/bilstm.h5")

i = 13  # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]]))  # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1)  # 확률 벡터를 정수 인코딩으로 변경함.
true = np.argmax(y_test[i], -1)  # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for word, tag, pred in zip(X_test[i], true, y_predicted[0]):
    if word != 0:  # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

epochs = range(1, len(history.history["val_loss"]) + 1)
plt.plot(epochs, history.history["loss"])
plt.plot(epochs, history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("images/05-02", dpi=300)

from seqeval.metrics import f1_score, classification_report


def sequences_to_tag(sequences):  # 예측값을 index_to_ner를 사용하여 태깅 정보로 변경하는 함수.
    result = []
    for sequence in sequences:  # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
        temp = []
        for pred in sequence:  # 시퀀스로부터 예측값을 하나씩 꺼낸다.
            pred_index = np.argmax(pred)  # 예를 들어 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
            temp.append(index_to_ner[pred_index].replace("PAD", "O"))  # 'PAD'는 'O'로 변경
        result.append(temp)
    return result


y_predicted = model.predict([X_test])
pred_tags = sequences_to_tag(y_predicted)
test_tags = sequences_to_tag(y_test)

print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))

# BiLSTM-CRF를 이용한 개체명인식
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Input,
    Bidirectional,
    TimeDistributed,
    Embedding,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_crf import CRFModel
from seqeval.metrics import f1_score, classification_report

embedding_dim = 128
hidden_units = 64
dropout_ratio = 0.3

sequence_input = Input(shape=(max_len,), dtype=tf.int32, name="sequence_input")

model_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(
    sequence_input
)

model_bilstm = Bidirectional(LSTM(units=hidden_units, return_sequences=True))(model_embedding)

model_dropout = TimeDistributed(Dropout(dropout_ratio))(model_bilstm)

model_dense = TimeDistributed(Dense(tag_size, activation="relu"))(model_dropout)

base = Model(inputs=sequence_input, outputs=model_dense)
model = CRFModel(base, tag_size)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), metrics="accuracy")
model.summary()

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
mc = ModelCheckpoint(
    "bilstm_crf/cp.ckpt",
    monitor="val_decode_sequence_accuracy",
    mode="max",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)

history = model.fit(
    X_train, y_train_int, batch_size=128, epochs=15, validation_split=0.1, callbacks=[mc, es]
)

model.load_weights("bilstm_crf/cp.ckpt")

i = 13  # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]]))[0]  # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
true = np.argmax(y_test[i], -1)  # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for word, tag, pred in zip(X_test[i], true, y_predicted[0]):
    if word != 0:  # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

y_predicted = model.predict(X_test)[0]


def sequences_to_tag(sequences):  # 예측값을 index_to_ner를 사용하여 태깅 정보로 변경하는 함수.
    result = []
    for sequence in sequences:  # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
        temp = []
        for pred in sequence:  # 시퀀스로부터 예측값을 하나씩 꺼낸다.
            pred_index = np.argmax(pred)  # 예를 들어 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
            temp.append(index_to_ner[pred_index].replace("PAD", "O"))  # 'PAD'는 'O'로 변경
        result.append(temp)
    return result


def sequences_to_tag_for_crf(sequences):  # 예측값을 index_to_ner를 사용하여 태깅 정보로 변경하는 함수.
    result = []
    for sequence in sequences:  # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
        temp = []
        for pred in sequence:  # 시퀀스로부터 예측값을 하나씩 꺼낸다.
            # pred_index = np.argmax(pred) # 예를 들어 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
            pred_index = pred
            temp.append(index_to_ner[pred_index].replace("PAD", "O"))  # 'PAD'는 'O'로 변경
        result.append(temp)
    return result


pred_tags = sequences_to_tag_for_crf(y_predicted)
test_tags = sequences_to_tag(y_test)

print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))

# BiLSTM-CNN을 이용한 개체명인식
from tensorflow.keras.layers import (
    Embedding,
    Input,
    TimeDistributed,
    Dropout,
    concatenate,
    Bidirectional,
    LSTM,
    Conv1D,
    Dense,
    MaxPooling1D,
    Flatten,
)
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from seqeval.metrics import f1_score, classification_report

embedding_dim = 128
char_embedding_dim = 64
dropout_ratio = 0.5
hidden_units = 256
num_filters = 30
kernel_size = 3

# 단어 임베딩
words_input = Input(shape=(None,), dtype="int32", name="words_input")
words = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(words_input)

# char 임베딩
character_input = Input(
    shape=(
        None,
        max_len_char,
    ),
    name="char_input",
)
embed_char_out = TimeDistributed(
    Embedding(
        len(char_to_index),
        char_embedding_dim,
        embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
    ),
    name="char_embedding",
)(character_input)
dropout = Dropout(dropout_ratio)(embed_char_out)

# char 임베딩에 대해서는 Conv1D 수행
conv1d_out = TimeDistributed(
    Conv1D(
        kernel_size=kernel_size, filters=num_filters, padding="same", activation="tanh", strides=1
    )
)(dropout)
maxpool_out = TimeDistributed(MaxPooling1D(max_len_char))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(dropout_ratio)(char)

# char 임베딩을 Conv1D 수행한 뒤에 단어 임베딩과 연결
output = concatenate([words, char])

# 연결한 벡터를 가지고 문장의 길이만큼 LSTM을 수행
output = Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=dropout_ratio))(output)

# 출력층
output = TimeDistributed(Dense(tag_size, activation="softmax"))(output)

model = Model(inputs=[words_input, character_input], outputs=[output])
model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["acc"])
model.summary()

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
mc = ModelCheckpoint(
    "../data/bilstm_cnn.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True
)

history = model.fit(
    [X_train, X_char_train],
    y_train,
    batch_size=128,
    epochs=15,
    validation_split=0.1,
    verbose=1,
    callbacks=[es, mc],
)

model = load_model("../data/bilstm_cnn.h5")

i = 13  # 확인하고 싶은 테스트용 샘플의 인덱스.

# 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = model.predict([np.array([X_test[i]]), np.array([X_char_test[i]])])
y_predicted = np.argmax(y_predicted, axis=-1)  # 확률 벡터를 정수 인코딩으로 변경함.
true = np.argmax(y_test[i], -1)  # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for word, tag, pred in zip(X_test[i], true, y_predicted[0]):
    if word != 0:  # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

epochs = range(1, len(history.history["val_loss"]) + 1)
plt.plot(epochs, history.history["loss"])
plt.plot(epochs, history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("images/05-03", dpi=300)


def sequences_to_tag(sequences):  # 예측값을 index_to_ner를 사용하여 태깅 정보로 변경하는 함수.
    result = []
    for sequence in sequences:  # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
        temp = []
        for pred in sequence:  # 시퀀스로부터 예측값을 하나씩 꺼낸다.
            pred_index = np.argmax(pred)  # 예를 들어 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
            temp.append(index_to_ner[pred_index].replace("PAD", "O"))  # 'PAD'는 'O'로 변경
        result.append(temp)
    return result


y_predicted = model.predict([X_test, X_char_test])
pred_tags = sequences_to_tag(y_predicted)
test_tags = sequences_to_tag(y_test)

print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))

# BiLSTM-CNN-CRF를 이용한 개체명인식
import tensorflow as tf
from keras_crf import CRFModel

embedding_dim = 128
char_embedding_dim = 64
dropout_ratio = 0.5
hidden_units = 256
num_filters = 30
kernel_size = 3

# 단어 임베딩
words_input = Input(shape=(None,), dtype="int32", name="words_input")
words = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(words_input)

# char 임베딩
character_input = Input(
    shape=(
        None,
        max_len_char,
    ),
    name="char_input",
)
embed_char_out = TimeDistributed(
    Embedding(
        len(char_to_index),
        char_embedding_dim,
        embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
    ),
    name="char_embedding",
)(character_input)
dropout = Dropout(dropout_ratio)(embed_char_out)

# char 임베딩에 대해서는 Conv1D 수행
conv1d_out = TimeDistributed(
    Conv1D(
        kernel_size=kernel_size, filters=num_filters, padding="same", activation="tanh", strides=1
    )
)(dropout)
maxpool_out = TimeDistributed(MaxPooling1D(max_len_char))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(dropout_ratio)(char)

# char 임베딩을 Conv1D 수행한 뒤에 단어 임베딩과 연결
output = concatenate([words, char])

# 연결한 벡터를 가지고 문장의 길이만큼 LSTM을 수행
output = Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=dropout_ratio))(output)

# 출력층
output = TimeDistributed(Dense(tag_size, activation="relu"))(output)

base = Model(inputs=[words_input, character_input], outputs=[output])
model = CRFModel(base, tag_size)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), metrics="accuracy")
model.summary()

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
mc = ModelCheckpoint(
    "../data/bilstm_cnn_crf/cp.ckpt",
    monitor="val_decode_sequence_accuracy",
    mode="max",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)

history = model.fit(
    [X_train, X_char_train],
    y_train_int,
    batch_size=128,
    epochs=15,
    validation_split=0.1,
    callbacks=[mc, es],
)

model.load_weights("../data/bilstm_cnn_crf/cp.ckpt")

i = 13  # 확인하고 싶은 테스트용 샘플의 인덱스.

# 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = model.predict([np.array([X_test[i]]), np.array([X_char_test[i]])])[0]
true = np.argmax(y_test[i], -1)  # 원-핫 벡터를 정수 인코딩으로 변경.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for word, tag, pred in zip(X_test[i], true, y_predicted[0]):
    if word != 0:  # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

epochs = range(1, len(history.history["val_loss"]) + 1)
plt.plot(epochs, history.history["loss"])
plt.plot(epochs, history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("images/05-04", dpi=300)

y_predicted = model.predict([X_test, X_char_test])[0]
pred_tags = sequences_to_tag_for_crf(y_predicted)
test_tags = sequences_to_tag(y_test)

print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))

# BiLSTM-BiLSTM-CRF을 이용한 개체명 인식
embedding_dim = 128
char_embedding_dim = 64
dropout_ratio = 0.3
hidden_units = 64

word_ids = Input(batch_shape=(None, None), dtype="int32", name="word_input")
word_embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="word_embedding")(
    word_ids
)

char_ids = Input(batch_shape=(None, None, None), dtype="int32", name="char_input")
char_embeddings = Embedding(
    input_dim=(len(char_to_index)),
    output_dim=char_embedding_dim,
    embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
    name="char_embedding",
)(char_ids)

char_embeddings = TimeDistributed(Bidirectional(LSTM(hidden_units)))(char_embeddings)
word_embeddings = concatenate([word_embeddings, char_embeddings])

word_embeddings = Dropout(dropout_ratio)(word_embeddings)
output = Bidirectional(LSTM(units=hidden_units, return_sequences=True))(word_embeddings)
output = TimeDistributed(Dense(tag_size, activation="relu"))(output)

base = Model(inputs=[word_ids, char_ids], outputs=[output])
model = CRFModel(base, tag_size)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), metrics="accuracy")
model.summary()

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
mc = ModelCheckpoint(
    "../data/bilstm_bilstm_crf/cp.ckpt",
    monitor="val_decode_sequence_accuracy",
    mode="max",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)

history = model.fit(
    [X_train, X_char_train],
    y_train_int,
    batch_size=128,
    epochs=15,
    validation_split=0.1,
    callbacks=[mc, es],
)

model.load_weights("../data/bilstm_bilstm_crf/cp.ckpt")

i = 13  # 확인하고 싶은 테스트용 샘플의 인덱스.

# 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = model.predict([np.array([X_test[i]]), np.array([X_char_test[i]])])[0]
true = np.argmax(y_test[i], -1)  # 원-핫 벡터를 정수 인코딩으로 변경.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for word, tag, pred in zip(X_test[i], true, y_predicted[0]):
    if word != 0:  # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

epochs = range(1, len(history.history["val_loss"]) + 1)
plt.plot(epochs, history.history["loss"])
plt.plot(epochs, history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("images/05-05", dpi=300)

y_predicted = model.predict([X_test, X_char_test])[0]
pred_tags = sequences_to_tag_for_crf(y_predicted)
test_tags = sequences_to_tag(y_test)

print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))
