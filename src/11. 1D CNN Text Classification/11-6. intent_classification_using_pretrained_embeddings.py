import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report


train_data = pd.read_csv("../data/intent_train_data.csv")
test_data = pd.read_csv("../data/intent_test_data.csv")
print(train_data)
print(test_data)

intent_train = train_data["intent"].tolist()
label_train = train_data["label"].tolist()
intent_test = test_data["intent"].tolist()
label_test = test_data["label"].tolist()

print("훈련용 문장의 수 :", len(intent_train))
print("훈련용 레이블의 수 :", len(label_train))
print("테스트용 문장의 수 :", len(intent_test))
print("테스트용 레이블의 수 :", len(label_test))

print(intent_train[:5])
print(label_train[:5])

print(intent_train[2000:2002])
print(label_train[2000:2002])

print(intent_train[4000:4002])
print(label_train[4000:4002])

print(intent_train[6000:6002])
print(label_train[6000:6002])

print(intent_train[8000:8002])
print(label_train[8000:8002])

print(intent_train[10000:10002])
print(label_train[10000:10002])

train_data["label"].value_counts().plot(kind="bar")
plt.savefig("images/06-01", dpi=300)

# 레이블 인코딩. 레이블에 고유한 정수를 부여
idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(label_train)

label_train = idx_encode.transform(label_train)  # 주어진 고유한 정수로 변환
label_test = idx_encode.transform(label_test)  # 고유한 정수로 변환

label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
print("레이블과 정수의 맵핑 관계 :", label_idx)

print(intent_train[:5])
print(label_train[:5])

print(intent_test[:5])
print(label_test[:5])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(intent_train)
sequences = tokenizer.texts_to_sequences(intent_train)
print(sequences[:5])  # 상위 5개 샘플 출력

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print("단어 집합(Vocabulary)의 크기 :", vocab_size)

print("문장의 최대 길이 :", max(len(l) for l in sequences))
print("문장의 평균 길이 :", sum(map(len, sequences)) / len(sequences))

fig = plt.figure(figsize=(10, 6))
plt.hist(x=[len(s) for s in sequences], bins=50)
plt.xlabel("length of samples")
plt.ylabel("number of samples")
plt.savefig("images/06-02", dpi=300)


max_len = 35
intent_train = pad_sequences(sequences, maxlen=max_len)
label_train = to_categorical(np.asarray(label_train))
print("훈련 데이터의 크기(shape):", intent_train.shape)
print("훈련 데이터 레이블의 크기(shape):", label_train.shape)

print("훈련 데이터의 첫번째 샘플 :", intent_train[0])
print("훈련 데이터의 첫번째 샘플의 레이블 :", label_train[0])

indices = np.arange(intent_train.shape[0])
np.random.shuffle(indices)
print("랜덤 시퀀스 :", indices)

intent_train = intent_train[indices]
label_train = label_train[indices]

n_of_val = int(0.1 * intent_train.shape[0])
print("검증 데이터의 개수 :", n_of_val)

X_train = intent_train[:-n_of_val]
y_train = label_train[:-n_of_val]
X_val = intent_train[-n_of_val:]
y_val = label_train[-n_of_val:]
X_test = intent_test
y_test = label_test

print("훈련 데이터의 크기(shape):", X_train.shape)
print("검증 데이터의 크기(shape):", X_val.shape)
print("훈련 데이터 레이블의 크기(shape):", y_train.shape)
print("검증 데이터 레이블의 크기(shape):", y_val.shape)
print("테스트 데이터의 개수 :", len(X_test))
print("테스트 데이터 레이블의 개수 :", len(y_test))

# 2. 사전 훈련된 워드 임베딩 사용하기
# 윈도우 환경을 사용하시는 분들은 http://nlp.stanford.edu/data/glove.6B.zip 링크에 가셔서 직접 다운로드하시고 압축푸시면 됩니다.

embedding_dict = dict()
f = open(os.path.join("../data/glove.6B/glove.6B.100d.txt"), encoding="utf-8")
for line in f:
    word_vector = line.split()
    word = word_vector[0]

    # 100개의 값을 가지는 array로 변환
    word_vector_arr = np.asarray(word_vector[1:], dtype="float32")
    embedding_dict[word] = word_vector_arr
f.close()

print("%s개의 Embedding vector가 있습니다." % len(embedding_dict))
print(embedding_dict["respectable"])
print(len(embedding_dict["respectable"]))

embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
print("임베딩 테이블의 크기(shape) :", np.shape(embedding_matrix))

for word, i in word_index.items():
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 3. 1D CNN을 이용한 의도 분류
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding,
    Dropout,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Input,
    Flatten,
    Concatenate,
)

kernel_sizes = [2, 3, 5]
num_filters = 512
dropout_ratio = 0.5

model_input = Input(shape=(max_len,))
output = Embedding(
    vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False
)(model_input)

conv_blocks = []

for size in kernel_sizes:
    conv = Conv1D(
        filters=num_filters, kernel_size=size, padding="valid", activation="relu", strides=1
    )(output)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)

output = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
output = Dropout(dropout_ratio)(output)
model_output = Dense(len(label_idx), activation="softmax")(output)

model = Model(model_input, model_output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.summary()

history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))

fig = plt.figure(figsize=(10, 6))
epochs = range(1, len(history.history["acc"]) + 1)
plt.plot(epochs, history.history["acc"])
plt.plot(epochs, history.history["val_acc"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend(["train", "test"], loc="lower right")
plt.savefig("images/06-03", dpi=300)

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_len)

y_predicted = model.predict(X_test)
y_predicted = y_predicted.argmax(axis=-1)  # 예측을 정수 시퀀스로 변환

y_predicted = idx_encode.inverse_transform(y_predicted)  # 정수 시퀀스를 레이블에 해당하는 텍스트 시퀀스로 변환
y_test = idx_encode.inverse_transform(y_test)  # 정수 시퀀스를 레이블에 해당하는 텍스트 시퀀스로 변환

print("accuracy: ", sum(y_predicted == y_test) / len(y_test))
print("Precision, Recall and F1-Score:\n\n", classification_report(y_test, y_predicted))
