import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

texts = ["먹고 싶은 사과", "먹고 싶은 바나나", "길고 노란 바나나 바나나", "저는 과일이 좋아요"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
print(tokenizer.word_index)

print(tokenizer.texts_to_matrix(texts, mode="count"))

print(tokenizer.texts_to_matrix(texts, mode="binary"))

print(tokenizer.texts_to_matrix(texts, mode="tfidf").round(2))

print(tokenizer.texts_to_matrix(texts, mode="freq").round(2))

# 2. 20개 뉴스 그룹(Twenty Newsgroups) 데이터에 대한 이해

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

newsdata = fetch_20newsgroups(subset="train")

print(newsdata.keys())

print("훈련용 샘플의 개수 : {}".format(len(newsdata.data)))

print("총 주제의 개수 : {}".format(len(newsdata.target_names)))
print(newsdata.target_names)

print("첫번째 샘플의 레이블 : {}".format(newsdata.target[0]))

print("7번 레이블이 의미하는 주제 : {}".format(newsdata.target_names[7]))

print(newsdata.data[0])  # 첫번째 샘플 출력

data = pd.DataFrame(newsdata.data, columns=["email"])
data["target"] = pd.Series(newsdata.target)
print(data[:5])
data.info()

print(data.isnull().values.any())

print("중복을 제외한 샘플의 수 : {}".format(data["email"].nunique()))
print("중복을 제외한 주제의 수 : {}".format(data["target"].nunique()))

data["target"].value_counts().plot(kind="bar")

print(data.groupby("target").size().reset_index(name="count"))

newsdata_test = fetch_20newsgroups(subset="test", shuffle=True)
train_email = data["email"]
train_label = data["target"]
test_email = newsdata_test.data
test_label = newsdata_test.target

max_words = 10000
num_classes = 20


def prepare_data(train_data, test_data, mode):  # 전처리 함수
    tokenizer = Tokenizer(num_words=max_words)  # max_words 개수만큼의 단어만 사용한다.
    tokenizer.fit_on_texts(train_data)
    X_train = tokenizer.texts_to_matrix(train_data, mode=mode)  # 샘플 수 × max_words 크기의 행렬 생성
    X_test = tokenizer.texts_to_matrix(test_data, mode=mode)  # 샘플 수 × max_words 크기의 행렬 생성
    return X_train, X_test, tokenizer.index_word


X_train, X_test, index_to_word = prepare_data(train_email, test_email, "binary")  # binary 모드로 변환
y_train = to_categorical(train_label, num_classes)  # 원-핫 인코딩
y_test = to_categorical(test_label, num_classes)  # 원-핫 인코딩

print("훈련 샘플 본문의 크기 : {}".format(X_train.shape))
print("훈련 샘플 레이블의 크기 : {}".format(y_train.shape))
print("테스트 샘플 본문의 크기 : {}".format(X_test.shape))
print("테스트 샘플 레이블의 크기 : {}".format(y_test.shape))

print("빈도수 상위 1번 단어 : {}".format(index_to_word[1]))
print("빈도수 상위 9999번 단어 : {}".format(index_to_word[9999]))

# 3. 다층 퍼셉트론(Multilayer Perceptron, MLP)을 사용하여 텍스트 분류하기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def fit_and_evaluate(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(256, input_shape=(max_words,), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.1)
    score = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
    return score[1]


modes = ["binary", "count", "tfidf", "freq"]  # 4개의 모드를 리스트에 저장.

for mode in modes:  # 4개의 모드에 대해서 각각 아래의 작업을 반복한다.
    X_train, X_test, _ = prepare_data(train_email, test_email, mode)  # 모드에 따라서 데이터를 전처리
    score = fit_and_evaluate(X_train, y_train, X_test, y_test)  # 모델을 훈련하고 평가.
    print(mode + " 모드의 테스트 정확도:", score)
