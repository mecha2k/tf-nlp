# 1. Colab에 Mecab 설치

# 아래의 Mecab 설치는 Colab에서 실행한다고 가정하고 작성되었습니다.
# 다른 환경이라면 별도의 Mecab 설치 과정을 거치거나 Okt 등과 같은 다른 형태소 분석기를 사용해주세요.


# Commented out IPython magic to ensure Python compatibility.
# Colab에 Mecab 설치
# !git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab
# !bash install_mecab-ko_on_colab190912.sh

# 2. 네이버 쇼핑 리뷰 데이터에 대한 이해와 전처리

from konlpy.tag import Mecab

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt",
#     filename="ratings_total.txt",
# )

total_data = pd.read_table("../data/ratings_total.txt", names=["ratings", "reviews"])

print("전체 리뷰 개수 :", len(total_data))  # 전체 리뷰 개수 출력
print(total_data[:5])

total_data["label"] = np.select([total_data.ratings > 3], [1], default=0)
print(total_data[:5])
print(
    total_data["ratings"].nunique(), total_data["reviews"].nunique(), total_data["label"].nunique()
)

total_data.drop_duplicates(subset=["reviews"], inplace=True)  # reviews 열에서 중복인 내용이 있다면 중복 제거
print("총 샘플의 수 :", len(total_data))
print(total_data.isnull().values.any())

train_data, test_data = train_test_split(total_data, test_size=0.25, random_state=42)
print("훈련용 리뷰의 개수 :", len(train_data))
print("테스트용 리뷰의 개수 :", len(test_data))

train_data["label"].value_counts().plot(kind="bar")
print(train_data.groupby("label").size().reset_index(name="count"))

train_data["reviews"] = train_data["reviews"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
# 한글과 공백을 제외하고 모두 제거
print(train_data[:5])

train_data["reviews"].replace("", np.nan, inplace=True)
print(train_data.isnull().sum())

test_data.drop_duplicates(subset=["reviews"], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
test_data["reviews"] = test_data["reviews"].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True
)  # 정규 표현식 수행
test_data["reviews"].replace("", np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how="any")  # Null 값 제거
print("전처리 후 테스트용 샘플의 개수 :", len(test_data))

import platform

osname = platform.system()
if osname == "Windows":
    mecab = Mecab(dicpath="C:/mecab/mecab-ko-dic")
else:
    mecab = Mecab()

print(mecab.morphs("와 이런 것도 상품이라고 차라리 내가 만드는 게 나을 뻔"))

stopwords = [
    "도",
    "는",
    "다",
    "의",
    "가",
    "이",
    "은",
    "한",
    "에",
    "하",
    "고",
    "을",
    "를",
    "인",
    "듯",
    "과",
    "와",
    "네",
    "들",
    "듯",
    "지",
    "임",
    "게",
]

train_data["tokenized"] = train_data["reviews"].apply(mecab.morphs)
train_data["tokenized"] = train_data["tokenized"].apply(
    lambda x: [item for item in x if item not in stopwords]
)

from collections import Counter

print(train_data[train_data.label == 0]["tokenized"].values)

negative_words = np.hstack(train_data[train_data.label == 0]["tokenized"].values)
positive_words = np.hstack(train_data[train_data.label == 1]["tokenized"].values)

negative_word_count = Counter(negative_words)  # 파이썬의 Counter 모듈을 이용하면 단어의 모든 빈도를 쉽게 계산할 수 있습니다.
print(negative_word_count)

negative_word_count.most_common(20)

print(negative_word_count.most_common(20))

positive_word_count = Counter(positive_words)  # 파이썬의 Counter 모듈을 이용하면 단어의 모든 빈도를 쉽게 계산할 수 있습니다.
print(positive_word_count)

positive_word_count.most_common(20)

print(positive_word_count.most_common(20))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
text_len = train_data[train_data["label"] == 1]["tokenized"].map(lambda x: len(x))
ax1.hist(text_len, color="red")
ax1.set_title("Positive Reviews")
ax1.set_xlabel("length of samples")
ax1.set_ylabel("number of samples")
print("긍정 리뷰의 평균 길이 :", np.mean(text_len))

text_len = train_data[train_data["label"] == 0]["tokenized"].map(lambda x: len(x))
ax2.hist(text_len, color="blue")
ax2.set_title("Negative Reviews")
fig.suptitle("Words in texts")
ax2.set_xlabel("length of samples")
ax2.set_ylabel("number of samples")
print("부정 리뷰의 평균 길이 :", np.mean(text_len))
plt.show()

test_data["tokenized"] = test_data["reviews"].apply(mecab.morphs)
test_data["tokenized"] = test_data["tokenized"].apply(
    lambda x: [item for item in x if item not in stopwords]
)

X_train = train_data["tokenized"].values
y_train = train_data["label"].values
X_test = test_data["tokenized"].values
y_test = test_data["label"].values

print(X_train[:3])
print(y_train[:3])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

print(tokenizer.word_index)

threshold = 2
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if value < threshold:
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print("단어 집합(vocabulary)의 크기 :", total_cnt)
print("등장 빈도가 %s번 이하인 희귀 단어의 수: %s" % (threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
print("단어 집합의 크기 :", vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token="OOV")
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print(tokenizer.word_index)

print(X_train[:3])

print(X_test[:3])

print("리뷰의 최대 길이 :", max(len(l) for l in X_train))
print("리뷰의 평균 길이 :", sum(map(len, X_train)) / len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel("length of samples")
plt.ylabel("number of samples")
plt.show()


def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if len(s) <= max_len:
            cnt = cnt + 1
    print("전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s" % (max_len, (cnt / len(nested_list)) * 100))


max_len = 80
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

print(X_train.shape)
print(X_train[:3])

# 3. GRU를 이용한 분류

from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(GRU(128))
model.add(Dense(1, activation="sigmoid"))

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
mc = ModelCheckpoint(
    "../data/best_model.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True
)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2
)

loaded_model = load_model("../data/best_model.h5")
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

# 4. 리뷰 예측해보기


def sentiment_predict(new_sentence):
    new_sentence = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", new_sentence)
    new_sentence = mecab.morphs(new_sentence)  # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩

    score = float(loaded_model.predict(pad_new))  # 예측
    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))


sentiment_predict("이 상품 진짜 좋아요... 저는 강추합니다. 대박")

sentiment_predict("진짜 배송도 늦고 개짜증나네요. 뭐 이런 걸 상품이라고 만듬?")

sentiment_predict("판매자님... 너무 짱이에요.. 대박나삼")

sentiment_predict("ㅁㄴㅇㄻㄴㅇㄻㄴㅇ리뷰쓰기도 귀찮아")

epochs = range(1, len(history.history["acc"]) + 1)
plt.plot(epochs, history.history["loss"])
plt.plot(epochs, history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

epochs = range(1, len(history.history["acc"]) + 1)
plt.plot(epochs, history.history["acc"])
plt.plot(epochs, history.history["val_acc"])
plt.title("model acc")
plt.ylabel("acc")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
