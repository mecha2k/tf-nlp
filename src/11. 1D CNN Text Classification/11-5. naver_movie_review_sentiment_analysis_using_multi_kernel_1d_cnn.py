import tensorflow as tf
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
#     filename="../data/ratings_train.txt",
# )
# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
#     filename="../data/ratings_test.txt",
# )

train_data = pd.read_table("../data/ratings_train.txt")
test_data = pd.read_table("../data/ratings_test.txt")

print("훈련 샘플의 개수 :", len(train_data))  # 훈련용 리뷰 개수 출력
print("테스트 샘플의 개수 :", len(test_data))  # 테스트용 리뷰 개수 출력
print(train_data[:5])  # 상위 5개 출력
print(train_data["document"].nunique(), train_data["label"].nunique())

train_data.drop_duplicates(subset=["document"], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
print("총 샘플의 수 :", len(train_data))

train_data["label"].value_counts().plot(kind="bar")
print(train_data.groupby("label").size().reset_index(name="count"))
print(train_data.isnull().values.any())
print(train_data.isnull().sum())
print(train_data.loc[train_data.document.isnull()])

train_data = train_data.dropna(how="any")  # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any())  # Null 값이 존재하는지 확인
print(len(train_data))

train_data["document"] = train_data["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
# 한글과 공백을 제외하고 모두 제거
print(train_data[:5])

# white space 데이터를 empty value로 변경
train_data["document"] = train_data["document"].str.replace("^ +", "", regex=True)
train_data["document"].replace("", np.nan, inplace=True)
print(train_data.isnull().sum())
print(train_data.loc[train_data.document.isnull()][:5])

train_data = train_data.dropna(how="any")
print(len(train_data))

test_data.drop_duplicates(subset=["document"], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
test_data["document"] = test_data["document"].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True
)  # 정규 표현식 수행
test_data["document"] = test_data["document"].str.replace("^ +", "", regex=True)  # 공백은 empty 값으로 변경
test_data["document"].replace("", np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how="any")  # Null 값 제거
print("전처리 후 테스트용 샘플의 개수 :", len(test_data))
print("전처리 후 테스트용 샘플의 개수 :", len(test_data))

stopwords = [
    "의",
    "가",
    "이",
    "은",
    "들",
    "는",
    "좀",
    "잘",
    "걍",
    "과",
    "도",
    "를",
    "으로",
    "자",
    "에",
    "와",
    "한",
    "하다",
]

okt = Okt()

X_train = []
for sentence in tqdm(train_data["document"]):
    tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
    stopwords_removed_sentence = [
        word for word in tokenized_sentence if not word in stopwords
    ]  # 불용어 제거
    X_train.append(stopwords_removed_sentence)

X_test = []
for sentence in tqdm(test_data["document"]):
    tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
    stopwords_removed_sentence = [
        word for word in tokenized_sentence if not word in stopwords
    ]  # 불용어 제거
    X_test.append(stopwords_removed_sentence)
print("전처리 후 테스트용 샘플의 개수 :", len(X_test))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index)
print(tokenizer.word_counts.items())

threshold = 3
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

vocab_size = total_cnt - rare_cnt + 1
print("단어 집합의 크기 :", vocab_size)

tokenizer = Tokenizer(vocab_size)  # 빈도수 2 이하인 단어는 제거
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(train_data["label"])
y_test = np.array(test_data["label"])

print(len(X_train))
print(len(y_train))
print(X_train[:3])

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
drop_test = [index for index, sentence in enumerate(X_test) if len(sentence) < 1]
print(drop_train)
print(len(drop_train))

X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)

print(len(X_train))
print(len(y_train))

print(len(X_test))
print(len(y_test))

X_test = np.delete(X_test, drop_test, axis=0)
y_test = np.delete(y_test, drop_test, axis=0)

print(len(X_test))
print(len(y_test))

print("리뷰의 최대 길이 :", max(len(l) for l in X_train))
print("리뷰의 평균 길이 :", sum(map(len, X_train)) / len(X_train))
plt.hist(x=[len(s) for s in X_train], bins=50)
plt.xlabel("length of samples")
plt.ylabel("number of samples")
plt.savefig("images/05-01", dpi=300)


def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if len(s) <= max_len:
            cnt = cnt + 1
    print("전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s" % (max_len, (cnt / len(nested_list)) * 100))


max_len = 30
below_threshold_len(max_len, X_train)

# 전체 데이터의 길이는 30으로 맞춘다.
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


from tensorflow.keras.models import Sequential, Model
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

embedding_dim = 128
dropout_ratio = (0.5, 0.8)
num_filters = 128
hidden_units = 128

model_input = Input(shape=(max_len,))
z = Embedding(vocab_size, embedding_dim, input_length=max_len, name="embedding")(model_input)
z = Dropout(dropout_ratio[0])(z)

conv_blocks = []

for sz in [3, 4, 5]:
    conv = Conv1D(
        filters=num_filters, kernel_size=sz, padding="valid", activation="relu", strides=1
    )(z)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)

z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_ratio[1])(z)
z = Dense(hidden_units, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
mc = ModelCheckpoint(
    "../data/11_CNN_model.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True
)

model.fit(
    X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=2, callbacks=[es, mc]
)

loaded_model = load_model("../data/11_CNN_model.h5")
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))


def sentiment_predict(new_sentence):
    new_sentence = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True)  # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
    score = float(loaded_model.predict(pad_new))  # 예측
    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))


sentiment_predict("이 영화 개꿀잼 ㅋㅋㅋ")
sentiment_predict("이 영화 핵노잼 ㅠㅠ")
sentiment_predict("이딴게 영화냐 ㅉㅉ")
sentiment_predict("감독 뭐하는 놈이냐?")
sentiment_predict("와 개쩐다 정말 세계관 최강자들의 영화다")
