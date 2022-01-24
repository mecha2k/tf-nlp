import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

text = "경마장에 있는 말이 뛰고 있다\n그의 말이 법이다\n가는 말이 고와야 오는 말이 곱다\n"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index) + 1
print("단어 집합의 크기 : %d" % vocab_size)
print(tokenizer.word_index)

sequences = list()
for line in text.split("\n"):  # Wn을 기준으로 문장 토큰화
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[: i + 1]
        sequences.append(sequence)

print("학습에 사용할 샘플의 개수: %d" % len(sequences))
print(sequences)

max_len = max(len(l) for l in sequences)  # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력
print("샘플의 최대 길이 : {}".format(max_len))

sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")
print(sequences)

sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]
print(X)
print(y)  # 모든 샘플에 대한 레이블 출력

y = to_categorical(y, num_classes=vocab_size)
print(y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN

embedding_dim = 10
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(vocab_size, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=200, verbose=0)


def sentence_generation(model, tokenizer, current_word, n):  # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word
    sentence = ""

    # n번 반복
    for _ in range(n):
        # 현재 단어에 대한 정수 인코딩과 패딩
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=5, padding="pre")
        # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        word = None
        for word, index in tokenizer.word_index.items():
            # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면 break
            if index == result:
                break
        # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        current_word = current_word + " " + word
        # 예측 단어를 문장에 저장
        sentence = sentence + " " + word

    sentence = init_word + sentence
    return sentence


print(sentence_generation(model, tokenizer, "경마장에", 4))
print(sentence_generation(model, tokenizer, "그의", 2))
print(sentence_generation(model, tokenizer, "가는", 5))

# 2. LSTM을 이용하여 텍스트 생성하기

import pandas as pd
import numpy as np
from string import punctuation

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("../data/ArticlesApril2018.csv")
print(df.head())
print("열의 개수: ", len(df.columns))
print(df.columns)
print(df["headline"].isna().values.any())

headline = []
# 헤드라인의 값들을 리스트로 저장
headline.extend(list(df.headline.values))
print(headline[:5])
print("총 샘플의 개수 : {}".format(len(headline)))  # 현재 샘플의 개수

# Unknown 값을 가진 샘플 제거
headline = [word for word in headline if word != "Unknown"]
print("노이즈값 제거 후 샘플의 개수 : {}".format(len(headline)))


def repreprocessing(raw_sentence):
    preproceseed_sentence = raw_sentence.encode("utf8").decode("ascii", "ignore")
    # 구두점 제거와 동시에 소문자화
    return "".join(word for word in preproceseed_sentence if word not in punctuation).lower()


preporcessed_headline = [repreprocessing(x) for x in headline]
print(preporcessed_headline[:5])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preporcessed_headline)
vocab_size = len(tokenizer.word_index) + 1
print("단어 집합의 크기 : %d" % vocab_size)

sequences = list()
for sentence in preporcessed_headline:
    # 각 샘플에 대한 정수 인코딩
    encoded = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[: i + 1]
        sequences.append(sequence)
print(sequences[:11])

index_to_word = {}
for key, value in tokenizer.word_index.items():  # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    index_to_word[value] = key
print("빈도수 상위 582번 단어 : {}".format(index_to_word[582]))

max_len = max(len(l) for l in sequences)
print("샘플의 최대 길이 : {}".format(max_len))

sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")
print(sequences[:3])

sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]
print(X[:3])
print(y[:3])

y = to_categorical(y, num_classes=vocab_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

embedding_dim = 10
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(vocab_size, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=200, verbose=0)


def sentence_generation(model, tokenizer, current_word, n):  # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word
    sentence = ""

    # n번 반복
    for _ in range(n):
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=max_len - 1, padding="pre")

        # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        word = None
        for word, index in tokenizer.word_index.items():
            # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
            if index == result:
                break
        # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        current_word = current_word + " " + word
        # 예측 단어를 문장에 저장
        sentence = sentence + " " + word

    sentence = init_word + sentence
    return sentence


print(sentence_generation(model, tokenizer, "i", 10))
print(sentence_generation(model, tokenizer, "how", 10))
