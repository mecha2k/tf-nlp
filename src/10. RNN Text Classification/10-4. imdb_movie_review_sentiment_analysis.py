import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data()

print("훈련용 리뷰 개수 : {}".format(len(X_train)))
print("테스트용 리뷰 개수 : {}".format(len(X_test)))
num_classes = len(set(y_train))
print("카테고리 : {}".format(num_classes))

print(X_train[0])
print(y_train[0])

len_result = [len(s) for s in X_train]

print("리뷰의 최대 길이 : {}".format(np.max(len_result)))
print("리뷰의 평균 길이 : {}".format(np.mean(len_result)))

plt.subplot(1, 2, 1)
plt.boxplot(len_result)
plt.subplot(1, 2, 2)
plt.hist(len_result, bins=50)
plt.show()

unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("각 레이블에 대한 빈도수:")
print(np.asarray((unique_elements, counts_elements)))

word_to_index = imdb.get_word_index()
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value + 3] = key

print("빈도수 상위 1등 단어 : {}".format(index_to_word[4]))

print("빈도수 상위 3938등 단어 : {}".format(index_to_word[3941]))

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token

print(" ".join([index_to_word[index] for index in X_train[0]]))

# 2. GRU로 IMDB 리뷰 감성 분류하기

import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GRU(hidden_units))
model.add(Dense(1, activation="sigmoid"))

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
mc = ModelCheckpoint(
    "../data/GRU_model.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True
)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2
)

loaded_model = load_model("../data/GRU_model.h5")
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))


def sentiment_predict(new_sentence):
    # 알파벳과 숫자를 제외하고 모두 제거 및 알파벳 소문자화
    new_sentence = re.sub("[^0-9a-zA-Z ]", "", new_sentence).lower()

    # 정수 인코딩
    encoded = []
    for word in new_sentence.split():
        try:
            # 단어 집합의 크기를 10,000으로 제한.
            if word_to_index[word] <= 10000:
                encoded.append(word_to_index[word] + 3)
            else:
                # 10,000 이상의 숫자는 <unk> 토큰으로 변환.
                encoded.append(2)
        # 단어 집합에 없는 단어는 <unk> 토큰으로 변환.
        except KeyError:
            encoded.append(2)

    pad_sequence = pad_sequences([encoded], maxlen=max_len)  # 패딩
    score = float(loaded_model.predict(pad_sequence))  # 예측

    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))


test_input = (
    "This movie was just way too overrated. The fighting was not professional and in slow motion. "
    "I was expecting more from a 200 million budget movie. The little sister of T.Challa was just trying "
    "too hard to be funny. The story was really dumb as well. Don't watch this movie if you are going "
    "because others say its great unless you are a Black Panther fan or Marvels fan."
)

sentiment_predict(test_input)

test_input = (
    " I was lucky enough to be included in the group to see the advanced screening in Melbourne on the 15th of "
    "April, 2012. And, firstly, I need to say a big thank-you to Disney and Marvel Studios. Now, the film... how can "
    "I even begin to explain how I feel about this film? It is, as the title of this review says a "
    "'comic book triumph'. I went into the film with very, very high expectations and I was not disappointed. Seeing "
    "Joss Whedon's direction and envisioning of the film come to life on the big screen is perfect. The script is "
    "amazingly detailed and laced with sharp wit a humor. The special effects are literally mind-blowing and "
    "the action scenes are both hard-hitting and beautifully choreographed."
)

sentiment_predict(test_input)
