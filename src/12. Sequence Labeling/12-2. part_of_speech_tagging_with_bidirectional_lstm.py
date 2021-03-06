import nltk
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

nltk.download("treebank", quiet=True)

tagged_sentences = nltk.corpus.treebank.tagged_sents()  # 토큰화에 품사 태깅이 된 데이터 받아오기
print("품사 태깅이 된 문장 개수: ", len(tagged_sentences))  # 문장 샘플의 개수 출력
print(tagged_sentences[0])  # 첫번째 문장 샘플 출력

sentences, pos_tags = [], []
for tagged_sentence in tagged_sentences:  # 3,914개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence)  # 각 샘플에서 단어는 sentence에 품사 태깅 정보는 tags에 저장한다.
    sentences.append(list(sentence))  # 각 샘플에서 단어 정보만 저장한다.
    pos_tags.append(list(tag_info))  # 각 샘플에서 품사 태깅 정보만 저장한다.

print(sentences[0])
print(pos_tags[0])

print(sentences[8])
print(pos_tags[8])


def tokenize(samples):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(samples)
    return tokenizer


src_tokenizer = tokenize(sentences)
tar_tokenizer = tokenize(pos_tags)

vocab_size = len(src_tokenizer.word_index) + 1
print("단어 집합의 크기 : {}".format(vocab_size))

tag_size = len(tar_tokenizer.word_index) + 1
print("태깅 정보 집합의 크기 : {}".format(tag_size))

X_train = src_tokenizer.texts_to_sequences(sentences)
y_train = tar_tokenizer.texts_to_sequences(pos_tags)
print(X_train[:2])
print(y_train[:2])

print("샘플의 최대 길이 : %d" % max(len(l) for l in X_train))
print("샘플의 평균 길이 : %f" % (sum(map(len, X_train)) / len(X_train)))
plt.hist(x=[len(s) for s in X_train], bins=50)
plt.xlabel("length of samples")
plt.ylabel("number of samples")
plt.savefig("images/02-01", dpi=300)


max_len = 150
X_train = pad_sequences(X_train, padding="post", maxlen=max_len)
y_train = pad_sequences(y_train, padding="post", maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=777
)

print("훈련 샘플 문장의 크기 : {}".format(X_train.shape))
print("훈련 샘플 레이블의 크기 : {}".format(y_train.shape))
print("테스트 샘플 문장의 크기 : {}".format(X_test.shape))
print("테스트 샘플 레이블의 크기 : {}".format(y_test.shape))

# 2. 양방향 LSTM(Bi-directional LSTM)으로 POS Tagger 만들기
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
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
model.add(TimeDistributed(Dense(tag_size, activation="softmax")))
model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=128, epochs=7, validation_data=(X_test, y_test))
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

index_to_word = src_tokenizer.index_word
index_to_tag = tar_tokenizer.index_word

i = 10  # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]]))  # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1)  # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for word, tag, pred in zip(X_test[i], y_test[i], y_predicted[0]):
    if word != 0:  # PAD값은 제외함.
        print(
            "{:17}: {:7} {}".format(
                index_to_word[word], index_to_tag[tag].upper(), index_to_tag[pred].upper()
            )
        )
