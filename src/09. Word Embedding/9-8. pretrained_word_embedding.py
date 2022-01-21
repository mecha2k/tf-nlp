# 1. 케라스 임베딩 층(Keras Embedding layer)

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "nice great best amazing",
    "stop lies",
    "pitiful nerd",
    "excellent work",
    "supreme quality",
    "bad",
    "highly respectable",
]
y_train = [1, 0, 0, 1, 1, 0, 1]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

X_encoded = tokenizer.texts_to_sequences(sentences)
print(X_encoded)

max_len = max(len(l) for l in X_encoded)
print(max_len)

X_train = pad_sequences(X_encoded, maxlen=max_len, padding="post")
y_train = np.array(y_train)
print(X_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

embedding_dim = 4

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.fit(X_train, y_train, epochs=100, verbose=2)

"""# 2. 사전 훈련된 GloVe 사용하기"""

from urllib.request import urlretrieve, urlopen
import gzip
import zipfile

# urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="../data/glove.6B.zip")
# zf = zipfile.ZipFile("../data/glove.6B.zip")
# zf.extractall()
# zf.close()

embedding_dict = dict()

f = open("../data/glove.6B/glove.6B.100d.txt", encoding="utf8")

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

# 단어 집합 크기의 행과 100개의 열을 가지는 행렬 생성. 값은 전부 0으로 채워진다.
embedding_matrix = np.zeros((vocab_size, 100))
np.shape(embedding_matrix)

print(tokenizer.word_index.items())

print(tokenizer.word_index["great"])

print(embedding_dict["great"])

for word, index in tokenizer.word_index.items():
    # 단어와 맵핑되는 사전 훈련된 임베딩 벡터값
    vector_value = embedding_dict.get(word)
    if vector_value is not None:
        embedding_matrix[index] = vector_value

print(embedding_matrix[2])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False)

model.add(e)
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.fit(X_train, y_train, epochs=100, verbose=2)

"""# 3. 사전 훈련된 Word2Vec 사용하기"""

import gensim

# urlretrieve(
#     "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz",
#     filename="GoogleNews-vectors-negative300.bin.gz",
# )
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    "../data/GoogleNews-vectors-negative300.bin.gz", binary=True
)

print(word2vec_model.vectors.shape)  # 모델의 크기 확인

# 단어 집합 크기의 행과 300개의 열을 가지는 행렬 생성. 값은 전부 0으로 채워진다.
embedding_matrix = np.zeros((vocab_size, 300))
np.shape(embedding_matrix)


def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None


for word, index in tokenizer.word_index.items():
    # 단어와 맵핑되는 사전 훈련된 임베딩 벡터값
    vector_value = get_vector(word)
    if vector_value is not None:
        embedding_matrix[index] = vector_value

print(word2vec_model["nice"])

print(word2vec_model["great"])

print(embedding_matrix[2])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input

model = Sequential()
model.add(Input(shape=(max_len,), dtype="int32"))
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.fit(X_train, y_train, epochs=100, verbose=2)
