import numpy as np
import gensim

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input
from icecream import ic


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
ic(vocab_size)

X_encoded = tokenizer.texts_to_sequences(sentences)
ic(X_encoded)

max_len = max(len(l) for l in X_encoded)
ic(max_len)

X_train = pad_sequences(X_encoded, maxlen=max_len, padding="post")
y_train = np.array(y_train)
ic(X_train)


embedding_dim = 4

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.fit(X_train, y_train, epochs=100, verbose=0)
model.summary()

# 2. 사전 훈련된 GloVe 사용하기
# from urllib.request import urlretrieve, urlopen
# import gzip
# import zipfile
#
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

ic("%s개의 Embedding vector가 있습니다." % len(embedding_dict))
ic(embedding_dict["respectable"])
ic(len(embedding_dict["respectable"]))

# 단어 집합 크기의 행과 100개의 열을 가지는 행렬 생성. 값은 전부 0으로 채워진다.
embedding_matrix = np.zeros((vocab_size, 100))
ic(embedding_matrix.shape)
ic(tokenizer.word_index.items())
ic(tokenizer.word_index["great"])
ic(embedding_dict["great"])

for word, index in tokenizer.word_index.items():
    # 단어와 맵핑되는 사전 훈련된 임베딩 벡터값
    vector_value = embedding_dict.get(word)
    if vector_value is not None:
        embedding_matrix[index] = vector_value
ic(embedding_matrix[2])


model = Sequential()
model.add(
    Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False)
)
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.fit(X_train, y_train, epochs=100, verbose=0)
model.summary()

# 3. 사전 훈련된 Word2Vec 사용하기
# urlretrieve(
#     "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz",
#     filename="GoogleNews-vectors-negative300.bin.gz",
# )
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    "../data/GoogleNews-vectors-negative300.bin.gz", binary=True
)
ic(word2vec_model.vectors.shape)

# 단어 집합 크기의 행과 300개의 열을 가지는 행렬 생성. 값은 전부 0으로 채워진다.
embedding_matrix = np.zeros((vocab_size, 300))
ic(embedding_matrix.shape)


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
ic(word2vec_model["nice"])
ic(word2vec_model["great"])
ic(embedding_matrix[2])


model = Sequential()
model.add(Input(shape=(max_len,), dtype="int32"))
model.add(
    Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
)
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()
model.fit(X_train, y_train, epochs=100, verbose=0)
