import pandas as pd
import numpy as np
import nltk
import gensim

from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Activation, Input
from tensorflow.keras.layers import Dot
from tensorflow.keras.utils import plot_model
from icecream import ic

# 1. 20뉴스그룹 데이터 전처리하기
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=("headers", "footers", "quotes"))
# print(dataset.DESCR)
print(dataset.target_names)
print(dataset.target.shape)
print(dataset.filenames.shape)

documents = dataset.data
ic("총 샘플 수 :", len(documents))

news_df = pd.DataFrame({"document": documents})
# 특수 문자 제거
news_df["clean_doc"] = news_df["document"].str.replace("[^a-zA-Z]", " ", regex=True)
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df["clean_doc"] = news_df["clean_doc"].apply(
    lambda x: " ".join([w for w in x.split() if len(w) > 3])
)
# 전체 단어에 대한 소문자 변환
news_df["clean_doc"] = news_df["clean_doc"].apply(lambda x: x.lower())
ic(news_df.head())
ic(news_df.isna().values.any())
ic(news_df.isna().any())
ic(news_df.isna().sum())

news_df.replace("", float("NaN"), inplace=True)
ic(news_df.isna().values.any())

news_df.dropna(inplace=True, how="any")
ic(len(news_df))

stop_words = stopwords.words("english")  # NLTK로부터 불용어를 받아옵니다.
tokenized_doc = news_df["clean_doc"].apply(lambda x: x.split())  # 토큰화

# 불용어를 제거합니다.
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = tokenized_doc.to_numpy()
ic(tokenized_doc.shape)

# 단어가 1개 이하인 경우 중심 단어, 주변 단어가 존재하지 않으므로 불가.
drop_train = np.array([index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1])
ic(drop_train.shape)

tokenized_doc = np.delete(tokenized_doc, drop_train, axis=0)
ic(len(tokenized_doc))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_doc)

word2idx = tokenizer.word_index
idx2word = {value: key for key, value in word2idx.items()}
encoded = tokenizer.texts_to_sequences(tokenized_doc)
print(encoded[:5])

vocab_size = len(word2idx) + 1
ic("단어 집합의 크기 :", vocab_size)


# 네거티브 샘플링
skip_grams = [
    skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:10]
]
ic(np.array(skip_grams, dtype=object).shape)

# 첫번째 샘플인 skip_grams[0] 내 skipgrams로 형성된 데이터셋 확인
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(5):
    print(
        "({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
            idx2word[pairs[i][0]], pairs[i][0], idx2word[pairs[i][1]], pairs[i][1], labels[i]
        )
    )
ic(len(skip_grams))

# 첫번째 샘플에 대해서 생긴 pairs와 labels
ic(len(pairs))
ic(len(labels))

s_grams = skipgrams(encoded[0], vocabulary_size=vocab_size, window_size=10)
print(encoded[0])
# print(s_grams[0])
# print(s_grams[1])
print(s_grams[0][0][0])
print(idx2word[s_grams[0][0][0]])
print(idx2word[s_grams[0][0][1]])
print(s_grams[1][0])


skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded]

# 2. Skip-Gram with Negative Sampling(SGNS) 구현하기

embedding_dim = 100

# 중심 단어를 위한 임베딩 테이블
w_inputs = Input(shape=(1,), dtype="int32")
word_embedding = Embedding(vocab_size, embedding_dim)(w_inputs)

# 주변 단어를 위한 임베딩 테이블
c_inputs = Input(shape=(1,), dtype="int32")
context_embedding = Embedding(vocab_size, embedding_dim)(c_inputs)

dot_product = Dot(axes=2)([word_embedding, context_embedding])
dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
output = Activation("sigmoid")(dot_product)

model = Model(inputs=[w_inputs, c_inputs], outputs=output)
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam")
plot_model(
    model, to_file="images/skip_gram.png", show_shapes=True, show_layer_names=True, rankdir="TB"
)

for epoch in range(1, 6):
    loss = 0
    for _, elem in enumerate(skip_grams):
        first_elem = np.array(list(zip(*elem[0]))[0], dtype="int32")
        second_elem = np.array(list(zip(*elem[0]))[1], dtype="int32")
        labels = np.array(elem[1], dtype="int32")
        X = [first_elem, second_elem]
        Y = labels
        loss += model.train_on_batch(X, Y)
    print("Epoch :", epoch, "Loss :", loss)

# 3. 결과 확인하기
f = open("../data/vectors.txt", "w")
f.write("{} {}\n".format(vocab_size - 1, embedding_dim))
vectors = model.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write("{} {}\n".format(word, " ".join(map(str, list(vectors[i, :])))))
f.close()

w2v = gensim.models.KeyedVectors.load_word2vec_format("../data/vectors.txt", binary=False)

ic(w2v.most_similar(positive=["disease"]))
ic(w2v.most_similar(positive=["soldiers"]))
ic(w2v.most_similar(positive=["police"]))
ic(w2v.most_similar(positive=["hero"]))
ic(w2v.most_similar(positive=["engine"]))
ic(w2v.most_similar(positive=["doctor"]))
