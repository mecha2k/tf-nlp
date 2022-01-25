import gensim
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from PIL import Image
from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def remove_html(text):
    html_pattern = re.compile("<.*?>")
    return html_pattern.sub(r"", text)


def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text


nltk.download("stopwords", quiet=True)

# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/data.csv",
#     filename="../data/data.csv",
# )

df = pd.read_csv("../data/data.csv")
print("전체 문서의 수 :", len(df))
print(df[:5])

df["cleaned"] = df["Desc"].apply(lambda s: "".join(i for i in s if ord(i) < 128))
df["cleaned"] = df.cleaned.apply(lambda x: x.lower())
df["cleaned"] = df.cleaned.apply(remove_stop_words)
df["cleaned"] = df.cleaned.apply(remove_punctuation)
df["cleaned"] = df.cleaned.apply(remove_html)
print(df["cleaned"][:5])

df["cleaned"].replace("", np.nan, inplace=True)
df = df.loc[df["cleaned"].notna()]
print("전체 문서의 수 :", len(df))

corpus = []
for words in df["cleaned"]:
    corpus.append(words.split())
print(corpus[0])

# 2. 사전 훈련된 워드 임베딩 사용하기
# urllib.request.urlretrieve(
#     "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz",
#     filename="GoogleNews-vectors-negative300.bin.gz",
# )

w2v_model = Word2Vec(vector_size=300, window=5, min_count=2, workers=-1)
w2v_model.build_vocab(corpus)
w2v_model.wv.vectors_lockf = np.ones(len(w2v_model.wv), dtype=float)
w2v_model.wv.intersect_word2vec_format(
    "../data/GoogleNews-vectors-negative300.bin.gz", lockf=1.0, binary=True
)
w2v_model.train(corpus, total_examples=w2v_model.corpus_count, epochs=15)

# 3. 단어 벡터의 평균 구하기
def get_document_vectors(document_list):
    document_embedding_list = []
    # 각 문서에 대해서
    for line in document_list:
        doc2vec = None
        count = 0
        for word in line.split():
            if word in w2v_model.wv.key_to_index:
                count += 1
                # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is None:
                    doc2vec = w2v_model.wv.get_vector(word)
                else:
                    doc2vec = doc2vec + w2v_model.wv.get_vector(word)
        if doc2vec is not None:
            # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠준다.
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)
    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return document_embedding_list


document_embedding_list = get_document_vectors(df["cleaned"])
print("문서 벡터의 수 :", len(document_embedding_list))

# 4. 추천 시스템 구현하기
cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)
print(cosine_similarities[0])
print("코사인 유사도 매트릭스의 크기 :", cosine_similarities.shape)


def recommendations(title, filename):
    books = df[["title", "image_link"]]

    # 책의 제목을 입력하면 해당 제목의 인덱스를 리턴받아 idx에 저장.
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    idx = indices[title]

    # 입력된 책과 줄거리(document embedding)가 유사한 책 5개 선정.
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    # 가장 유사한 책 5권의 인덱스
    book_indices = [i[0] for i in sim_scores]

    # 전체 데이터프레임에서 해당 인덱스의 행만 추출. 5개의 행을 가진다.
    recommend = books.iloc[book_indices].reset_index(drop=True)
    print(recommend["title"])

    fig = plt.figure(figsize=(20, 12))
    # 데이터프레임으로부터 순차적으로 이미지를 출력
    for index, row in recommend.iterrows():
        response = requests.get(row["image_link"])
        img = Image.open(BytesIO(response.content))
        fig.add_subplot(1, 5, index + 1)
        plt.imshow(img)
        plt.title(row["title"])
    plt.savefig(filename, dpi=300)


recommendations("The Da Vinci Code", "images/davinci")
recommendations("The Murder of Roger Ackroyd", "images/roger")
