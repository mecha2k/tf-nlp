# 1. 특이값 분해(Singular Value Decomposition, SVD)
# 시작하기 앞서, 여기서의 특이값 분해(Singular Value Decomposition, SVD)는 실수 벡터 공간에 한정하여 내용을 설명함을 명시합니다.
# SVD란 A가 m × n 행렬일 때, 다음과 같이 3개의 행렬의 곱으로 분해(decomposition)하는 것을 말합니다.
# $$A=UΣV^\text{T}$$

# 2. 절단된 SVD(Truncated SVD)
# 위에서 설명한 SVD를 풀 SVD(full SVD)라고 합니다. 하지만 LSA의 경우 풀 SVD에서 나온 3개의 행렬에서 일부 벡터들을 삭제시킨 절단된
# SVD(truncated SVD)를 사용하게 됩니다.

# 3. 잠재 의미 분석(Latent Semantic Analysis, LSA)

import numpy as np

## 3-1. full SVD

A = np.array(
    [
        [0, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 2, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
    ]
)
print("DTM의 크기(shape) :", np.shape(A))

U, s, VT = np.linalg.svd(A, full_matrices=True)
print("행렬 U :")
print(U.round(2))
print("행렬 U의 크기(shape) :", np.shape(U))

print("특이값 벡터 :")
print(s.round(2))
print("특이값 벡터의 크기(shape) :", np.shape(s))

# 대각 행렬의 크기인 4 x 9의 임의의 행렬 생성
S = np.zeros((4, 9))

# 특이값을 대각행렬에 삽입
S[:4, :4] = np.diag(s)

print("대각 행렬 S :")
print(S.round(2))

print("대각 행렬의 크기(shape) :")
print(np.shape(S))

print("직교행렬 VT :")
print(VT.round(2))

print("직교 행렬 VT의 크기(shape) :")
print(np.shape(VT))

A = U @ S @ VT

# same?
np.allclose(A, np.dot(np.dot(U, S), VT).round(2))

## 3-2. 절단된 SVD(Truncated SVD)

# 특이값 사위 2개만 보존
S = S[:2, :2]

print("대각 행렬 S :")
print(S.round(2))

U = U[:, :2]
print("행렬 U :")
print(U.round(2))

VT = VT[:2, :]
print("직교행렬 VT :")
print(VT.round(2))

# 이제 축소된 행렬 U, S, VT에 대해서 다시 U × S × VT연산을 하면 기존의 A와는 다른 결과가 나오게 됩니다. 값이 손실되었기 때문에 이 세
# 개의 행렬로는 이제 기존의 A행렬을 복구할 수 없습니다. U × S × VT연산을 해서 나오는 값을 A_prime이라 하고 기존의 행렬 A와 값을
# 비교해보도록 하겠습니다.

A_prime = np.dot(np.dot(U, S), VT)
print(A)
print(A_prime.round(2))

# 4. 실습을 통한 이해

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

nltk.download("stopwords")

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=("headers", "footers", "quotes"))
documents = dataset.data
print("샘플의 수 :", len(documents))
print(documents[1])
print(dataset.target_names)

news_df = pd.DataFrame({"document": documents})
# 특수 문자 제거
news_df["clean_doc"] = news_df["document"].str.replace("[^a-zA-Z]", " ")
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df["clean_doc"] = news_df["clean_doc"].apply(
    lambda x: " ".join([w for w in x.split() if len(w) > 3])
)
# 전체 단어에 대한 소문자 변환
news_df["clean_doc"] = news_df["clean_doc"].apply(lambda x: x.lower())
print(news_df["clean_doc"][1])

# NLTK로부터 불용어를 받아온다.
stop_words = stopwords.words("english")
tokenized_doc = news_df["clean_doc"].apply(lambda x: x.split())  # 토큰화
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# 역토큰화 (토큰화 작업을 역으로 되돌림)
detokenized_doc = []
for i in range(len(news_df)):
    t = " ".join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df["clean_doc"] = detokenized_doc
print(news_df["clean_doc"][1])

vectorizer = TfidfVectorizer(
    stop_words="english", max_features=1000, max_df=0.5, smooth_idf=True  # 상위 1,000개의 단어를 보존
)

X = vectorizer.fit_transform(news_df["clean_doc"])

# TF-IDF 행렬의 크기 확인
print("TF-IDF 행렬의 크기 :", X.shape)

svd_model = TruncatedSVD(n_components=20, algorithm="randomized", n_iter=100, random_state=122)
svd_model.fit(X)
len(svd_model.components_)

np.shape(svd_model.components_)

# 단어 집합. 1,000개의 단어가 저장됨.
terms = vectorizer.get_feature_names()


def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print(
            "Topic %d:" % (idx + 1),
            [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[: -n - 1 : -1]],
        )


get_topics(svd_model.components_, terms)
