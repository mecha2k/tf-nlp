from numpy import dot
from numpy.linalg import norm
import numpy as np


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


doc1 = np.array([0, 1, 1, 1])
doc2 = np.array([1, 0, 1, 1])
doc3 = np.array([2, 0, 2, 2])

print("문서 1과 문서2의 유사도 :", cos_sim(doc1, doc2))
print("문서 1과 문서3의 유사도 :", cos_sim(doc1, doc3))
print("문서 2와 문서3의 유사도 :", cos_sim(doc2, doc3))

# 2. 유사도를 이용한 추천 시스템 구현하기

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("../data/movies_metadata.csv", low_memory=False)
print(data.head(2))

data = data.head(20000)
print("overview 열의 결측값의 수:", data["overview"].isnull().sum())

data["overview"] = data["overview"].fillna("")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["overview"])
print("TF-IDF 행렬의 크기(shape) :", tfidf_matrix.shape)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("코사인 유사도 연산 결과 :", cosine_sim.shape)

title_to_index = dict(zip(data["title"], data.index))

idx = title_to_index["Father of the Bride Part II"]
print(idx)


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [idx[0] for idx in sim_scores]
    return data["title"].iloc[movie_indices]


get_recommendations("The Dark Knight Rises")
