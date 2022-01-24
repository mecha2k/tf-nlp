from math import log
import pandas as pd


def tf(t, d):
    return d.count(t)


def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N / (df + 1))


def tfidf(t, d):
    return tf(t, d) * idf(t)


docs = ["먹고 싶은 사과", "먹고 싶은 바나나", "길고 노란 바나나 바나나", "저는 과일이 좋아요"]
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()
print("단어장의 크기 :", len(vocab))
print(vocab)

# 총 문서의 수
N = len(docs)

# 각 문서에 대해서 아래 연산을 반복
result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns=vocab)
print(tf_)

result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))
idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
print(idf_)

result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t, d))

tfidf_ = pd.DataFrame(result, columns=vocab)
print(tfidf_)

# 3. 사이킷런을 이용한 DTM과 TF-IDF 실습
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "you know I want your love",
    "I like you",
    "what should I do ",
]

vector = CountVectorizer()
# 코퍼스로부터 각 단어의 빈도수를 기록
print(vector.fit_transform(corpus).toarray())
# 각 단어와 맵핑된 인덱스 출력
print(vector.vocabulary_)


tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)
