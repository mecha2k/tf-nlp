import numpy as np
import gensim
from urllib.request import urlretrieve, urlopen
import gzip
import zipfile

# urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")
# zf = zipfile.ZipFile('glove.6B.zip')
# zf.extractall()
# zf.close()

glove_dict = dict()
f = open("../data/glove.6B/glove.6B.100d.txt", encoding="utf8")  # 100차원의 GloVe 벡터를 사용

for line in f:
    word_vector = line.split()
    word = word_vector[0]
    word_vector_arr = np.asarray(word_vector[1:], dtype="float32")  # 100개의 값을 가지는 array로 변환
    glove_dict[word] = word_vector_arr
f.close()

print(glove_dict["cat"])

embedding_dim = 100
zero_vector = np.zeros(embedding_dim)

# 단어 벡터의 평균으로부터 문장 벡터를 얻는다.
def calculate_sentence_vector(sentence):
    return sum([glove_dict.get(word, zero_vector) for word in sentence]) / len(sentence)


eng_sent = ["I", "am", "a", "student"]
sentence_vector = calculate_sentence_vector(eng_sent)
print(len(sentence_vector))

kor_sent = ["전", "좋은", "학생", "입니다"]
sentence_vector = calculate_sentence_vector(kor_sent)
print(sentence_vector)

import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from urllib.request import urlretrieve
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk

nltk.download("stopwords")
nltk.download("punkt")

stop_words = stopwords.words("english")

# urlretrieve(
#     "https://raw.githubusercontent.com/prateekjoshi565/textrank_text_summarization/master/tennis_articles_v4.csv",
#     filename="../data/tennis_articles_v4.csv",
# )
data = pd.read_csv("../data/tennis_articles_v4.csv")
data.head()

data = data[["article_text"]]
data["sentences"] = data["article_text"].apply(sent_tokenize)
print(data)

# 토큰화 함수
def tokenization(sentences):
    return [word_tokenize(sentence) for sentence in sentences]


# 전처리 함수
def preprocess_sentence(sentence):
    # 영어를 제외한 숫자, 특수 문자 등은 전부 제거. 모든 알파벳은 소문자화
    sentence = [re.sub(r"[^a-zA-z\s]", "", word).lower() for word in sentence]
    # 불용어가 아니면서 단어가 실제로 존재해야 한다.
    return [word for word in sentence if word not in stop_words and word]


# 위 전처리 함수를 모든 문장에 대해서 수행. 이 함수를 호출하면 모든 행에 대해서 수행.
def preprocess_sentences(sentences):
    return [preprocess_sentence(sentence) for sentence in sentences]


data["tokenized_sentences"] = data["sentences"].apply(tokenization)
data["tokenized_sentences"] = data["tokenized_sentences"].apply(preprocess_sentences)
print(data)

embedding_dim = 100
zero_vector = np.zeros(embedding_dim)

# 단어 벡터의 평균으로부터 문장 벡터를 얻는다.
def calculate_sentence_vector(sentence):
    if len(sentence) != 0:
        return sum([glove_dict.get(word, zero_vector) for word in sentence]) / len(sentence)
    else:
        return zero_vector


# 각 문장에 대해서 문장 벡터를 반환
def sentences_to_vectors(sentences):
    return [calculate_sentence_vector(sentence) for sentence in sentences]


data["SentenceEmbedding"] = data["tokenized_sentences"].apply(sentences_to_vectors)
print(data[["SentenceEmbedding"]])


def similarity_matrix(sentence_embedding):
    sim_mat = np.zeros([len(sentence_embedding), len(sentence_embedding)])
    for i in range(len(sentence_embedding)):
        for j in range(len(sentence_embedding)):
            sim_mat[i][j] = cosine_similarity(
                sentence_embedding[i].reshape(1, embedding_dim),
                sentence_embedding[j].reshape(1, embedding_dim),
            )[0, 0]
    return sim_mat


data["SimMatrix"] = data["SentenceEmbedding"].apply(similarity_matrix)
print(data["SimMatrix"])

print("두번째 샘플의 문장 개수 :", len(data["tokenized_sentences"][1]))
print("두번째 샘플의 문장 벡터가 모인 문장 행렬의 크기(shape) :", np.shape(data["SentenceEmbedding"][1]))
print("두번째 샘플의 유사도 행렬의 크기(shape) :", data["SimMatrix"][1].shape)


def draw_graphs(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, with_labels=True, font_weight="bold")
    nx.draw_networkx_edge_labels(nx_graph, pos, font_color="red")
    plt.show()


draw_graphs(data["SimMatrix"][1])


def calculate_score(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    return scores


data["score"] = data["SimMatrix"].apply(calculate_score)
print(data[["SimMatrix", "score"]])
print(data["score"][1])


def ranked_sentences(sentences, scores, n=3):
    top_scores = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_n_sentences = [sentence for score, sentence in top_scores[:n]]
    return " ".join(top_n_sentences)


data["summary"] = data.apply(lambda x: ranked_sentences(x.sentences, x.score), axis=1)

for i in range(0, len(data)):
    print(i + 1, "번 문서")
    print("원문 :", data.loc[i].article_text)
    print("")
    print("요약 :", data.loc[i].summary)
    print("")
