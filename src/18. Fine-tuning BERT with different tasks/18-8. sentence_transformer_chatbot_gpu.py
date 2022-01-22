# pip install sentence_transformers

import urllib.request
import pandas as pd

from sentence_transformers import SentenceTransformer

# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
#     filename="../data/ChatBotData.csv",
# )
train_data = pd.read_csv("../data/ChatBotData.csv")
train_data.head()

model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")

train_data["embedding"] = train_data.apply(lambda row: model.encode(row.Q), axis=1)
print(train_data)

import numpy as np
from numpy import dot
from numpy.linalg import norm


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def return_similar_answer(input_):
    embedding = model.encode(input_)
    train_data["score"] = train_data.apply(lambda x: cos_sim(x["embedding"], embedding), axis=1)
    return train_data.loc[train_data["score"].idxmax()]["A"]


return_similar_answer("결혼하고싶어")
return_similar_answer("나랑 커피먹을래?")
return_similar_answer("반가워")
return_similar_answer("사랑해")
return_similar_answer("너는 누구니?")
return_similar_answer("영화")
return_similar_answer("너무 짜증나")
return_similar_answer("화가납니다")
return_similar_answer("나랑 놀자")
return_similar_answer("나랑 게임하자")
return_similar_answer("출근하기싫어")
return_similar_answer("여행가고싶다")
return_similar_answer("너 말 잘한다")
