import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

# pip install sentence_transformers

# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
#     filename="../data/ChatBotData.csv",
# )

train_data = pd.read_csv("../data/ChatBotData.csv")
train_data.head()

model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")

train_data["embedding"] = train_data.apply(lambda row: model.encode(row.Q), axis=1)
print(train_data)


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def return_similar_answer(input_):
    embedding = model.encode(input_)
    train_data["score"] = train_data.apply(lambda x: cos_sim(x["embedding"], embedding), axis=1)
    return train_data.loc[train_data["score"].idxmax()]["A"]


print(return_similar_answer("결혼하고싶어"))
print(return_similar_answer("나랑 커피먹을래?"))
print(return_similar_answer("반가워"))
print(return_similar_answer("사랑해"))
print(return_similar_answer("너는 누구니?"))
print(return_similar_answer("영화"))
print(return_similar_answer("너무 짜증나"))
print(return_similar_answer("화가납니다"))
print(return_similar_answer("나랑 놀자"))
print(return_similar_answer("나랑 게임하자"))
print(return_similar_answer("출근하기싫어"))
print(return_similar_answer("여행가고싶다"))
print(return_similar_answer("너 말 잘한다"))

# 좋은 사람이랑 결혼할 수 있을 거예요.
# 카페인이 필요한 시간인가 봐요.
# 저도 반가워요.
# 상대방에게 전해보세요.
# 저는 위로봇입니다.
# 저도 영화 보여주세요.
# 짜증날 땐 짜장면
# 화를 참는 연습을 해보세요.
# 지금 그러고 있어요.
# 같이 놀아요.
# 씻고 푹 쉬세요.
# 이김에 떠나보세요.
# 그런 사람이 있으면 저 좀 소개시켜주세요.
