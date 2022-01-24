import gensim
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from konlpy.tag import Okt
import nltk
import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from icecream import ic
from gensim.utils import effective_n_jobs

nltk.download("punkt", quiet=True)
print(gensim.__version__)
print("CPU threads : ", effective_n_jobs(-1))

# 1. 영어 Word2Vec
# 데이터 다운로드
# url = (
#     "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset"
#     "/ted_en-20160408.xml"
# )
# urllib.request.urlretrieve(
#     url,
#     filename="../data/ted_en-20160408.xml",
# )
#
# targetXML = open("../data/ted_en-20160408.xml", "r", encoding="UTF8")
# target_text = etree.parse(targetXML)
#
# # xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
# parse_text = "\n".join(target_text.xpath("//content/text()"))
# ic(parse_text[0])
#
# # 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# # 해당 코드는 괄호로 구성된 내용을 제거.
# content_text = re.sub(r"\([^)]*\)", "", parse_text)
# ic(content_text[0])
#
# # 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
# sent_text = sent_tokenize(content_text)
# ic(sent_text[0])
#
# # 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
# normalized_text = []
# for string in sent_text:
#     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
#     normalized_text.append(tokens)
#
# # 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
# result = [word_tokenize(sentence) for sentence in normalized_text]
# result = np.array(result)
# np.save("../data/ted_en_token", result)

# result = np.load("../data/ted_en_token.npy", allow_pickle=True)
# print("총 샘플의 개수 : {}".format(len(result)))
#
# # 샘플 3개만 출력
# for line in result[:3]:
#     print(line)
#
# model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=32, sg=0)
# # 여기서 Word2Vec의 인자는 다음과 같습니다.
# # * size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
# # * window = 컨텍스트 윈도우 크기
# # * min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
# # * workers = 학습을 위한 프로세스 수
# # * sg = 0은 CBOW, 1은 Skip-gram.
#
# # model_result = model.wv.most_similar("man")
# # print(model_result)
#
# model.wv.save_word2vec_format("../data/eng_w2v")
# loaded_model = KeyedVectors.load_word2vec_format("../data/eng_w2v")
#
# model_result = loaded_model.most_similar("man")
# print(model_result)

# 2. 한국어 Word2Vec
# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt"
# )

# train_data = pd.read_table("../data/ratings.txt")
#
# print(train_data[:5])  # 상위 5개 출력
# print("리뷰 개수 :", len(train_data))
# print("NULL 값 존재 유무 :", train_data.isna().values.any())
#
# train_data = train_data.dropna(how="any")  # Null 값이 존재하는 행 제거
# print("NULL 값 존재 유무 :", train_data.isna().values.any())  # Null 값이 존재하는지 확인
# print("리뷰 개수 :", len(train_data))
#
# # 정규 표현식을 통한 한글 외 문자 제거
# train_data["document"] = train_data["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
# print(train_data[:5])  # 상위 5개 출력
#
# # 불용어 정의
# stopwords = ["의", "가", "이", "은", "들", "는", "좀", "잘"]
# stopwords += ["걍", "과", "도", "를", "으로", "자", "에", "와", "한", "하다"]
# print(stopwords)
#
# # 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
# okt = Okt()
#
# tokenized_data = []
# for sentence in tqdm(train_data["document"]):
#     tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
#     stopwords_removed_sentence = [
#         word for word in tokenized_sentence if not word in stopwords
#     ]  # 불용어 제거
#     tokenized_data.append(stopwords_removed_sentence)
# tokenized_data = np.array(tokenized_data)
# np.save("../data/naver_review_token", tokenized_data)

# tokenized_data = np.load("../data/naver_review_token.npy", allow_pickle=True)
# print(tokenized_data[:3])
#
# # 리뷰 길이 분포 확인
# print("리뷰의 최대 길이 :", max(len(l) for l in tokenized_data))
# print("리뷰의 평균 길이 :", sum(map(len, tokenized_data)) / len(tokenized_data))
# plt.hist([len(s) for s in tokenized_data], bins=50)
# plt.xlabel("length of samples")
# plt.ylabel("number of samples")
# plt.savefig("images/03-01", dpi=300)
#
# # model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=5, workers=32, sg=0)
# # model.wv.save_word2vec_format("../data/kor_w2v")
#
# model = KeyedVectors.load_word2vec_format("../data/kor_w2v")
#
# print("완성된 임베딩 매트릭스의 크기 확인 :", model.vectors.shape)
# print(model.most_similar("최민식"))
# print(model.most_similar("히어로"))
# print(model.most_similar("발연기"))

# 3. 사전 훈련된 Word2Vec
# urllib.request.urlretrieve(
#     "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz",
#     filename="../data/GoogleNews-vectors-negative300.bin.gz",
# )
word2vec_model = KeyedVectors.load_word2vec_format(
    "../data/GoogleNews-vectors-negative300.bin.gz", binary=True
)

print(word2vec_model.vectors.shape)
print(word2vec_model.similarity("this", "is"))
print(word2vec_model.similarity("post", "book"))
print(word2vec_model["book"])
