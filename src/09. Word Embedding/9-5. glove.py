import nltk
import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from glove import Corpus, Glove

nltk.download("punkt", quiet=True)

# 데이터 다운로드
# urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification"
#                            "/master/ted_en-20160408.xml", filename="../data/ted_en-20160408.xml")
# targetXML = open("../data/ted_en-20160408.xml", "r", encoding="UTF8")
# target_text = etree.parse(targetXML)
#
# # xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
# parse_text = "\n".join(target_text.xpath("//content/text()"))
#
# # 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# # 해당 코드는 괄호로 구성된 내용을 제거.
# content_text = re.sub(r"\([^)]*\)", "", parse_text)
#
# # 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
# sent_text = sent_tokenize(content_text)
#
# # 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
# normalized_text = []
# for string in sent_text:
#     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
#     normalized_text.append(tokens)
#
# # 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
# result = [word_tokenize(sentence) for sentence in normalized_text]

result = np.load("../data/ted_en_token.npy")
print("총 샘플의 개수 : {}".format(len(result)))

corpus = Corpus()

# 훈련 데이터로부터 GloVe에서 사용할 동시 등장 행렬 생성
corpus.fit(result, window=5)
glove = Glove(no_components=100, learning_rate=0.05)

# 학습에 이용할 쓰레드의 개수는 4로 설정, 에포크는 20.
glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

print(glove.most_similar("man"))
print(glove.most_similar("boy"))
print(glove.most_similar("university"))
print(glove.most_similar("water"))
print(glove.most_similar("physics"))
print(glove.most_similar("muscle"))
print(glove.most_similar("clean"))
