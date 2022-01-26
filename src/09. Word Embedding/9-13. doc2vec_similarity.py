import numpy as np
import pandas as pd
import gensim
from konlpy.tag import Mecab
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import platform

osname = platform.system()
if osname == "Windows":
    mecab = Mecab(dicpath="C:/mecab/mecab-ko-dic")
else:
    mecab = Mecab()

print(gensim.__version__)
print(mecab.pos("아버지가방에들어가신다"))

df = pd.read_csv("../data/dart.csv", sep=",")
df = df.dropna()
print(df)

df = df[:100]

tagged_corpus_list = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    text = row["business"]
    tag = row["name"]
    tagged_corpus_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text)))
print("문서의 수 :", len(tagged_corpus_list))
print(tagged_corpus_list[0])

# 아래 코드는 굉장히 오래 걸립니다. 1시간 이상 걸릴 수도 있습니다.
model = Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, workers=-1, window=8)
# Vocabulary 빌드
model.build_vocab(tagged_corpus_list)
print(f"Tag Size: {len(model.dv.index_to_key)}")
# Doc2Vec 학습
model.train(tagged_corpus_list, total_examples=model.corpus_count, epochs=50)
# 모델 저장
model.save("../data/dart1.doc2vec")

# 코드를 다 수행하고나면 3개의 파일이 생깁니다.
# * dart.doc2vec
# * dart.doc2vec.trainables.syn1neg.npy
# * dart.doc2vec.wv.vectors.npy
#
# 여러분들이 1시간 이상 기다릴 수는 없으므로 제가 이미 만들어놓은 파일들을 다운로드하여 실습을 진행합니다.
#
# dart.doc2vec 파일 다운로드
# !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet
# --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate
# 'https://docs.google.com/uc?export=download&id=1HyGiCxE748kt3dAhHTHxHHHMjj0VcpMY' -O- | sed -rn
# 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HyGiCxE748kt3dAhHTHxHHHMjj0VcpMY" -O dart.doc2vec && rm -rf /tmp/cookies.txt
# dart.doc2vec.trainables.syn1neg.npy 파일 다운로드
# !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet
# --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate
# 'https://docs.google.com/uc?export=download&id=1XTgLCu0BmLPpaU960po3YWZsGYMNH7r3' -O- | sed -rn
# 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XTgLCu0BmLPpaU960po3YWZsGYMNH7r3" -O dart.doc2vec.trainables.syn1neg.npy
# && rm -rf /tmp/cookies.txt
# dart.doc2vec.wv.vectors.npy 파일 다운로드
# !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet
# --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate
# 'https://docs.google.com/uc?export=download&id=1tZOkSypiT0djwAZOInFnnQrJO9rGOdv2' -O- | sed -rn
# 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XTgLCu0BmLPpaU960po3YWZsGYMNH7r3" -O dart.doc2vec.wv.vectors.npy
# && rm -rf /tmp/cookies.txt
#
# # 모델을 로드합니다.
model = Doc2Vec.load("../data/dart1.doc2vec")
model.dv.norms = None

similar_doc = model.dv.most_similar("동화약품")
print(similar_doc)
similar_doc = model.dv.most_similar("하이트진로")
print(similar_doc)
similar_doc = model.dv.most_similar("LG이노텍")
print(similar_doc)
similar_doc = model.dv.most_similar("메리츠화재")
print(similar_doc)
similar_doc = model.dv.most_similar("카카오")
print(similar_doc)
