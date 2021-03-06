# pip install wikiextractor

# Commented out IPython magic to ensure Python compatibility.
# Colab에 Mecab 설치
# !git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab
# !bash install_mecab-ko_on_colab190912.sh
# !wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2

# import urllib.request
# urllib.request.urlretrieve(
#     "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2",
#     filename="../data/kowiki-latest-pages-articles.xml.bz2",
# )

# !python -m wikiextractor.WikiExtractor kowiki-latest-pages-articles.xml.bz2

from konlpy.tag import Mecab
from gensim.models import Word2Vec

from tqdm import tqdm
import platform
import os, re

osname = platform.system()
if osname == "Windows":
    mecab = Mecab(dicpath="C:/mecab/mecab-ko-dic")
else:
    mecab = Mecab()


def list_wiki(dirname):
    filepaths = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)

        if os.path.isdir(filepath):
            # 재귀 함수
            filepaths.extend(list_wiki(filepath))
        else:
            find = re.findall(r"wiki_[0-9][0-9]", filepath)
            if 0 < len(find):
                filepaths.append(filepath)
    return sorted(filepaths)


# filepaths = list_wiki("../data/text")
# print(os.listdir("../data/text"))
# print(len(filepaths))
# print(type(filepaths))
#
#
# with open("../data/text/output_file.txt", "w") as outfile:
#     for filename in filepaths:
#         with open(filename) as infile:
#             contents = infile.read()
#             outfile.write(contents)
#
# f = open("../data/text/output_file.txt", encoding="utf8")
#
# i = 0
# while True:
#     line = f.readline()
#     if line != "\n":
#         i = i + 1
#         print("%d번째 줄 :" % i + line)
#     if i == 10:
#         break
# f.close()


f = open("../data/text/output_file.txt", encoding="utf8")
lines = f.read().splitlines()
print(len(lines))
print(lines[:10])


lines = lines[:1000000]

result = []
for line in tqdm(lines):
    # 빈 문자열이 아닌 경우에만 수행
    if line:
        result.append(mecab.morphs(line))
print(len(result))

model = Word2Vec(result, vector_size=100, window=5, min_count=5, workers=16, sg=0)
model_result1 = model.wv.most_similar("대한민국")
print(model_result1)
model_result2 = model.wv.most_similar("어벤져스")
print(model_result2)
model_result3 = model.wv.most_similar("반도체")
print(model_result3)
