import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import os

from konlpy.tag import Mecab
from collections import Counter
from wordcloud import WordCloud
import squarify


np.random.seed(42)
plt.style.use("seaborn")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

if platform.system() == "Windows":
    mecab = Mecab(dicpath="c:/mecab/mecab-ko-dic")
else:
    mecab = Mecab()


df = pd.read_table("../data/ratings.txt")
print(len(df))
df = df.drop_duplicates(subset=["document"])
df = df.dropna(how="any", axis=0)
print(len(df))
print(df.head())
print(df.isna().values.any())
print(df.isna().sum())

df["document"] = df["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
reviews = df["document"].tolist()
reviews = reviews[:1000]
print(reviews[0])

stop_words = "영화 전 난 일 걸 뭐 줄 만 건 분 개 끝 잼 이거 번 중 듯 때 게 내 말 나 수 거 점 것"
stop_words = stop_words.split()
print(stop_words)

nouns = []
for review in reviews:
    for noun in mecab.nouns(review):
        if noun not in stop_words:
            nouns.append(noun)
print(nouns[:10])


counter = Counter(nouns)
top_nouns = dict(counter.most_common(n=50))

y_pos = np.arange(len(top_nouns))
plt.figure(figsize=(12, 12))
plt.barh(y_pos, top_nouns.values())
plt.title("Word Top Counts")
plt.yticks(y_pos, top_nouns.keys())
plt.savefig("images/02-nouns_count")

font_path = "../data/NanumBarunGothic.ttf"
wc = WordCloud(font_path=font_path, width=800, height=600, background_color="black")
wc.generate_from_frequencies(top_nouns)

plt.figure(figsize=(8, 6))
plt.imshow(wc)
plt.axis("off")
plt.savefig("images/02-wordcloud")

norm = mpl.colors.Normalize(vmin=min(top_nouns.values()), vmax=max(top_nouns.values()), clip=False)
colors = [mpl.cm.Blues(norm(value)) for value in top_nouns.values()]

plt.figure(figsize=(10, 6))
squarify.plot(label=top_nouns.keys(), sizes=top_nouns.values(), color=colors, alpha=0.7)
plt.savefig("images/02-squarify")
