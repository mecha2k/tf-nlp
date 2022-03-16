import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from transformers import BertTokenizer


DATA_IN_PATH = "./data_in/KOR"


TRAIN_XNLI_DF = os.path.join(DATA_IN_PATH, "KorNLI", "multinli.train.ko.tsv")

multinli_data = pd.read_csv(TRAIN_XNLI_DF, sep="\t", error_bad_lines=False)
multinli_data.head(10)


print("전체 multinli_data 개수: {}".format(len(multinli_data)))


TRAIN_SNLI_DF = os.path.join(DATA_IN_PATH, "KorNLI", "snli_1.0_train.kor.tsv")

snli_data = pd.read_csv(TRAIN_SNLI_DF, sep="\t", error_bad_lines=False)
snli_data.head(10)


print("전체 snli_data 개수: {}".format(len(snli_data)))


train_data = pd.concat([multinli_data, snli_data], axis=0)
train_data.head(10)


print("전체 train_data 개수: {}".format(len(train_data)))


train_set = pd.Series(train_data["sentence1"].tolist() + train_data["sentence2"].tolist()).astype(
    str
)
train_set.head()


print("전체 문장 데이터의 개수: {}".format(len(train_set)))


print("유일한 총 문장 수 : {}".format(len(np.unique(train_set))))
print("반복해서 나타나는 문장의 수: {}".format(np.sum(train_set.value_counts() > 1)))


# 그래프에 대한 이미지 사이즈 선언
# figsize: (가로, 세로) 형태의 튜플로 입력
plt.figure(figsize=(12, 5))
# 히스토그램 선언
# bins: 히스토그램 값들에 대한 버켓 범위
# range: x축 값의 범위
# alpha: 그래프 색상 투명도
# color: 그래프 색상
# label: 그래프에 대한 라벨
plt.hist(train_set.value_counts(), bins=50, alpha=0.5, color="r", label="word")
plt.yscale("log", nonposy="clip")
# 그래프 제목
plt.title("Log-Histogram of sentence appearance counts")
# 그래프 x 축 라벨
plt.xlabel("Number of occurrences of sentence")
# 그래프 y 축 라벨
plt.ylabel("Number of sentence")


print("중복 최대 개수: {}".format(np.max(train_set.value_counts())))
print("중복 최소 개수: {}".format(np.min(train_set.value_counts())))
print("중복 평균 개수: {:.2f}".format(np.mean(train_set.value_counts())))
print("중복 표준편차: {:.2f}".format(np.std(train_set.value_counts())))
print("중복 중간길이: {}".format(np.median(train_set.value_counts())))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print("제 1 사분위 중복: {}".format(np.percentile(train_set.value_counts(), 25)))
print("제 3 사분위 중복: {}".format(np.percentile(train_set.value_counts(), 75)))


plt.figure(figsize=(12, 5))
# 박스플롯 생성
# 첫번째 파라메터: 여러 분포에 대한 데이터 리스트를 입력
# labels: 입력한 데이터에 대한 라벨
# showmeans: 평균값을 마크함

plt.boxplot([train_set.value_counts()], labels=["counts"], showmeans=True)


train_length = train_set.apply(len)


plt.figure(figsize=(15, 10))
plt.hist(train_length, bins=200, range=[0, 200], facecolor="r", density=True, label="train")
plt.title("Distribution of character count in sentence", fontsize=15)
plt.legend()
plt.xlabel("Number of characters", fontsize=15)
plt.ylabel("Probability", fontsize=15)


print("문장 길이 최대 값: {}".format(np.max(train_length)))
print("문장 길이 평균 값: {:.2f}".format(np.mean(train_length)))
print("문장 길이 표준편차: {:.2f}".format(np.std(train_length)))
print("문장 길이 중간 값: {}".format(np.median(train_length)))
print("문장 길이 제 1 사분위: {}".format(np.percentile(train_length, 25)))
print("문장 길이 제 3 사분위: {}".format(np.percentile(train_length, 75)))


plt.figure(figsize=(12, 5))

plt.boxplot(train_length, labels=["char counts"], showmeans=True)


train_word_counts = train_set.apply(lambda x: len(x.split(" ")))


plt.figure(figsize=(15, 10))
plt.hist(train_word_counts, bins=50, range=[0, 50], facecolor="r", density=True, label="train")
plt.title("Distribution of word count in sentence", fontsize=15)
plt.legend()
plt.xlabel("Number of words", fontsize=15)
plt.ylabel("Probability", fontsize=15)


print("문장 단어 개수 최대 값: {}".format(np.max(train_word_counts)))
print("문장 단어 개수 평균 값: {:.2f}".format(np.mean(train_word_counts)))
print("문장 단어 개수 표준편차: {:.2f}".format(np.std(train_word_counts)))
print("문장 단어 개수 중간 값: {}".format(np.median(train_word_counts)))
print("문장 단어 개수 제 1 사분위: {}".format(np.percentile(train_word_counts, 25)))
print("문장 단어 개수 제 3 사분위: {}".format(np.percentile(train_word_counts, 75)))
print("문장 단어 개수 99 퍼센트: {}".format(np.percentile(train_word_counts, 99)))


plt.figure(figsize=(12, 5))

plt.boxplot(train_word_counts, labels=["counts"], showmeans=True)


qmarks = np.mean(train_set.apply(lambda x: "?" in x))  # 물음표가 구두점으로 쓰임
math = np.mean(train_set.apply(lambda x: "[math]" in x))  # []
fullstop = np.mean(train_set.apply(lambda x: "." in x))  # 마침표
capital_first = np.mean(train_set.apply(lambda x: x[0].isupper()))  # 첫번째 대문자
capitals = np.mean(train_set.apply(lambda x: max([y.isupper() for y in x])))  # 대문자가 몇개
numbers = np.mean(train_set.apply(lambda x: max([y.isdigit() for y in x])))  # 숫자가 몇개


qmarks = np.mean(train_set.apply(lambda x: "?" in x))  # 물음표가 구두점으로 쓰임
math = np.mean(train_set.apply(lambda x: "[math]" in x))  # []
fullstop = np.mean(train_set.apply(lambda x: "." in x))  # 마침표
capital_first = np.mean(train_set.apply(lambda x: x[0].isupper()))  # 첫번째 대문자
capitals = np.mean(train_set.apply(lambda x: max([y.isupper() for y in x])))  # 대문자가 몇개
numbers = np.mean(train_set.apply(lambda x: max([y.isdigit() for y in x])))  # 숫자가 몇개

print("물음표가있는 문장: {:.2f}%".format(qmarks * 100))
print("수학 태그가있는 문장: {:.2f}%".format(math * 100))
print("마침표를 포함한 문장: {:.2f}%".format(fullstop * 100))
print("첫 글자가 대문자 인 문장: {:.2f}%".format(capital_first * 100))
print("대문자가있는 문장: {:.2f}%".format(capitals * 100))
print("숫자가있는 문장: {:.2f}%".format(numbers * 100))


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased", cache_dir="bert_ckpt", do_lower_case=False
)

# ## Tokenizer Cased


train_bert_token_counts = train_set.apply(lambda x: len(tokenizer.tokenize(x)))


plt.figure(figsize=(15, 10))
plt.hist(
    train_bert_token_counts, bins=200, range=[0, 200], facecolor="r", density=True, label="train"
)
plt.title("Distribution of tokens count in sentence", fontsize=15)
plt.legend()
plt.xlabel("Number of tokens", fontsize=15)
plt.ylabel("Probability", fontsize=15)


print("문장 tokens 개수 최대 값: {}".format(np.max(train_bert_token_counts)))
print("문장 tokens 개수 평균 값: {:.2f}".format(np.mean(train_bert_token_counts)))
print("문장 tokens 개수 표준편차: {:.2f}".format(np.std(train_bert_token_counts)))
print("문장 tokens 개수 중간 값: {}".format(np.median(train_bert_token_counts)))
print("문장 tokens 개수 제 1 사분위: {}".format(np.percentile(train_bert_token_counts, 25)))
print("문장 tokens 개수 제 3 사분위: {}".format(np.percentile(train_bert_token_counts, 75)))
print("문장 tokens 개수 99 퍼센트: {}".format(np.percentile(train_bert_token_counts, 99)))


plt.figure(figsize=(12, 5))
plt.boxplot(train_bert_token_counts, labels=["counts"], showmeans=True)


from wordcloud import WordCloud

font_path = os.path.join(DATA_IN_PATH, "NanumGothic.ttf")
cloud = WordCloud(font_path=font_path, width=800, height=600).generate(
    " ".join(train_set.astype(str))
)
plt.figure(figsize=(15, 10))
plt.imshow(cloud)
plt.axis("off")

fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(10, 3)
sns.countplot(train_data["gold_label"])
