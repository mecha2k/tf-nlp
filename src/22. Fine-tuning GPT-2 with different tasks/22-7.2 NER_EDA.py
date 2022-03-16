import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from transformers import BertTokenizer


train_df = pd.read_csv("../data/ner_train_data.csv")
test_df = pd.read_csv("../data/ner_test_data.csv")
print(train_df.head())

data_preparation = lambda df, column: [sent.split() for sent in df[column].values]
x_train = data_preparation(train_df, "Sentence")
x_test = data_preparation(test_df, "Sentence")
y_train = data_preparation(train_df, "Tag")
y_test = data_preparation(test_df, "Tag")
print("train samples : ", len(x_train))

labels = [sent.strip() for sent in open("../data/ner_label.txt", "r", encoding="utf-8")]
print("NER tagging info : ", labels)
print("NER tagging length : ", len(labels))

# # In[2]:
#
#
# # 데이터 불러오기
# DATA_IN_PATH = "data_in/KOR"
# DATA_TRAIN_PATH = os.path.join(DATA_IN_PATH, "NER", "train.tsv")
# DATA_TEST_PATH = os.path.join(DATA_IN_PATH, "NER", "test.tsv")
# DATA_LABEL_PATH = os.path.join(DATA_IN_PATH, "NER", "label.txt")
#
#
# # In[3]:
#
#
# def read_file(input_path):
#     """Read tsv file, and return words and label as list"""
#     with open(input_path, "r", encoding="utf-8") as f:
#         sentences = []
#         labels = []
#         for line in f:
#             split_line = line.strip().split("\t")
#             sentences.append(split_line[0])
#             labels.append(split_line[1])
#         return sentences, labels
#
#
# train_sentences, train_labels = read_file(DATA_TRAIN_PATH)
# test_sentences, test_labels = read_file(DATA_TEST_PATH)

ner_sentences = x_train + x_test
ner_labels = y_train + y_test

ner_df = pd.DataFrame({"sentence": ner_sentences, "label": ner_labels})
print("전체 ner_data 개수: ", len(ner_df))

sentences = ner_df["sentence"]
# print("유일한 총 문장 수 : {}".format(len(np.unique(sentences))))
print("unique sentences : ", sentences.nunique())
# print("반복해서 나타나는 문장의 수: {}".format(np.sum(train_set.value_counts() > 1)))


# # 그래프에 대한 이미지 사이즈 선언
# # figsize: (가로, 세로) 형태의 튜플로 입력
# plt.figure(figsize=(12, 5))
# # 히스토그램 선언
# # bins: 히스토그램 값들에 대한 버켓 범위
# # range: x축 값의 범위
# # alpha: 그래프 색상 투명도
# # color: 그래프 색상
# # label: 그래프에 대한 라벨
# plt.hist(train_set.value_counts(), bins=50, alpha=0.5, color="r", label="word")
# plt.yscale("log", nonposy="clip")
# # 그래프 제목
# plt.title("Log-Histogram of sentence appearance counts")
# # 그래프 x 축 라벨
# plt.xlabel("Number of occurrences of sentence")
# # 그래프 y 축 라벨
# plt.ylabel("Number of sentence")
#
#
# # In[7]:
#
#
# print("중복 최대 개수: {}".format(np.max(train_set.value_counts())))
# print("중복 최소 개수: {}".format(np.min(train_set.value_counts())))
# print("중복 평균 개수: {:.2f}".format(np.mean(train_set.value_counts())))
# print("중복 표준편차: {:.2f}".format(np.std(train_set.value_counts())))
# print("중복 중간길이: {}".format(np.median(train_set.value_counts())))
# # 사분위의 대한 경우는 0~100 스케일로 되어있음
# print("제 1 사분위 중복: {}".format(np.percentile(train_set.value_counts(), 25)))
# print("제 3 사분위 중복: {}".format(np.percentile(train_set.value_counts(), 75)))
#
#
# # In[8]:
#
#
# plt.figure(figsize=(12, 5))
# # 박스플롯 생성
# # 첫번째 파라메터: 여러 분포에 대한 데이터 리스트를 입력
# # labels: 입력한 데이터에 대한 라벨
# # showmeans: 평균값을 마크함
#
# plt.boxplot([train_set.value_counts()], labels=["counts"], showmeans=True)
#
#
# # In[9]:
#
#
# train_length = train_set.apply(len)
#
#
# # In[12]:
#
#
# train_set[0]
#
#
# # In[10]:
#
#
# train_length
#
#
# # In[11]:
#
#
# print("문장 길이 최대 값: {}".format(np.max(train_length)))
# print("문장 길이 평균 값: {:.2f}".format(np.mean(train_length)))
# print("문장 길이 표준편차: {:.2f}".format(np.std(train_length)))
# print("문장 길이 중간 값: {}".format(np.median(train_length)))
# print("문장 길이 제 1 사분위: {}".format(np.percentile(train_length, 25)))
# print("문장 길이 제 3 사분위: {}".format(np.percentile(train_length, 75)))
#
#
# # In[13]:
#
#
# plt.figure(figsize=(15, 10))
# plt.hist(train_length, bins=200, range=[0, 200], facecolor="r", density=True, label="train")
# plt.title("Distribution of character count in sentence", fontsize=15)
# plt.legend()
# plt.xlabel("Number of characters", fontsize=15)
# plt.ylabel("Probability", fontsize=15)
#
#
# # In[47]:
#
#
# plt.figure(figsize=(12, 5))
#
# plt.boxplot(train_length, labels=["char counts"], showmeans=True)
#
#
# # In[24]:
#
#
# train_word_counts = train_set.apply(lambda x: len(x.split(" ")))
#
# print("문장 단어 개수 최대 값: {}".format(np.max(train_word_counts)))
# print("문장 단어 개수 평균 값: {:.2f}".format(np.mean(train_word_counts)))
# print("문장 단어 개수 표준편차: {:.2f}".format(np.std(train_word_counts)))
# print("문장 단어 개수 중간 값: {}".format(np.median(train_word_counts)))
# print("문장 단어 개수 제 1 사분위: {}".format(np.percentile(train_word_counts, 25)))
# print("문장 단어 개수 제 3 사분위: {}".format(np.percentile(train_word_counts, 75)))
# print("문장 단어 개수 99 퍼센트: {}".format(np.percentile(train_word_counts, 99)))
#
# # 문장 단어 개수 최대 값: 175
# # 문장 단어 개수 평균 값: 11.81
# # 문장 단어 개수 표준편차: 7.03
# # 문장 단어 개수 중간 값: 10.0
# # 문장 단어 개수 제 1 사분위: 7.0
# # 문장 단어 개수 제 3 사분위: 15.0
# # 문장 단어 개수 99 퍼센트: 35.0
#
#
# # In[23]:
#
#
# plt.figure(figsize=(15, 10))
# plt.hist(train_word_counts, bins=50, range=[0, 50], facecolor="r", density=True, label="train")
# plt.title("Distribution of word count in sentence", fontsize=15)
# plt.legend()
# plt.xlabel("Number of words", fontsize=15)
# plt.ylabel("Probability", fontsize=15)
#
#
# # In[18]:
#
#
# plt.figure(figsize=(12, 5))
#
# plt.boxplot(train_word_counts, labels=["counts"], showmeans=True)
#
#
# # In[19]:
#
#
# qmarks = np.mean(train_set.apply(lambda x: "?" in x))  # 물음표가 구두점으로 쓰임
# math = np.mean(train_set.apply(lambda x: "[math]" in x))  # []
# fullstop = np.mean(train_set.apply(lambda x: "." in x))  # 마침표
# capital_first = np.mean(train_set.apply(lambda x: x[0].isupper()))  #  첫번째 대문자
# capitals = np.mean(train_set.apply(lambda x: max([y.isupper() for y in x])))  # 대문자가 몇개
# numbers = np.mean(train_set.apply(lambda x: max([y.isdigit() for y in x])))  # 숫자가 몇개
#
#
# # In[20]:
#
#
# qmarks = np.mean(train_set.apply(lambda x: "?" in x))  # 물음표가 구두점으로 쓰임
# math = np.mean(train_set.apply(lambda x: "[math]" in x))  # []
# fullstop = np.mean(train_set.apply(lambda x: "." in x))  # 마침표
# capital_first = np.mean(train_set.apply(lambda x: x[0].isupper()))  #  첫번째 대문자
# capitals = np.mean(train_set.apply(lambda x: max([y.isupper() for y in x])))  # 대문자가 몇개
# numbers = np.mean(train_set.apply(lambda x: max([y.isdigit() for y in x])))  # 숫자가 몇개
#
# print("물음표가있는 문장: {:.2f}%".format(qmarks * 100))
# print("수학 태그가있는 문장: {:.2f}%".format(math * 100))
# print("마침표를 포함한 문장: {:.2f}%".format(fullstop * 100))
# print("첫 글자가 대문자 인 문장: {:.2f}%".format(capital_first * 100))
# print("대문자가있는 문장: {:.2f}%".format(capitals * 100))
# print("숫자가있는 문장: {:.2f}%".format(numbers * 100))
#
#
# # ## Tokenizer cased
#
# # In[20]:
#
#
# tokenizer = BertTokenizer.from_pretrained(
#     "bert-base-multilingual-cased", cache_dir="bert_ckpt", do_lower_case=False
# )
#
#
# # In[21]:
#
#
# train_bert_token_counts = train_set.apply(lambda x: len(tokenizer.tokenize(x)))
#
#
# # In[22]:
#
#
# plt.figure(figsize=(15, 10))
# plt.hist(
#     train_bert_token_counts, bins=200, range=[0, 200], facecolor="r", density=True, label="train"
# )
# plt.title("Distribution of tokens count in sentence", fontsize=15)
# plt.legend()
# plt.xlabel("Number of tokens", fontsize=15)
# plt.ylabel("Probability", fontsize=15)
#
#
# # In[55]:
#
#
# print("문장 tokens 개수 최대 값: {}".format(np.max(train_bert_token_counts)))
# print("문장 tokens 개수 평균 값: {:.2f}".format(np.mean(train_bert_token_counts)))
# print("문장 tokens 개수 표준편차: {:.2f}".format(np.std(train_bert_token_counts)))
# print("문장 tokens 개수 중간 값: {}".format(np.median(train_bert_token_counts)))
# print("문장 tokens 개수 제 1 사분위: {}".format(np.percentile(train_bert_token_counts, 25)))
# print("문장 tokens 개수 제 3 사분위: {}".format(np.percentile(train_bert_token_counts, 75)))
# print("문장 tokens 개수 99 퍼센트: {}".format(np.percentile(train_bert_token_counts, 99)))
#
#
# # In[61]:
#
#
# plt.figure(figsize=(12, 5))
#
# plt.boxplot(train_bert_token_counts, labels=["counts"], showmeans=True)
