import tensorflow as tf
import tensorflow_datasets as tfds
import urllib.request
import pandas as pd

train_df = pd.read_csv("../data/IMDb_Reviews.csv")
print(train_df["review"])

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    train_df["review"], target_vocab_size=2 ** 13
)

print(tokenizer.subwords[:100])
print(train_df["review"][20])
print("Tokenized sample question: {}".format(tokenizer.encode(train_df["review"][20])))

# train_df에 존재하는 문장 중 일부를 발췌
sample_string = "It's mind-blowing to me that this film was even made."

# 인코딩한 결과를 tokenized_string에 저장
tokenized_string = tokenizer.encode(sample_string)
print("정수 인코딩 후의 문장 : {}".format(tokenized_string))

# 이를 다시 디코딩
original_string = tokenizer.decode(tokenized_string)
print("기존 문장 : {}".format(original_string))

print("단어 집합의 크기(Vocab size) :", tokenizer.vocab_size)

for ts in tokenized_string:
    print("{} ----> {}".format(ts, tokenizer.decode([ts])))

sample_string = "It's mind-blowing to me that this film was evenxyz made."

# encode
tokenized_string = tokenizer.encode(sample_string)
print("정수 인코딩 후의 문장 {}".format(tokenized_string))

# encoding한 문장을 다시 decode
original_string = tokenizer.decode(tokenized_string)
print("기존 문장: {}".format(original_string))

assert original_string == sample_string

for ts in tokenized_string:
    print("{} ----> {}".format(ts, tokenizer.decode([ts])))

# 네이버 영화 리뷰에 대해서도 위에서 IMDB 영화 리뷰에 대해서 수행한 동일한 작업을 진행해봅시다.
train_data = pd.read_table("../data/ratings_train.txt")
test_data = pd.read_table("../data/ratings_test.txt")

print(train_data[:5])  # 상위 5개 출력
print(train_data.isnull().values.any())
print(train_data.isnull().sum())

train_data = train_data.dropna(how="any")  # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any())  # Null 값이 존재하는지 확인

# The vocabulary is "trained" on a corpus and all wordpieces are stored in a vocabulary file. To generate a vocabulary
# from a corpus, use tfds.features.text.SubwordTextEncoder.build_from_corpus.

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    train_data["document"], target_vocab_size=2 ** 13
)

print(tokenizer.subwords[:100])
print(train_data["document"][20])
print("Tokenized sample question: {}".format(tokenizer.encode(train_data["document"][20])))

sample_string = train_data["document"][21]

# encode
tokenized_string = tokenizer.encode(sample_string)
print("정수 인코딩 후의 문장 {}".format(tokenized_string))

# encoding한 문장을 다시 decode
original_string = tokenizer.decode(tokenized_string)
print("기존 문장: {}".format(original_string))

assert original_string == sample_string

for ts in tokenized_string:
    print("{} ----> {}".format(ts, tokenizer.decode([ts])))

sample_string = "이 영화 굉장히 재밌다 킄핫핫ㅎ"

# encode
tokenized_string = tokenizer.encode(sample_string)
print("정수 인코딩 후의 문장 {}".format(tokenized_string))

# encoding한 문장을 다시 decode
original_string = tokenizer.decode(tokenized_string)
print("기존 문장: {}".format(original_string))

assert original_string == sample_string

for ts in tokenized_string:
    print("{} ----> {}".format(ts, tokenizer.decode([ts])))
