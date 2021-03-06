import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from icecream import ic

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# 1. 정수 인코딩(Integer Encoding)
## 1) dictionary 사용하기
raw_text = (
    "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret "
    "He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept "
    "his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the "
    "barber went up a huge mountain."
)

# 문장 토큰화
sentences = sent_tokenize(raw_text)
ic(sentences)

vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words("english"))

for sentence in sentences:
    # 단어 토큰화
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence:
        word = word.lower()  # 모든 단어를 소문자화하여 단어의 개수를 줄인다.
        if word not in stop_words:  # 단어 토큰화 된 결과에 대해서 불용어를 제거한다.
            if len(word) > 2:  # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거한다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
    preprocessed_sentences.append(result)
ic(preprocessed_sentences)
ic(vocab)
ic(vocab["barber"])  # 'barber'라는 단어의 빈도수 출력

vocab_sorted = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
ic(vocab_sorted)

word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted:
    if frequency > 1:  # 빈도수가 작은 단어는 제외.
        i = i + 1
        word_to_index[word] = i

ic(word_to_index)

vocab_size = 5
words_frequency = [
    word for word, index in word_to_index.items() if index >= vocab_size + 1
]  # 인덱스가 5 초과인 단어 제거
for w in words_frequency:
    del word_to_index[w]  # 해당 단어에 대한 인덱스 정보를 삭제
ic(word_to_index)

word_to_index["OOV"] = len(word_to_index) + 1

ic(word_to_index)

encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            encoded_sentence.append(word_to_index["OOV"])
    encoded_sentences.append(encoded_sentence)
ic(encoded_sentences)

## 2) Counter 사용하기

from collections import Counter

ic(preprocessed_sentences)

# words = np.hstack(preprocessed_sentences)으로도 수행 가능.
all_words_list = sum(preprocessed_sentences, [])
ic(all_words_list)

# 파이썬의 Counter 모듈을 이용하여 단어의 빈도수 카운트
vocab = Counter(all_words_list)
ic(vocab)
ic(vocab["barber"])  # 'barber'라는 단어의 빈도수 출력

vocab_size = 5
vocab = vocab.most_common(vocab_size)  # 등장 빈도수가 높은 상위 5개의 단어만 저장
ic(vocab)

word_to_index = {}
i = 0
for (word, frequency) in vocab:
    i = i + 1
    word_to_index[word] = i

ic(word_to_index)

## 3) NLTK의 FreqDist 사용하기

from nltk import FreqDist
import numpy as np

vocab = FreqDist(np.hstack(preprocessed_sentences))
ic(vocab["barber"])  # 'barber'라는 단어의 빈도수 출력

vocab_size = 5
vocab = vocab.most_common(vocab_size)  # 등장 빈도수가 높은 상위 5개의 단어만 저장
ic(vocab)

word_to_index = {word[0]: index + 1 for index, word in enumerate(vocab)}
ic(word_to_index)

## 4) enumerate 이해하기

test_input = ["a", "b", "c", "d", "e"]
for index, value in enumerate(test_input):  # 입력의 순서대로 0부터 인덱스를 부여함.
    ic("value : {}, index: {}".format(value, index))

from tensorflow.keras.preprocessing.text import Tokenizer

# 2. 케라스(Keras)의 텍스트 전처리
tokenizer = Tokenizer()

# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성.
tokenizer.fit_on_texts(preprocessed_sentences)

ic(tokenizer.word_index)
ic(tokenizer.word_counts)
ic(tokenizer.texts_to_sequences(preprocessed_sentences))

vocab_size = 5
tokenizer = Tokenizer(num_words=vocab_size + 1)  # 상위 5개 단어만 사용
tokenizer.fit_on_texts(preprocessed_sentences)

ic(tokenizer.word_index)
ic(tokenizer.word_counts)
ic(tokenizer.texts_to_sequences(preprocessed_sentences))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)

vocab_size = 5
# 인덱스가 5 초과인 단어 제거
words_frequency = [word for word, index in tokenizer.word_index.items() if index >= vocab_size + 1]
for word in words_frequency:
    del tokenizer.word_index[word]  # 해당 단어에 대한 인덱스 정보를 삭제
    del tokenizer.word_counts[word]  # 해당 단어에 대한 카운트 정보를 삭제

ic(tokenizer.word_index)
ic(tokenizer.word_counts)
ic(tokenizer.texts_to_sequences(preprocessed_sentences))

# 숫자 0과 OOV를 고려해서 단어 집합의 크기는 +2
vocab_size = 5
tokenizer = Tokenizer(num_words=vocab_size + 2, oov_token="OOV")
tokenizer.fit_on_texts(preprocessed_sentences)

ic("단어 OOV의 인덱스 : {}".format(tokenizer.word_index["OOV"]))
ic(tokenizer.texts_to_sequences(preprocessed_sentences))
