import tensorflow as tf

# 메모리 네트워크를 이용한 한국어 QA

# 데이터는 각각 아래의 링크에서 다운로드 할 수 있습니다.
# 훈련 데이터 : https://bit.ly/31SqtHy
# 테스트 데이터 : https://bit.ly/3f7rH5g

## 단어 사전 등록이 간편한 형태소 분석기 customized_konlpy


# pip install customized_konlpy

from ckonlpy.tag import Twitter

twitter = Twitter()
twitter.morphs("은경이는 사무실로 갔습니다.")

twitter.add_dictionary("은경이", "Noun")
twitter.morphs("은경이는 사무실로 갔습니다.")

## 데이터 로드

from ckonlpy.tag import Twitter
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from nltk import FreqDist
from functools import reduce
import os
import re
import matplotlib.pyplot as plt

TRAIN_FILE = os.path.join("../data/qa1_single-supporting-fact_train_kor.txt")
TEST_FILE = os.path.join("../data/qa1_single-supporting-fact_test_kor.txt")

## Babi 데이터셋 확인

i = 0
lines = open(TRAIN_FILE, "rb")
for line in lines:
    line = line.decode("utf-8").strip()
    # lno, text = line.split(" ", 1) # ID와 TEXT 분리
    i = i + 1
    print(line)
    if i == 20:
        break

## 스토리, 질문, 답변 분리


def read_data(dir_):
    stories, questions, answers = [], [], []  # 각각 스토리, 질문, 답변을 저장할 예정
    story_temp = []  # 현재 시점의 스토리 임시 저장
    lines = open(dir_, "rb")

    for line in lines:
        line = line.decode("utf-8")  # b' 제거
        line = line.strip()  # '\n' 제거
        idx, text = line.split(" ", 1)  # 맨 앞에 있는 id number 분리
        # 여기까지는 모든 줄에 적용되는 전처리

        if int(idx) == 1:
            story_temp = []

        if "\t" in text:  # 현재 읽는 줄이 질문 (tab) 답변 (tab)인 경우
            question, answer, _ = text.split("\t")  # 질문과 답변을 각각 저장
            stories.append([x for x in story_temp if x])  # 지금까지의 누적 스토리를 스토리에 저장
            questions.append(question)
            answers.append(answer)

        else:  # 현재 읽는 줄이 스토리인 경우
            story_temp.append(text)  # 임시 저장

    lines.close()
    return stories, questions, answers


train_data = read_data(TRAIN_FILE)
test_data = read_data(TEST_FILE)

train_stories, train_questions, train_answers = read_data(TRAIN_FILE)
test_stories, test_questions, test_answers = read_data(TEST_FILE)

print("훈련용 스토리의 개수 :", len(train_stories))
print("훈련용 질문의 개수 :", len(train_questions))
print("훈련용 답변의 개수 :", len(train_answers))
print("테스트용 스토리의 개수 :", len(test_stories))
print("테스트용 질문의 개수 :", len(test_questions))
print("테스트용 답변의 개수 :", len(test_answers))
print(train_stories[3572])
print(train_questions[3572])
print(train_answers[3572])

## 단어 집합 생성 및 토큰화 및 스토리와 질문의 최대 길이 구하기

# 이제 토큰화 함수를 정의하고, 이로부터 Vocabulary를 생성하는 함수를 만들어봅시다. 아래의 함수는 영어 데이터셋에 사용했던 토큰화 함수와
# 동일합니다. 현재는 한국어이므로 아래의 토큰화 함수를 그대로 사용하는 것은 바람직하지는 않지만, 임시로 사용해보겠습니다. 어절 단위로 했을
# 때 어떤 단어들이 있는지 출력해보기 위함입니다.


def tokenize(sent):
    return [x.strip() for x in re.split("(\W+)?", sent) if x.strip()]


def preprocess_data(train_data, test_data):
    counter = FreqDist()

    # 두 문장의 story를 하나의 문장으로 통합하는 함수
    flatten = lambda data: reduce(lambda x, y: x + y, data)

    # 각 샘플의 길이를 저장하는 리스트
    story_len = []
    question_len = []

    for stories, questions, answers in [train_data, test_data]:
        for story in stories:
            stories = tokenize(flatten(story))  # 스토리의 문장들을 펼친 후 토큰화
            story_len.append(len(stories))  # 각 story의 길이 저장
            for word in stories:  # 단어 집합에 단어 추가
                counter[word] += 1
        for question in questions:
            question = tokenize(question)
            question_len.append(len(question))
            for word in question:
                counter[word] += 1
        for answer in answers:
            answer = tokenize(answer)
            for word in answer:
                counter[word] += 1

    # 단어 집합 생성
    word2idx = {word: (idx + 1) for idx, (word, _) in enumerate(counter.most_common())}
    idx2word = {idx: word for word, idx in word2idx.items()}

    # 가장 긴 샘플의 길이
    story_max_len = np.max(story_len)
    question_max_len = np.max(question_len)

    return word2idx, idx2word, story_max_len, question_max_len


# word2idx를 출력해봅시다.
word2idx, idx2word, story_max_len, question_max_len = preprocess_data(train_data, test_data)
print(word2idx)

# 띄어쓰기 단위, 다시 말해 어절 단위로 했을 때 나오는 총 토큰의 수는 24개입니다. 19번 토큰부터 24번 토큰까지를 봤을 때 장소에 해당되는
# 명사들은 '화장실', '정원', '사무실', '침실', '복도', '부엌'이 있는 것 같습니다. 그렇다면, 11번 토큰부터 19번 토큰 사이에 등장하는
# '화장실로', '정원으로', '복도로', '부엌으로', '사무실로', '침실로'로 분리된 토큰들은 형태소 분석을 하였을 때 전부 '화장실', '정원',
# '사무실', '침실', '복도', '부엌'으로 분리되어야 합니다.

## 형태소 분석기 사전 등록
# '화장실로, 정원으로, 복도로, 부엌으로, 사무실로, 침실로'가 형태소 분석기가 정상적으로 분리하는지 확인이 필요함.


twitter = Twitter()

print(twitter.morphs("은경이는 화장실로 이동했습니다."))
print(twitter.morphs("경임이는 정원으로 가버렸습니다."))
print(twitter.morphs("수종이는 복도로 뛰어갔습니다."))
print(twitter.morphs("필웅이는 부엌으로 복귀했습니다."))
print(twitter.morphs("수종이는 사무실로 갔습니다."))
print(twitter.morphs("은경이는 침실로 갔습니다."))

twitter.add_dictionary("은경이", "Noun")
twitter.add_dictionary("경임이", "Noun")
twitter.add_dictionary("수종이", "Noun")

print(twitter.morphs("은경이는 화장실로 이동했습니다."))
print(twitter.morphs("경임이는 정원으로 가버렸습니다."))
print(twitter.morphs("수종이는 복도로 뛰어갔습니다."))
print(twitter.morphs("필웅이는 부엌으로 복귀했습니다."))
print(twitter.morphs("수종이는 사무실로 갔습니다."))
print(twitter.morphs("은경이는 침실로 갔습니다."))

##단어 집합 생성 및 토큰화 및 스토리와 질문의 최대 길이 구하기(다시)


def tokenize(sent):
    return twitter.morphs(sent)


word2idx, idx2word, story_max_len, question_max_len = preprocess_data(train_data, test_data)
print(word2idx)
print(idx2word)

vocab_size = len(word2idx) + 1
print(vocab_size)

print("스토리의 최대 길이 :", story_max_len)
print("질문의 최대 길이 :", question_max_len)

## 정수 인코딩 및 패딩


def vectorize(data, word2idx, story_maxlen, question_maxlen):
    Xs, Xq, Y = [], [], []
    flatten = lambda data: reduce(lambda x, y: x + y, data)

    stories, questions, answers = data
    for story, question, answer in zip(stories, questions, answers):
        xs = [word2idx[w] for w in tokenize(flatten(story))]
        xq = [word2idx[w] for w in tokenize(question)]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2idx[answer])

        # 스토리와 질문은 각각의 최대 길이로 패딩
        # 정답은 원-핫 인코딩
    return (
        pad_sequences(Xs, maxlen=story_maxlen),
        pad_sequences(Xq, maxlen=question_maxlen),
        to_categorical(Y, num_classes=len(word2idx) + 1),
    )


Xstrain, Xqtrain, Ytrain = vectorize(train_data, word2idx, story_max_len, question_max_len)
Xstest, Xqtest, Ytest = vectorize(test_data, word2idx, story_max_len, question_max_len)

print(Xstrain.shape, Xqtrain.shape, Ytrain.shape, Xstest.shape, Xqtest.shape, Ytest.shape)

## 메모리 네트워크 구현

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Permute, dot, add, concatenate
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Activation

# 에포크 횟수
train_epochs = 120
# 배치 크기
batch_size = 32
# 임베딩 크기
embed_size = 50
# LSTM의 크기
lstm_size = 64
# 과적합 방지 기법인 드롭아웃 적용 비율
dropout_rate = 0.30

# 플레이스 홀더. 입력을 담는 변수
input_sequence = Input((story_max_len,))
question = Input((question_max_len,))

print("Stories :", input_sequence)
print("Question:", question)

# 스토리를 위한 첫번째 임베딩. 그림에서의 Embedding A
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=embed_size))
input_encoder_m.add(Dropout(dropout_rate))
# 결과 : (samples, story_max_len, embedding_dim) / 샘플의 수, 문장의 최대 길이, 임베딩 벡터의 차원

# 스토리를 위한 두번째 임베딩. 그림에서의 Embedding C
# 임베딩 벡터의 차원을 question_max_len(질문의 최대 길이)로 한다.
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=question_max_len))
input_encoder_c.add(Dropout(dropout_rate))
# 결과 : (samples, story_max_len, question_max_len) / 샘플의 수, 문장의 최대 길이, 질문의 최대 길이(임베딩 벡터의 차원)

# 질문을 위한 임베딩. 그림에서의 Embedding B
question_encoder = Sequential()
question_encoder.add(
    Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=question_max_len)
)
question_encoder.add(Dropout(dropout_rate))
# 결과 : (samples, question_max_len, embedding_dim) / 샘플의 수, 질문의 최대 길이, 임베딩 벡터의 차원

# 실질적인 임베딩 과정
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

print("Input encoded m", input_encoded_m)
print("Input encoded c", input_encoded_c)
print("Question encoded", question_encoded)

# 스토리 단어들과 질문 단어들 간의 유사도를 구하는 과정
# 유사도는 내적을 사용한다.
match = dot([input_encoded_m, question_encoded], axes=-1, normalize=False)
match = Activation("softmax")(match)
print("Match shape", match)
# 결과 : (samples, story_maxlen, question_max_len) / 샘플의 수, 문장의 최대 길이, 질문의 최대 길이

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_max_len, question_max_len)
response = Permute((2, 1))(response)  # (samples, question_max_len, story_max_len)
print("Response shape", response)

# concatenate the response vector with the question vector sequence
answer = concatenate([response, question_encoded])
print("Answer shape", answer)

answer = LSTM(lstm_size)(answer)  # Generate tensors of shape 32
answer = Dropout(dropout_rate)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer = Activation("softmax")(answer)

# build the final model
model = Model([input_sequence, question], answer)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])

print(model.summary())

# start training the model
history = model.fit(
    [Xstrain, Xqtrain], Ytrain, batch_size, train_epochs, validation_data=([Xstest, Xqtest], Ytest)
)

# save model
model.save("model.h5")

# plot accuracy and loss plot
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="g", label="train")
plt.plot(history.history["val_acc"], color="b", label="validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="train")
plt.plot(history.history["val_loss"], color="b", label="validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

# labels
ytest = np.argmax(Ytest, axis=1)

# get predictions
Ytest_ = model.predict([Xstest, Xqtest])
ytest_ = np.argmax(Ytest_, axis=1)

NUM_DISPLAY = 30

print("{:18}|{:5}|{}".format("질문", "실제값", "예측값"))
print(39 * "-")

for i in range(NUM_DISPLAY):
    question = " ".join([idx2word[x] for x in Xqtest[i].tolist()])
    label = idx2word[ytest[i]]
    prediction = idx2word[ytest_[i]]
    print("{:20}: {:7} {}".format(question, label, prediction))
