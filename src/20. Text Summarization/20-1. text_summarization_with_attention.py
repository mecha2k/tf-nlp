# 이번 실습에서 사용할 데이터는 아마존 리뷰 데이터입니다. 아래의 링크에서 데이터를 다운로드 가능합니다.
#
# 링크 : https://www.kaggle.com/snap/amazon-fine-food-reviews
#
# 위키독스의 https://wikidocs.net/72820 를 Colab에서 실행하였을 때의 코드입니다.
# 텐서플로우 2.6 버전에서 실행되었으며 구글 드라이브에 해당 Reviews.csv 파일을 업로드해놓고 진행했습니다.


# from google.colab import drive
# drive.mount('/content/gdrive/')

import tensorflow as tf
import nltk

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords")

# Commented out IPython magic to ensure Python compatibility.
# %cd gdrive

# Commented out IPython magic to ensure Python compatibility.
# %cd My Drive

data = pd.read_csv("../data/Reviews.csv", nrows=100000)
print(data.head())

data = data[["Text", "Summary"]]
data.head()

print("Text 열에서 중복을 배제한 유일한 샘플의 수 :", data["Text"].nunique())
print("Summary 열에서 중복을 배제한 유일한 샘플의 수 :", data["Summary"].nunique())

# text 열에서 중복인 내용이 있다면 중복 제거
data.drop_duplicates(subset=["Text"], inplace=True)
print("전체 샘플수 :", len(data))
print(data.isnull().sum())

# Null 값을 가진 샘플 제거
data.dropna(axis=0, inplace=True)
print("전체 샘플수 :", (len(data)))

# 전처리 함수 내 사용
contractions = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "here's": "here is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}

# NLTK의 불용어
stop_words = set(stopwords.words("english"))
print("불용어 개수 :", len(stop_words))
print(stop_words)

# 전처리 함수
def preprocess_sentence(sentence, remove_stopwords=True):
    sentence = sentence.lower()  # 텍스트 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text  # <br />, <a href = ...> 등의 html 태그 제거
    sentence = re.sub(
        r"\([^)]*\)", "", sentence
    )  # 괄호로 닫힌 문자열  제거 Ex) my husband (and myself) for => my husband for
    sentence = re.sub('"', "", sentence)  # 쌍따옴표 " 제거
    sentence = " ".join(
        [contractions[t] if t in contractions else t for t in sentence.split(" ")]
    )  # 약어 정규화
    sentence = re.sub(r"'s\b", "", sentence)  # 소유격 제거. Ex) roland's -> roland
    sentence = re.sub("[^a-zA-Z]", " ", sentence)  # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub("[m]{2,}", "mm", sentence)  # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah

    # 불용어 제거 (Text)
    if remove_stopwords:
        tokens = " ".join(
            word for word in sentence.split() if not word in stop_words if len(word) > 1
        )
    # 불용어 미제거 (Summary)
    else:
        tokens = " ".join(word for word in sentence.split() if len(word) > 1)
    return tokens


temp_text = (
    "Everything I bought was great, infact I ordered twice and the third ordered was<br />for my mother "
    "and father."
)
temp_summary = "Great way to start (or finish) the day!!!"
print(preprocess_sentence(temp_text))
print(preprocess_sentence(temp_summary, 0))

# Text 열 전처리
clean_text = []
for s in data["Text"]:
    clean_text.append(preprocess_sentence(s))
print(clean_text[:5])

# Summary 열 전처리
clean_summary = []
for s in data["Summary"]:
    clean_summary.append(preprocess_sentence(s, 0))
print(clean_summary[:5])

data["Text"] = clean_text
data["Summary"] = clean_summary

# 길이가 공백인 샘플은 NULL 값으로 변환
data.replace("", np.nan, inplace=True)
print(data.isnull().sum())

data.dropna(axis=0, inplace=True)
print("전체 샘플수 :", (len(data)))

# 길이 분포 출력
text_len = [len(s.split()) for s in data["Text"]]
summary_len = [len(s.split()) for s in data["Summary"]]

print("텍스트의 최소 길이 : {}".format(np.min(text_len)))
print("텍스트의 최대 길이 : {}".format(np.max(text_len)))
print("텍스트의 평균 길이 : {}".format(np.mean(text_len)))
print("요약의 최소 길이 : {}".format(np.min(summary_len)))
print("요약의 최대 길이 : {}".format(np.max(summary_len)))
print("요약의 평균 길이 : {}".format(np.mean(summary_len)))

plt.subplot(1, 2, 1)
plt.boxplot(summary_len)
plt.title("Summary")
plt.subplot(1, 2, 2)
plt.boxplot(text_len)
plt.title("Text")
plt.tight_layout()
plt.show()

plt.title("Summary")
plt.hist(summary_len, bins=40)
plt.xlabel("length of samples")
plt.ylabel("number of samples")
plt.show()

plt.title("Text")
plt.hist(text_len, bins=40)
plt.xlabel("length of samples")
plt.ylabel("number of samples")
plt.show()


text_max_len = 50
summary_max_len = 8


def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if len(s.split()) <= max_len:
            cnt = cnt + 1
    print("전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s" % (max_len, (cnt / len(nested_list))))


below_threshold_len(text_max_len, data["Text"])
below_threshold_len(summary_max_len, data["Summary"])

data = data[data["Text"].apply(lambda x: len(x.split()) <= text_max_len)]
data = data[data["Summary"].apply(lambda x: len(x.split()) <= summary_max_len)]
print("전체 샘플수 :", (len(data)))

data.head()

# 요약 데이터에는 시작 토큰과 종료 토큰을 추가한다.
data["decoder_input"] = data["Summary"].apply(lambda x: "sostoken " + x)
data["decoder_target"] = data["Summary"].apply(lambda x: x + " eostoken")
data.head()

encoder_input = np.array(data["Text"])
decoder_input = np.array(data["decoder_input"])
decoder_target = np.array(data["decoder_target"])

np.random.seed(seed=0)

indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)
print(indices)

encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]

n_of_val = int(len(encoder_input) * 0.2)
print(n_of_val)

encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]

print("훈련 데이터의 개수 :", len(encoder_input_train))
print("훈련 레이블의 개수 :", len(decoder_input_train))
print("테스트 데이터의 개수 :", len(encoder_input_test))
print("테스트 레이블의 개수 :", len(decoder_input_test))

src_tokenizer = Tokenizer()
src_tokenizer.fit_on_texts(encoder_input_train)

threshold = 7
total_cnt = len(src_tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in src_tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if value < threshold:
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print("단어 집합(vocabulary)의 크기 :", total_cnt)
print("등장 빈도가 %s번 이하인 희귀 단어의 수: %s" % (threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s" % (total_cnt - rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

src_vocab = 8000
src_tokenizer = Tokenizer(num_words=src_vocab)
src_tokenizer.fit_on_texts(encoder_input_train)

# 텍스트 시퀀스를 정수 시퀀스로 변환
encoder_input_train = src_tokenizer.texts_to_sequences(encoder_input_train)
encoder_input_test = src_tokenizer.texts_to_sequences(encoder_input_test)

print(encoder_input_train[:5])

tar_tokenizer = Tokenizer()
tar_tokenizer.fit_on_texts(decoder_input_train)

threshold = 6
total_cnt = len(tar_tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tar_tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if value < threshold:
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print("단어 집합(vocabulary)의 크기 :", total_cnt)
print("등장 빈도가 %s번 이하인 희귀 단어의 수: %s" % (threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s" % (total_cnt - rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

tar_vocab = 2000
tar_tokenizer = Tokenizer(num_words=tar_vocab)
tar_tokenizer.fit_on_texts(decoder_input_train)
tar_tokenizer.fit_on_texts(decoder_target_train)

# 텍스트 시퀀스를 정수 시퀀스로 변환
decoder_input_train = tar_tokenizer.texts_to_sequences(decoder_input_train)
decoder_target_train = tar_tokenizer.texts_to_sequences(decoder_target_train)

decoder_input_test = tar_tokenizer.texts_to_sequences(decoder_input_test)
decoder_target_test = tar_tokenizer.texts_to_sequences(decoder_target_test)

print(decoder_input_train[:5])
print(decoder_target_train[:5])
print(decoder_input_test[:5])
print(decoder_target_test[:5])

drop_train = [index for index, sentence in enumerate(decoder_input_train) if len(sentence) == 1]
drop_test = [index for index, sentence in enumerate(decoder_input_test) if len(sentence) == 1]

print("삭제할 훈련 데이터의 개수 :", len(drop_train))
print(len(drop_test))

encoder_input_train = np.delete(encoder_input_train, drop_train, axis=0)
decoder_input_train = np.delete(decoder_input_train, drop_train, axis=0)
decoder_target_train = np.delete(decoder_target_train, drop_train, axis=0)

encoder_input_test = np.delete(encoder_input_test, drop_test, axis=0)
decoder_input_test = np.delete(decoder_input_test, drop_test, axis=0)
decoder_target_test = np.delete(decoder_target_test, drop_test, axis=0)

print("훈련 데이터의 개수 :", len(encoder_input_train))
print("훈련 레이블의 개수 :", len(decoder_input_train))
print("테스트 데이터의 개수 :", len(encoder_input_test))
print("테스트 레이블의 개수 :", len(decoder_input_test))

encoder_input_train = pad_sequences(encoder_input_train, maxlen=text_max_len, padding="post")
encoder_input_test = pad_sequences(encoder_input_test, maxlen=text_max_len, padding="post")

decoder_input_train = pad_sequences(decoder_input_train, maxlen=summary_max_len, padding="post")
decoder_target_train = pad_sequences(decoder_target_train, maxlen=summary_max_len, padding="post")

decoder_input_test = pad_sequences(decoder_input_test, maxlen=summary_max_len, padding="post")
decoder_target_test = pad_sequences(decoder_target_test, maxlen=summary_max_len, padding="post")

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 128
hidden_size = 256

# 인코더
encoder_inputs = Input(shape=(text_max_len,))

# 인코더의 임베딩 층
enc_emb = Embedding(src_vocab, embedding_dim)(encoder_inputs)

# 인코더의 LSTM 1
encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# 인코더의 LSTM 2
encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# 인코더의 LSTM 3
encoder_lstm3 = LSTM(hidden_size, return_state=True, return_sequences=True, dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

# 디코더
decoder_inputs = Input(shape=(None,))

# 디코더의 임베딩 층
dec_emb_layer = Embedding(tar_vocab, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)

# 디코더의 LSTM
decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# 디코더의 출력층
decoder_softmax_layer = Dense(tar_vocab, activation="softmax")
decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs)

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()

# 바다나우 어텐션을 구현합니다.

import tensorflow as tf
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class AttentionLayer(Layer):

    # This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    # There are three sets of weights introduced W_a, U_a, and V_a

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(
            name="W_a",
            shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
            initializer="uniform",
            trainable=True,
        )
        self.U_a = self.add_weight(
            name="U_a",
            shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
            initializer="uniform",
            trainable=True,
        )
        self.V_a = self.add_weight(
            name="V_a",
            shape=tf.TensorShape((input_shape[0][2], 1)),
            initializer="uniform",
            trainable=True,
        )

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):

        inputs: [encoder_output_sequence, decoder_output_sequence]

        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print("encoder_out_seq>", encoder_out_seq.shape)
            print("decoder_out_seq>", decoder_out_seq.shape)

        def energy_step(inputs, states):
            # Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)

            assert_msg = "States must be an iterable. Got {} of type {}".format(
                states, type(states)
            )
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            # Some parameters required for shaping tensors
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            # Computing S.Wa where S=[s0, s1, ..., si]
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)

            # Computing hj.Ua
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print("Ua.h>", U_a_dot_h.shape)

            # tanh(S.Wa + hj.Ua)
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
            if verbose:
                print("Ws+Uh>", Ws_plus_Uh.shape)

            # softmax(va.tanh(S.Wa + hj.Ua))
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print("ei>", e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            # Step function for computing ci using ei

            assert_msg = "States must be an iterable. Got {} of type {}".format(
                states, type(states)
            )
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print("ci>", c_i.shape)
            return c_i, [c_i]

        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim

        # Computing energy outputs
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step,
            decoder_out_seq,
            [fake_state_e],
        )

        # Computing context vectors
        last_out, c_outputs, _ = K.rnn(
            context_step,
            e_outputs,
            [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        # Outputs produced by the layer
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1])),
        ]


# 어텐션 층(어텐션 함수)
attn_layer = AttentionLayer(name="attention_layer")
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# 어텐션의 결과와 디코더의 hidden state들을 연결
decoder_concat_input = Concatenate(axis=-1, name="concat_layer")([decoder_outputs, attn_out])

# 디코더의 출력층
decoder_softmax_layer = Dense(tar_vocab, activation="softmax")
decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=2)

history = model.fit(
    x=[encoder_input_train, decoder_input_train],
    y=decoder_target_train,
    validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
    batch_size=256,
    callbacks=[es],
    epochs=50,
)

src_index_to_word = src_tokenizer.index_word  # 원문 단어 집합에서 정수 -> 단어를 얻음
tar_word_to_index = tar_tokenizer.word_index  # 요약 단어 집합에서 단어 -> 정수를 얻음
tar_index_to_word = tar_tokenizer.index_word  # 요약 단어 집합에서 정수 -> 단어를 얻음

# 인코더 설계
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(hidden_size,))
decoder_state_input_c = Input(shape=(hidden_size,))

dec_emb2 = dec_emb_layer(decoder_inputs)
# 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용. 이는 뒤의 함수 decode_sequence()에 구현
# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태인 state_h와 state_c를 버리지 않음.
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c]
)

# 어텐션 함수
decoder_hidden_state_input = Input(shape=(text_max_len, hidden_size))
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name="concat")([decoder_outputs2, attn_out_inf])

# 디코더의 출력층
decoder_outputs2 = decoder_softmax_layer(decoder_inf_concat)

# 최종 디코더 모델
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2],
)


def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # <SOS>에 해당하는 토큰 생성
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tar_word_to_index["sostoken"]

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:  # stop_condition이 True가 될 때까지 루프 반복

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tar_index_to_word[sampled_token_index]

        if sampled_token != "eostoken":
            decoded_sentence += " " + sampled_token

        #  <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if sampled_token == "eostoken" or len(decoded_sentence.split()) >= (summary_max_len - 1):
            stop_condition = True

        # 길이가 1인 타겟 시퀀스를 업데이트
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # 상태를 업데이트 합니다.
        e_h, e_c = h, c

    return decoded_sentence


# 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2text(input_seq):
    temp = ""
    for i in input_seq:
        if i != 0:
            temp = temp + src_index_to_word[i] + " "
    return temp


# 요약문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2summary(input_seq):
    temp = ""
    for i in input_seq:
        if (i != 0 and i != tar_word_to_index["sostoken"]) and i != tar_word_to_index["eostoken"]:
            temp = temp + tar_index_to_word[i] + " "
    return temp


for i in range(500, 1000):
    print("원문 : ", seq2text(encoder_input_test[i]))
    print("실제 요약문 :", seq2summary(decoder_input_test[i]))
    print("예측 요약문 :", decode_sequence(encoder_input_test[i].reshape(1, text_max_len)))
    print("\n")

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="test")
plt.legend()
plt.show()

print(encoder_input_train[5])
print(decoder_input_train[5])
print(decoder_target_train[5])
