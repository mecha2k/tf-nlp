import numpy as np
import urllib.request
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences

# urllib.request.urlretrieve("http://www.gutenberg.org/files/11/11-0.txt", filename="11-0.txt")
f = open("11-0.txt", "rb")
sentences = []
for sentence in f:  # 데이터를 한 줄씩 읽는다.
    sentence = sentence.strip()  # strip()을 통해 \r, \n을 제거한다.
    sentence = sentence.lower()  # 소문자화.
    sentence = sentence.decode("ascii", "ignore")  # \xe2\x80\x99 등과 같은 바이트 열 제거
    if len(sentence) > 0:
        sentences.append(sentence)
f.close()
print(sentences[:5])

total_data = " ".join(sentences)
print("문자열의 길이 또는 총 글자의 개수: %d" % len(total_data))
print(total_data[:200])

char_vocab = sorted(list(set(total_data)))
vocab_size = len(char_vocab)
print("글자 집합의 크기 : {}".format(vocab_size))

char_to_index = dict((char, index) for index, char in enumerate(char_vocab))  # 글자에 고유한 정수 인덱스 부여
print(char_to_index)

index_to_char = {}
for key, value in char_to_index.items():
    index_to_char[value] = key

seq_length = 60  # 문장의 길이를 60으로 한다.
n_samples = int(np.floor((len(total_data) - 1) / seq_length))  # 문자열을 60등분한다. 그러면 즉, 총 샘플의 개수
print("문장 샘플의 수 : {}".format(n_samples))

train_X = []
train_y = []

for i in range(n_samples):
    # 0:60 -> 60:120 -> 120:180로 loop를 돌면서 문장 샘플을 1개씩 픽한다.
    X_sample = total_data[i * seq_length : (i + 1) * seq_length]

    # 정수 인코딩
    X_encoded = [char_to_index[c] for c in X_sample]
    train_X.append(X_encoded)

    # 오른쪽으로 1칸 쉬프트
    y_sample = total_data[i * seq_length + 1 : (i + 1) * seq_length + 1]
    y_encoded = [char_to_index[c] for c in y_sample]
    train_y.append(y_encoded)

print("X 데이터의 첫번째 샘플 :", train_X[0])
print("y 데이터의 첫번째 샘플 :", train_y[0])
print("-" * 50)
print("X 데이터의 첫번째 샘플 디코딩 :", [index_to_char[i] for i in train_X[0]])
print("y 데이터의 첫번째 샘플 디코딩 :", [index_to_char[i] for i in train_y[0]])

print(train_X[1])
print(train_y[1])

train_X = to_categorical(train_X)
train_y = to_categorical(train_y)

print("train_X의 크기(shape) : {}".format(train_X.shape))  # 원-핫 인코딩
print("train_y의 크기(shape) : {}".format(train_y.shape))  # 원-핫 인코딩


hidden_units = 256

model = Sequential()
model.add(LSTM(hidden_units, input_shape=(None, train_X.shape[2]), return_sequences=True))
model.add(LSTM(hidden_units, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_X, train_y, epochs=80, verbose=0)


def sentence_generation(model, length):
    # 글자에 대한 랜덤 인덱스 생성
    ix = [np.random.randint(vocab_size)]

    # 랜덤 익덱스로부터 글자 생성
    y_char = [index_to_char[ix[-1]]]
    print(ix[-1], "번 글자", y_char[-1], "로 예측을 시작!")

    # (1, length, 55) 크기의 X 생성. 즉, LSTM의 입력 시퀀스 생성
    X = np.zeros((1, length, vocab_size))

    for i in range(length):
        # X[0][i][예측한 글자의 인덱스] = 1, 즉, 예측 글자를 다음 입력 시퀀스에 추가
        X[0][i][ix[-1]] = 1
        print(index_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, : i + 1, :])[0], 1)
        y_char.append(index_to_char[ix[-1]])
    return "".join(y_char)


result = sentence_generation(model, 100)
print(result)

# 2. 글자 단위 RNN(Char RNN)으로 텍스트 생성하기
# 이번에는 다 대 일(many-to-one) 구조의 RNN을 글자 단위로 학습시키고, 텍스트 생성을 해보겠습니다.

raw_text = """
I get on with life as a programmer,
I like to contemplate beer.
But when I start to daydream,
My mind turns straight to wine.

Do I love wine more than beer?

I like to use words about beer.
But when I stop my talking,
My mind turns straight to wine.

I hate bugs and errors.
But I just think back to wine,
And I'm happy once again.

I like to hang out with programming and deep learning.
But when left alone,
My mind turns straight to wine.
"""

tokens = raw_text.split()
raw_text = " ".join(tokens)
print(raw_text)

# 중복을 제거한 글자 집합 생성
char_vocab = sorted(list(set(raw_text)))
print(char_vocab)

vocab_size = len(char_vocab)
print("글자 집합의 크기 : {}".format(vocab_size))

char_to_index = dict((c, i) for i, c in enumerate(char_vocab))  # 글자에 고유한 정수 인덱스 부여
print(char_to_index)

length = 11
sequences = []
for i in range(length, len(raw_text)):
    seq = raw_text[i - length : i]  # 길이 11의 문자열을 지속적으로 만든다.
    sequences.append(seq)
print("총 훈련 샘플의 수: %d" % len(sequences))
print(sequences[:10])

encoded_sequences = []
for sequence in sequences:  # 전체 데이터에서 문장 샘플을 1개씩 꺼낸다.
    encoded_sequence = [char_to_index[char] for char in sequence]  # 문장 샘플에서 각 글자에 대해서 정수 인코딩을 수행.
    encoded_sequences.append(encoded_sequence)
print(encoded_sequences[:5])

encoded_sequences = np.array(encoded_sequences)
X_data = encoded_sequences[:, :-1]

# 맨 마지막 위치의 글자를 분리
y_data = encoded_sequences[:, -1]

print(X_data[:5])
print(y_data[:5])

# 원-핫 인코딩
X_data_one_hot = [to_categorical(encoded, num_classes=vocab_size) for encoded in X_data]
X_data_one_hot = np.array(X_data_one_hot)
y_data_one_hot = to_categorical(y_data, num_classes=vocab_size)
print(X_data_one_hot.shape)


hidden_units = 64

model = Sequential()
model.add(LSTM(hidden_units, input_shape=(X_data_one_hot.shape[1], X_data_one_hot.shape[2])))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_data_one_hot, y_data_one_hot, epochs=100, verbose=0)


def sentence_generation(model, char_to_index, seq_length, seed_text, n):

    # 초기 시퀀스
    init_text = seed_text
    sentence = ""

    for _ in range(n):
        encoded = [char_to_index[char] for char in seed_text]  # 현재 시퀀스에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=seq_length, padding="pre")  # 데이터에 대한 패딩
        encoded = to_categorical(encoded, num_classes=len(char_to_index))

        # 입력한 X(현재 시퀀스)에 대해서 y를 예측하고 y(예측한 글자)를 result에 저장.
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        char = None
        for char, index in char_to_index.items():
            if index == result:
                break

        # 현재 시퀀스 + 예측 글자를 현재 시퀀스로 변경
        seed_text = seed_text + char

        # 예측 글자를 문장에 저장
        sentence = sentence + char

    sentence = init_text + sentence
    return sentence


print(sentence_generation(model, char_to_index, 10, "I get on w", 80))
