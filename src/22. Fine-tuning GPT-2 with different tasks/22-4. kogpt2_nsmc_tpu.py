import transformers
import pandas as pd
import numpy as np
import urllib.request
import os
from tqdm import tqdm
import tensorflow as tf
from transformers import AutoTokenizer, TFGPT2Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data = pd.read_table("../data/ratings_train.txt")
test_data = pd.read_table("../data/ratings_test.txt")

print("훈련용 리뷰 개수 :", len(train_data))  # 훈련용 리뷰 개수 출력
print("테스트용 리뷰 개수 :", len(test_data))  # 테스트용 리뷰 개수 출력

train_data = train_data.dropna(how="any")  # Null 값이 존재하는 행 제거
train_data = train_data.reset_index(drop=True)
print(train_data.isnull().values.any())  # Null 값이 존재하는지 확인

test_data = test_data.dropna(how="any")  # Null 값이 존재하는 행 제거
test_data = test_data.reset_index(drop=True)
print(test_data.isnull().values.any())  # Null 값이 존재하는지 확인
print(len(train_data))
print(len(test_data))

tokenizer = AutoTokenizer.from_pretrained(
    "skt/kogpt2-base-v2", bos_token="</s>", eos_token="</s>", pad_token="<pad>"
)

print(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))
print(tokenizer.tokenize("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))

tokenizer.decode(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))

for elem in tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"):
    print(tokenizer.decode(elem))

print(tokenizer.tokenize("전율을 일으키는 영화. 다시 보고싶은 영화"))
print(tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화"))

for elem in tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화"):
    print(tokenizer.decode(elem))

for elem in tokenizer.encode("happy birthday~!"):
    print(tokenizer.decode(elem))

print(tokenizer.decode(3))

max_seq_len = 128

encoded_result = tokenizer.encode(
    "전율을 일으키는 영화. 다시 보고싶은 영화", max_length=max_seq_len, pad_to_max_length=True
)
print(encoded_result)
print("길이 :", len(encoded_result))


def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):

    input_ids, data_labels = [], []

    for example, label in tqdm(zip(examples, labels), total=len(examples)):

        bos_token = [tokenizer.bos_token]
        eos_token = [tokenizer.eos_token]
        tokens = bos_token + tokenizer.tokenize(example) + eos_token
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_id = pad_sequences(
            [input_id], maxlen=max_seq_len, value=tokenizer.pad_token_id, padding="post"
        )[0]

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(
            len(input_id), max_seq_len
        )
        input_ids.append(input_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    data_labels = np.asarray(data_labels, dtype=np.int32)

    return input_ids, data_labels


train_X, train_y = convert_examples_to_features(
    train_data["document"], train_data["label"], max_seq_len=max_seq_len, tokenizer=tokenizer
)

test_X, test_y = convert_examples_to_features(
    test_data["document"], test_data["label"], max_seq_len=max_seq_len, tokenizer=tokenizer
)

# 최대 길이: 128
input_id = train_X[0]
label = train_y[0]

print("단어에 대한 정수 인코딩 :", input_id)
print("각 인코딩의 길이 :", len(input_id))
print("정수 인코딩 복원 :", tokenizer.decode(input_id))
print("레이블 :", label)

model = TFGPT2Model.from_pretrained("skt/kogpt2-base-v2", from_pt=True)

max_seq_len = 128

input_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
outputs = model([input_ids_layer])

print(outputs)
print(outputs[0])
print(outputs[1])
print(outputs[0][:, -1])


class TFGPT2ForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name):
        super(TFGPT2ForSequenceClassification, self).__init__()
        self.gpt = TFGPT2Model.from_pretrained(model_name, from_pt=True)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.classifier = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
            activation="sigmoid",
            name="classifier",
        )

    def call(self, inputs):
        outputs = self.gpt(input_ids=inputs)
        cls_token = outputs[0][:, -1]
        cls_token = self.dropout(cls_token)
        prediction = self.classifier(cls_token)

        return prediction


# TPU 사용법 : https://wikidocs.net/119990
# TPU 작동을 위한 코드
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
#     tpu="grpc://" + os.environ["COLAB_TPU_ADDR"]
# )
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)
# with strategy.scope():

model = TFGPT2ForSequenceClassification("skt/kogpt2-base-v2")
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model.fit(train_X, train_y, epochs=2, batch_size=32, validation_split=0.2)

results = model.evaluate(test_X, test_y, batch_size=1024)
print("test loss, test acc: ", results)


def sentiment_predict(new_sentence):

    bos_token = [tokenizer.bos_token]
    eos_token = [tokenizer.eos_token]
    tokens = bos_token + tokenizer.tokenize(new_sentence) + eos_token
    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_id = pad_sequences(
        [input_id], maxlen=max_seq_len, value=tokenizer.pad_token_id, padding="post"
    )[0]
    input_id = np.array([input_id])
    score = model.predict(input_id)[0][0]

    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))


sentiment_predict("보던거라 계속보고있는데 전개도 느리고 주인공인 은희는 한두컷 나오면서 소극적인모습에 ")

sentiment_predict(
    "스토리는 확실히 실망이였지만 배우들 연기력이 대박이였다 특히 이제훈 연기 정말 ... 이 배우들로 이렇게밖에 만들지 못한 영화는 아쉽지만 배우들 연기력과 사운드는 정말 빛났던 영화. 기대하고 극장에서 보면 많이 실망했겠지만 평점보고 기대없이 집에서 편하게 보면 괜찮아요. 이제훈님 연기력은 최고인 것 같습니다"
)

sentiment_predict("남친이 이 영화를 보고 헤어지자고한 영화. 자유롭게 살고 싶다고 한다. 내가 무슨 나비를 잡은 덫마냥 나에겐 다시 보고싶지 않은 영화.")

sentiment_predict("이 영화 존잼입니다 대박")

sentiment_predict("이 영화 개꿀잼 ㅋㅋㅋ")

sentiment_predict("이 영화 핵노잼 ㅠㅠ")

sentiment_predict("이딴게 영화냐 ㅉㅉ")

sentiment_predict("감독 뭐하는 놈이냐?")

sentiment_predict("와 개쩐다 정말 세계관 최강자들의 영화다")
