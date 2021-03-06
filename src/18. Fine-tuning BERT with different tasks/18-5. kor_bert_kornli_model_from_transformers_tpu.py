import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn import preprocessing
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# https://github.com/kakaobrain/KorNLUDatasets
# koNLI (Natural Language Inference, 자연어추론), koSTS (Semantic Textual Similarity, 의미적 텍스트 유사성)

train_snli = pd.read_csv("../data/snli_1.0_train.ko.tsv", sep="\t", quoting=3)
train_xnli = pd.read_csv("../data/multinli.train.ko.tsv", sep="\t", quoting=3)
val_data = pd.read_csv("../data/xnli.dev.ko.tsv", sep="\t", quoting=3)
test_data = pd.read_csv("../data/xnli.test.ko.tsv", sep="\t", quoting=3)
# print(train_snli.head())
# print(train_xnli.head())

# 결합 후 섞기
train_data = train_snli.append(train_xnli)
train_data = train_data.sample(frac=1)
# print(train_data.head())
# print(val_data.head())
# print(test_data.head())


def drop_na_and_duplciates(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


# 결측값 및 중복 샘플 제거
train_data = drop_na_and_duplciates(train_data)
val_data = drop_na_and_duplciates(val_data)
test_data = drop_na_and_duplciates(test_data)

train_data = train_data[:5000]
val_data = val_data[:1000]
test_data = test_data[:1000]
print(train_data)
print(val_data)
print(test_data)


def convert_examples_to_features(sent_list1, sent_list2, max_seq_len, tokenizer):
    input_ids, attention_masks, token_type_ids = [], [], []

    for sent1, sent2 in tqdm(zip(sent_list1, sent_list2), total=len(sent_list1)):
        encoding_result = tokenizer.encode_plus(
            sent1, sent2, max_length=max_seq_len, padding="max_length", truncation=True
        )

        assert (
            len(encoding_result["input_ids"]) == max_seq_len
        ), "Error with input length {} vs {}".format(len(encoding_result["input_ids"]), max_seq_len)
        assert (
            len(encoding_result["attention_mask"]) == max_seq_len
        ), "Error with attention mask length {} vs {}".format(
            len(encoding_result["attention_mask"]), max_seq_len
        )
        assert (
            len(encoding_result["token_type_ids"]) == max_seq_len
        ), "Error with token type length {} vs {}".format(
            len(encoding_result["token_type_ids"]), max_seq_len
        )

        input_ids.append(encoding_result["input_ids"])
        attention_masks.append(encoding_result["attention_mask"])
        token_type_ids.append(encoding_result["token_type_ids"])

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    return input_ids, attention_masks, token_type_ids


max_seq_len = 128

tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

X_train = convert_examples_to_features(
    train_data["sentence1"], train_data["sentence2"], max_seq_len=max_seq_len, tokenizer=tokenizer
)

# 최대 길이: 128
nid = 1126
input_id = X_train[0][nid]
attention_mask = X_train[1][nid]
token_type_id = X_train[2][nid]

print("단어에 대한 정수 인코딩 :", input_id)
print("어텐션 마스크 :", attention_mask)
print("세그먼트 인코딩 :", token_type_id)
print("각 인코딩의 길이 :", len(input_id))
print("정수 인코딩 복원 :", tokenizer.decode(input_id))

X_val = convert_examples_to_features(
    val_data["sentence1"], val_data["sentence2"], max_seq_len=max_seq_len, tokenizer=tokenizer
)

# 최대 길이: 128
input_id = X_val[0][0]
attention_mask = X_val[1][0]
token_type_id = X_val[2][0]

print("단어에 대한 정수 인코딩 :", input_id)
print("어텐션 마스크 :", attention_mask)
print("세그먼트 인코딩 :", token_type_id)
print("각 인코딩의 길이 :", len(input_id))
print("정수 인코딩 복원 :", tokenizer.decode(input_id))

X_test = convert_examples_to_features(
    test_data["sentence1"], test_data["sentence2"], max_seq_len=max_seq_len, tokenizer=tokenizer
)

train_label = train_data["gold_label"].tolist()
val_label = val_data["gold_label"].tolist()
test_label = test_data["gold_label"].tolist()

idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(train_label)

y_train = idx_encode.transform(train_label)  # 주어진 고유한 정수로 변환
y_val = idx_encode.transform(val_label)  # 고유한 정수로 변환
y_test = idx_encode.transform(test_label)  # 고유한 정수로 변환

label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
idx_label = {value: key for key, value in label_idx.items()}
print(label_idx)
print(idx_label)


# TPU 작동을 위한 코드
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
#     tpu="grpc://" + os.environ["COLAB_TPU_ADDR"]
# )
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)
#
# with strategy.scope():

model = TFBertForSequenceClassification.from_pretrained(
    "klue/bert-base", num_labels=3, from_pt=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=["accuracy"])
early_stopping = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=2)

model.fit(
    X_train,
    y_train,
    epochs=2,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test, batch_size=1024)[1]))
