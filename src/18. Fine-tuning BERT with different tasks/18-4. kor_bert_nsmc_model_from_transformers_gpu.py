import pandas as pd
import tensorflow as tf

from transformers import BertTokenizerFast, TFBertForSequenceClassification
from transformers import TextClassificationPipeline


def train_bert_model():
    train_data = pd.read_table("../data/ratings_train.txt")
    test_data = pd.read_table("../data/ratings_test.txt")

    train_data.drop_duplicates(subset=["document"], inplace=True)
    test_data.drop_duplicates(subset=["document"], inplace=True)

    train_data = train_data.dropna(how="any")  # Null 값이 존재하는 행 제거
    test_data = test_data.dropna(how="any")

    print(train_data["label"].value_counts())
    print(train_data.isna().values.any())  # Null 값이 존재하는지 확인

    train_data = train_data[:15000]
    test_data = test_data[:5000]

    print("총 훈련 샘플의 수 :", len(train_data))
    print("총 테스트 샘플의 수 :", len(test_data))

    # 토크나이저를 이용한 정수 인코딩
    # 이미 학습해놓은 모델을 사용한다고 하면
    # 1. 토크나이저 (이 모델이 만들어졌을 당시에 '사과' 라는 단어가 36번이었다. 정보를 알기 위해)
    # 2. 모델
    # 이 두 가지를 로드해야 합니다.
    #
    # from transformers import BertTokenizerFast
    # tokenizer = BertTokenizerFast.from_pretrained("사용하고자 하는 모델 이름")
    #
    # from transformers import TFBertForSequenceClassification
    # model = TFBertForSequenceClassification.from_pretrained
    #         ("사용하고자 하는 모델 이름", num_labels=클래스의 수(분류할 종류 개수), from_pt=True)

    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")

    X_train_list = train_data["document"].tolist()
    X_test_list = test_data["document"].tolist()
    y_train = train_data["label"].tolist()
    y_test = test_data["label"].tolist()

    max_seq_len = 128

    X_train = tokenizer(X_train_list, max_length=max_seq_len, padding="max_length")
    # max_length=max_seq_len, padding="max_length", truncation=True
    X_test = tokenizer(X_test_list, max_length=max_seq_len, padding="max_length")

    # type_ids는 지금 풀고자 하는 문제에서 문장의 종류의 개수를 의미하는데, 일반적으로 두 개 이상의 문장을 가지고 푸는 문제일 경우에는
    # [0, 0, 0, 0, 1, 1, 1, 1] 이런 식의 값이 들어가지만 네이버 영화 리뷰는 문장 1개를 보고 푸는 문제라서 [0, 0, 0, 0, 0, 0, 0, 0]이 들어간다.
    print(X_train[0].tokens)
    print(X_train[0].ids)
    print(X_train[0].type_ids)
    print(len(X_train[0].ids))
    print(X_train[0])
    print(X_train[0].attention_mask)

    # 데이터셋 생성 및 모델 학습
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test), y_test))

    model = TFBertForSequenceClassification.from_pretrained(
        "klue/bert-base", num_labels=2, from_pt=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=["accuracy"])
    print(model.hf_compute_loss)

    callback_earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", min_delta=0.001, patience=2
    )

    model.fit(
        train_dataset.shuffle(10000).batch(32),
        epochs=5,
        batch_size=64,
        validation_data=val_dataset.shuffle(10000).batch(64),
        callbacks=[callback_earlystop],
    )
    model.evaluate(val_dataset.batch(512))

    model.save_pretrained("../data/nsmc_model/bert-base")
    tokenizer.save_pretrained("../data/nsmc_model/bert-base")


def load_bert_model():
    loaded_tokenizer = BertTokenizerFast.from_pretrained("../data/nsmc_model/bert-base")
    loaded_model = TFBertForSequenceClassification.from_pretrained("../data/nsmc_model/bert-base")

    text_classifier = TextClassificationPipeline(
        tokenizer=loaded_tokenizer, model=loaded_model, framework="tf", return_all_scores=True
    )

    print(text_classifier("뭐야 이 평점들은.... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아")[0])
    print(text_classifier("오랜만에 평점 로긴했네ㅋㅋ 킹왕짱 쌈뽕한 영화를 만났습니다 강렬하게 육쾌함")[0])


if __name__ == "__main__":
    train_bert_model()
    load_bert_model()
