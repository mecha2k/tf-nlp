# !wget https://korquad.github.io/dataset/KorQuAD_v1.0_train.json -O KorQuAD_v1.0_train.json
# !wget https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json -O KorQuAD_v1.0_dev.json

import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizerFast
from transformers import TFBertModel
import tensorflow as tf


def read_squad(path):
    path = Path(path)
    with open(path, "rb") as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]
                for answer in qa["answers"]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


train_contexts, train_questions, train_answers = read_squad("../data/KorQuAD_v1.0_train.json")
val_contexts, val_questions, val_answers = read_squad("../data/KorQuAD_v1.0_dev.json")

print("훈련 데이터의 본문 개수 :", len(train_contexts))
print("훈련 데이터의 질문 개수 :", len(train_questions))
print("훈련 데이터의 답변 개수 :", len(train_answers))
print("테스트 데이터의 본문 개수 :", len(val_contexts))
print("테스트 데이터의 질문 개수 :", len(val_questions))
print("테스트 데이터의 답변 개수 :", len(val_answers))

print("첫번째 샘플의 본문")
print("-----------------")
print(train_contexts[0])

print("첫번째 샘플의 질문")
print("-----------------")
print(train_questions[0])

print("첫번째 샘플의 답변")
print("-----------------")
print(train_answers[0])


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        answer["text"] = answer["text"].rstrip()
        gold_text = answer["text"]
        start_idx = answer["answer_start"]
        end_idx = start_idx + len(gold_text)

        assert context[start_idx:end_idx] == gold_text, "end_index 계산에 에러가 있습니다."
        answer["answer_end"] = end_idx


add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

print("첫번째 샘플의 답변")
print("-----------------")
print(train_answers[0])
print(train_contexts[0][54])
print(train_contexts[0][55])
print(train_contexts[0][56])
print(train_contexts[0][54:57])

tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    deleting_list = []

    for i in tqdm(range(len(answers))):
        start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
        end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"] - 1))

        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
            deleting_list.append(i)

        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
            if i not in deleting_list:
                deleting_list.append(i)

    encodings.update({"start_positions": start_positions, "end_positions": end_positions})
    return deleting_list


deleting_list_for_train = add_token_positions(train_encodings, train_answers)
deleting_list_for_test = add_token_positions(val_encodings, val_answers)


def delete_samples(encodings, deleting_list):
    input_ids = np.delete(np.array(encodings["input_ids"]), deleting_list, axis=0)
    attention_masks = np.delete(np.array(encodings["attention_mask"]), deleting_list, axis=0)
    start_positions = np.delete(np.array(encodings["start_positions"]), deleting_list, axis=0)
    end_positions = np.delete(np.array(encodings["end_positions"]), deleting_list, axis=0)

    X_data = [input_ids, attention_masks]
    y_data = [start_positions, end_positions]

    return X_data, y_data


X_train, y_train = delete_samples(train_encodings, deleting_list_for_train)
X_test, y_test = delete_samples(val_encodings, deleting_list_for_test)

# TPU 작동을 위한 코드 TPU 작동을 위한 코드
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
#     tpu="grpc://" + os.environ["COLAB_TPU_ADDR"]
# )
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.TPUStrategy(resolver)


class TFBertForQuestionAnswering(tf.keras.Model):
    def __init__(self, model_name):
        super(TFBertForQuestionAnswering, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.qa_outputs = tf.keras.layers.Dense(
            2, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), name="qa_outputs"
        )
        self.softmax = tf.keras.layers.Activation(tf.keras.activations.softmax)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)

        # start_logits = (batch_size, sequence_length,)
        # end_logits = (batch_size, sequence_length,)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        start_probs = self.softmax(start_logits)
        end_probs = self.softmax(end_logits)

        return start_probs, end_probs


# 신규
# with strategy.scope():
model = TFBertForQuestionAnswering("klue/bert-base")
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer=optimizer, loss=loss)

history = model.fit(
    X_train,
    y_train,
    epochs=3,
    verbose=1,
    batch_size=16,
)


def predict_test_data_by_idx(idx):
    context = tokenizer.decode(X_test[0][idx]).split("[SEP] ")[0]
    question = tokenizer.decode(X_test[0][idx]).split("[SEP] ")[1]
    print("본문 :", context)
    print("질문 :", question)
    answer_encoded = X_test[0][idx][y_test[0][idx] : y_test[1][idx] + 1]
    print("정답 :", tokenizer.decode(answer_encoded))
    output = model([tf.constant(X_test[0][idx])[None, :], tf.constant(X_test[1][idx])[None, :]])
    start = tf.math.argmax(tf.squeeze(output[0]))
    end = tf.math.argmax(tf.squeeze(output[1])) + 1
    answer_encoded = X_test[0][idx][start:end]
    print("예측 :", tokenizer.decode(answer_encoded))
    print("----------------------------------------")


for i in range(0, 100):
    predict_test_data_by_idx(i)
