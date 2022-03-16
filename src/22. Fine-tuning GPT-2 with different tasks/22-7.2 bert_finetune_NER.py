import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import os, re
import json, copy

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Dense
from keras.initializers import TruncatedNormal
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import TFGPT2LMHeadModel, TFGPT2Model, BertTokenizer
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from tqdm import tqdm


epochs = 3
batch_size = 32
max_len = 120

np.random.seed(42)
tf.random.set_seed(42)


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

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased", cache_dir="../data/bert_ckpt"
)

pad_token_id = tokenizer.pad_token_id
pad_token_label_id = 0
cls_token_label_id = 0
sep_token_label_id = 0


def bert_tokenizer(sent, MAX_LEN):
    encoded_dict = tokenizer.encode_plus(
        text=sent,
        truncation=True,
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_attention_mask=True,
    )

    input_id = encoded_dict["input_ids"]
    attention_mask = encoded_dict["attention_mask"]
    token_type_id = encoded_dict["token_type_ids"]

    return input_id, attention_mask, token_type_id


def convert_label(words, labels_idx, ner_label, max_seq_len):
    tokens, label_ids = [], []
    for word, slot_label in zip(words, labels_idx):
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]
        tokens.extend(word_tokens)
        # 슬롯 레이블 값이 Begin이면 I로 추가
        if int(slot_label) in ner_label:
            label_ids.extend([int(slot_label)] + [int(slot_label) + 1] * (len(word_tokens) - 1))
        else:
            label_ids.extend([int(slot_label)] * len(word_tokens))

    # [CLS] and [SEP] 설정
    special_tokens_count = 2
    if len(label_ids) > max_seq_len - special_tokens_count:
        label_ids = label_ids[: (max_seq_len - special_tokens_count)]

    # [SEP] 토큰 추가
    label_ids += [sep_token_label_id]

    # [CLS] 토큰 추가
    label_ids = [cls_token_label_id] + label_ids
    padding_length = max_seq_len - len(label_ids)
    label_ids = label_ids + ([pad_token_label_id] * padding_length)
    return label_ids


ner_label = [labels.index(label) for label in labels if "B" in label]
ner_label_string = [labels[index] for index in ner_label]
print(ner_label)
print(ner_label_string)


def create_inputs_targets(df):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    label_list = []
    for i, data in enumerate(df[["sentence", "label"]].values):
        sentence, labels = data
        words = sentence.split()
        labels = labels.split()
        labels_idx = []
        for label in labels:
            labels_idx.append(labels.index(label) if label in labels else labels.index("UNK"))
        assert len(words) == len(labels_idx)

        input_id, attention_mask, token_type_id = bert_tokenizer(sentence, MAX_LEN)
        convert_label_id = convert_label(words, labels_idx, ner_label, MAX_LEN)
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        label_list.append(convert_label_id)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    label_list = np.asarray(label_list, dtype=int)  # 레이블 토크나이징 리스트
    inputs = (input_ids, attention_masks, token_type_ids)

    return inputs, label_list


# train_inputs, train_labels = create_inputs_targets(train_ner_df)
# test_inputs, test_labels = create_inputs_targets(test_ner_df)


class TFBertNERClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super().__init__()
        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            num_class,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                self.bert.config.initializer_range
            ),
        )

    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        # outputs : sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)
        return logits


#
# ner_model = TFBertNERClassifier(
#     model_name="bert-base-multilingual-cased", dir_path="bert_ckpt", num_class=len(labels)
# )
#
#
# # In[42]:
#
#
# def compute_loss(labels, logits):
#     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
#         from_logits=True, reduction=tf.keras.losses.Reduction.NONE
#     )
#
#     # 0의 레이블 값은 손실 값을 계산할 때 제외
#     active_loss = tf.reshape(labels, (-1,)) != 0
#
#     reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
#
#     labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
#
#     return loss_fn(labels, reduced_logits)
#
#
# # In[43]:
#
#
# class F1Metrics(tf.keras.callbacks.Callback):
#     def __init__(self, x_eval, y_eval):
#         self.x_eval = x_eval
#         self.y_eval = y_eval
#
#     def compute_f1_pre_rec(self, labels, preds):
#
#         return {
#             "precision": precision_score(labels, preds, suffix=True),
#             "recall": recall_score(labels, preds, suffix=True),
#             "f1": f1_score(labels, preds, suffix=True),
#         }
#
#     def show_report(self, labels, preds):
#         return classification_report(labels, preds, suffix=True)
#
#     def on_epoch_end(self, epoch, logs=None):
#
#         results = {}
#
#         pred = self.model.predict(self.x_eval)
#         label = self.y_eval
#         pred_argmax = np.argmax(pred, axis=2)
#
#         slot_label_map = {i: label for i, label in enumerate(labels)}
#
#         out_label_list = [[] for _ in range(label.shape[0])]
#         preds_list = [[] for _ in range(label.shape[0])]
#
#         for i in range(label.shape[0]):
#             for j in range(label.shape[1]):
#                 if label[i, j] != 0:
#                     out_label_list[i].append(slot_label_map[label[i][j]])
#                     preds_list[i].append(slot_label_map[pred_argmax[i][j]])
#
#         result = self.compute_f1_pre_rec(out_label_list, preds_list)
#         results.update(result)
#
#         print("********")
#         print("F1 Score")
#         for key in sorted(results.keys()):
#             print("{}, {:.4f}".format(key, results[key]))
#         print("\n" + self.show_report(out_label_list, preds_list))
#         print("********")
#
#
# f1_score_callback = F1Metrics(test_inputs, test_labels)
#
#
# # In[44]:
#
#
# # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
# optimizer = tf.keras.optimizers.Adam(3e-5)
# # ner_model.compile(optimizer=optimizer, loss=compute_loss, run_eagerly=True)
# ner_model.compile(optimizer=optimizer, loss=compute_loss)
#
#
# # In[ ]:
#
#
# model_name = "tf2_bert_ner"
#
# checkpoint_path = os.path.join(DATA_OUT_PATH, model_name, "weights.h5")
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # Create path if exists
# if os.path.exists(checkpoint_dir):
#     print("{} -- Folder already exists \n".format(checkpoint_dir))
# else:
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     print("{} -- Folder create complete \n".format(checkpoint_dir))
#
# cp_callback = ModelCheckpoint(
#     checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True
# )
#
# history = ner_model.fit(
#     train_inputs,
#     train_labels,
#     batch_size=BATCH_SIZE,
#     epochs=NUM_EPOCHS,
#     callbacks=[cp_callback, f1_score_callback],
# )
#
# print(history.history)
#
#
# # In[34]:
#
#
# plot_graphs(history, "loss")
