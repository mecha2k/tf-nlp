import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BertTokenizer, TFBertModel


epochs = 3
batch_size = 32
max_len = 24 * 2

np.random.seed(42)
tf.random.set_seed(42)


## KorNLI Dataset
train_data_snli = pd.read_csv("../data/snli_1.0_train.ko.tsv", header=0, delimiter="\t", quoting=3)
train_data_xnli = pd.read_csv("../data/multinli.train.ko.tsv", header=0, delimiter="\t", quoting=3)
dev_data_xnli = pd.read_csv("../data/xnli.dev.ko.tsv", header=0, delimiter="\t", quoting=3)
#
# train_data_snli_xnli = train_data_snli.append(train_data_xnli)
# train_data_snli_xnli = train_data_snli_xnli.dropna()
# train_data_snli_xnli = train_data_snli_xnli.reset_index()
#
# dev_data_xnli = dev_data_xnli.dropna()
#
# print("Total # dataset: train - {}, dev - {}".format(len(train_data_snli_xnli), len(dev_data_xnli)))
#
# # In[5]:
#
#
# # Bert Tokenizer
#
# # 참조: https://huggingface.co/transformers/main_classes/tokenizer.html?highlight=encode_plus#transformers.PreTrainedTokenizer.encode_plus
#
# tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir='bert_ckpt', do_lower_case=False)
#
#
# def bert_tokenizer_v2(sent1, sent2, MAX_LEN):
#     # For Two setenece input
#
#     encoded_dict = tokenizer.encode_plus(
#         text=sent1,
#         text_pair=sent2,
#         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#         max_length=MAX_LEN,  # Pad & truncate all sentences.
#         pad_to_max_length=True,
#         return_attention_mask=True  # Construct attn. masks.
#
#     )
#
#     input_id = encoded_dict['input_ids']
#     attention_mask = encoded_dict[
#         'attention_mask']  # And its attention mask (simply differentiates padding from non-padding).
#     token_type_id = encoded_dict['token_type_ids']  # differentiate two sentences
#
#     return input_id, attention_mask, token_type_id
#
#
# # In[13]:
#
#
# input_ids = []
# attention_masks = []
# token_type_ids = []
#
# for sent1, sent2 in zip(train_data_snli_xnli['sentence1'], train_data_snli_xnli['sentence2']):
#     try:
#         input_id, attention_mask, token_type_id = bert_tokenizer_v2(sent1, sent2, MAX_LEN)
#
#         input_ids.append(input_id)
#         attention_masks.append(attention_mask)
#         token_type_ids.append(token_type_id)
#     except Exception as e:
#         print(e)
#         print(sent1, sent2)
#         pass
#
# train_snli_xnli_input_ids = np.array(input_ids, dtype=int)
# train_snli_xnli_attention_masks = np.array(attention_masks, dtype=int)
# train_snli_xnli_type_ids = np.array(token_type_ids, dtype=int)
# train_snli_xnli_inputs = (train_snli_xnli_input_ids, train_snli_xnli_attention_masks, train_snli_xnli_type_ids)
#
# # In[18]:
#
#
# input_id = train_snli_xnli_input_ids[2]
# attention_mask = train_snli_xnli_attention_masks[2]
# token_type_id = train_snli_xnli_type_ids[2]
#
# print(input_id)
# print(attention_mask)
# print(token_type_id)
# print(tokenizer.decode(input_id))
#
# # # DEV SET Preprocessing
#
# # In[6]:
#
#
# # 토크나이저를 제외하고는 5장에서 처리한 방식과 유사하게 접근
# input_ids = []
# attention_masks = []
# token_type_ids = []
#
# for sent1, sent2 in zip(dev_data_xnli['sentence1'], dev_data_xnli['sentence2']):
#     try:
#         input_id, attention_mask, token_type_id = bert_tokenizer_v2(sent1, sent2, MAX_LEN)
#
#         input_ids.append(input_id)
#         attention_masks.append(attention_mask)
#         token_type_ids.append(token_type_id)
#     except Exception as e:
#         print(e)
#         print(sent1, sent2)
#         pass
#
# dev_xnli_input_ids = np.array(input_ids, dtype=int)
# dev_xnli_attention_masks = np.array(attention_masks, dtype=int)
# dev_xnli_type_ids = np.array(token_type_ids, dtype=int)
# dev_xnli_inputs = (dev_xnli_input_ids, dev_xnli_attention_masks, dev_xnli_type_ids)
#
# # In[ ]:
#
#
# # Label을 Netural, Contradiction, Entailment 에서 숫자 형으로 변경한다.
# label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2}
#
#
# def convert_int(label):
#     num_label = label_dict[label]
#     return num_label
#
#
# train_data_snli_xnli["gold_label_int"] = train_data_snli_xnli["gold_label"].apply(convert_int)
# train_data_labels = np.array(train_data_snli_xnli['gold_label_int'], dtype=int)
#
# dev_data_xnli["gold_label_int"] = dev_data_xnli["gold_label"].apply(convert_int)
# dev_data_labels = np.array(dev_data_xnli['gold_label_int'], dtype=int)
#
# print("# train labels: {}, #dev labels: {}".format(len(train_data_labels), len(dev_data_labels)))
#
#
# # In[ ]:
#
#
# class TFBertClassifier(tf.keras.Model):
#     def __init__(self, model_name, dir_path, num_class):
#         super(TFBertClassifier, self).__init__()
#
#         self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
#         self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
#         self.classifier = tf.keras.layers.Dense(num_class,
#                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(
#                                                     self.bert.config.initializer_range),
#                                                 name="classifier")
#
#     def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
#         # outputs 값: # sequence_output, pooled_output, (hidden_states), (attentions)
#         outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output, training=training)
#         logits = self.classifier(pooled_output)
#
#         return logits
#
#
# cls_model = TFBertClassifier(model_name='bert-base-multilingual-cased',
#                              dir_path='bert_ckpt',
#                              num_class=3)
#
# # In[ ]:
#
#
# # 학습 준비하기
# optimizer = tf.keras.optimizers.Adam(3e-5)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
# cls_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
#
# # In[ ]:
#
#
# # 학습 진행하기
# model_name = "tf2_KorNLI"
#
# # overfitting을 막기 위한 ealrystop 추가
# earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)
# checkpoint_path = os.path.join(DATA_OUT_PATH, model_name, 'weights.h5')
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
#     checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
#
# # 학습과 eval 시작
# history = cls_model.fit(train_snli_xnli_inputs, train_data_labels, epochs=NUM_EPOCHS,
#                         validation_data=(dev_xnli_inputs, dev_data_labels),
#                         batch_size=BATCH_SIZE, callbacks=[earlystop_callback, cp_callback])
#
# # steps_for_epoch
# print(history.history)
#
# # In[ ]:
#
#
# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'loss')
#
# # # KorNLI Test dataset
#
# # In[ ]:
#
#
# # Load Test dataset
# TEST_XNLI_DF = os.path.join(DATA_IN_PATH, 'KorNLI', 'xnli.test.ko.tsv')
#
# test_data_xnli = pd.read_csv(TEST_XNLI_DF, header=0, delimiter='\t', quoting=3)
# test_data_xnli = test_data_xnli.dropna()
# test_data_xnli.head()
#
# # In[ ]:
#
#
# # Test set도 똑같은 방법으로 구성한다.
#
# input_ids = []
# attention_masks = []
# token_type_ids = []
#
# for sent1, sent2 in zip(test_data_xnli['sentence1'], test_data_xnli['sentence2']):
#
#     try:
#         input_id, attention_mask, token_type_id = bert_tokenizer_v2(sent1, sent2, MAX_LEN)
#
#         input_ids.append(input_id)
#         attention_masks.append(attention_mask)
#         token_type_ids.append(token_type_id)
#     except Exception as e:
#         print(e)
#         print(sent1, sent2)
#         pass
#
# test_xnli_input_ids = np.array(input_ids, dtype=int)
# test_xnli_attention_masks = np.array(attention_masks, dtype=int)
# test_xnli_type_ids = np.array(token_type_ids, dtype=int)
# test_xnli_inputs = (test_xnli_input_ids, test_xnli_attention_masks, test_xnli_type_ids)
#
# # In[ ]:
#
#
# test_data_xnli["gold_label_int"] = test_data_xnli["gold_label"].apply(convert_int)
# test_data_xnli_labels = np.array(test_data_xnli['gold_label_int'], dtype=int)
#
# print("# sents: {}, # labels: {}".format(len(test_xnli_input_ids), len(test_data_xnli_labels)))
#
# # In[ ]:
#
#
# results = cls_model.evaluate(test_xnli_inputs, test_data_xnli_labels, batch_size=512)
# print("test loss, test acc: ", results)
#
