import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BertTokenizer, TFBertModel
from transformers import logging

logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

np.random.seed(42)
tf.random.set_seed(42)

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased", cache_dir="../data/bert_ckpt", do_lower_case=False
)

test_sentence = "안녕하세요, 반갑습니다."
encode = tokenizer.encode(test_sentence)
token_print = [tokenizer.decode(token) for token in encode]
print(encode)
print(tokenizer.decode(encode))
print(token_print)

test_sentence = "Hello world !!"
encode = tokenizer.encode(test_sentence)
token_print = [tokenizer.decode(token) for token in encode]
print(encode)
print(tokenizer.decode(encode))
print(token_print)

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)


# Korean Movie Review Classification
train_ds = pd.read_table("../data/ratings_train.txt")
test_ds = pd.read_table("../data/ratings_test.txt")


def clean_sentence(sentence):
    sentence = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 \s]", "", sentence)
    return re.sub("^ +", "", sentence)


train_ds = train_ds.drop_duplicates(subset=["document"]).dropna(how="any")
test_ds = test_ds.drop_duplicates(subset=["document"]).dropna(how="any")
print(train_ds.head())
print(len(train_ds))
train_ds["document"] = train_ds["document"].apply(clean_sentence)
test_ds["document"] = test_ds["document"].apply(clean_sentence)
train_ds["document"] = train_ds["document"].replace("", np.nan)
test_ds["document"] = test_ds["document"].replace("", np.nan)
train_ds = train_ds.dropna(how="any")
test_ds = test_ds.dropna(how="any")
print(train_ds.head())
print(len(train_ds))
print(len(test_ds))

train_ds_length = max([len(sent) for sent in train_ds["document"]])
print(train_ds_length)
train_ds_len = [sent for sent in train_ds["document"]]
print(train_ds_len[0])

# for fast test
train_ds = train_ds[:1000]
test_ds = test_ds[:100]


epochs = 1
batch_size = 128
max_len = 40


def make_review_data(df):
    input_ids, attention_masks, token_type_ids, labels = [], [], [], []
    for sentence, label in tqdm(zip(df["document"], df["label"])):
        encoded_sentence = tokenizer.encode_plus(
            text=sentence,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        input_ids.append(encoded_sentence["input_ids"])
        attention_masks.append((encoded_sentence["attention_mask"]))
        token_type_ids.append(encoded_sentence["token_type_ids"])

    input_ids = np.array(input_ids, dtype=np.int32)
    attention_masks = np.array(attention_masks, dtype=np.int32)
    token_type_ids = np.array(attention_masks, dtype=np.int32)
    labels = np.array(df["label"], dtype=np.int32)
    return (input_ids, attention_masks, token_type_ids), labels


x_train, y_train = make_review_data(train_ds)
x_test, y_test = make_review_data(test_ds)


class TFBertClassifier(tf.keras.Model):
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
        # outputs: sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits


model = TFBertClassifier(
    model_name="bert-base-multilingual-cased", dir_path="../data/bert_ckpt", num_class=2
)

optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


callbacks = [
    EarlyStopping(monitor="val_accuracy", min_delta=0.0001, patience=2),
    ModelCheckpoint(
        "../data/bert_nsmc_weights.h5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    ),
]

history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=callbacks,
)

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"], "")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["loss", "val_loss"])
plt.savefig("images/bert_nsmc", dpi=300)


results = model.evaluate(x_test, y_test, batch_size=1024)
print("test loss, test acc: ", results)
