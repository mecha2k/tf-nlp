import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import os, re

import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Dense
from keras.initializers import TruncatedNormal
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import TFGPT2LMHeadModel, TFGPT2Model

print(transformers.__version__)

np.random.seed(seed=42)
tf.random.set_seed(seed=42)


class GPT2Model(keras.Model):
    def __init__(self, path):
        super().__init__()
        self.gpt2 = TFGPT2LMHeadModel.from_pretrained(path)

    def call(self, inputs):
        return self.gpt2(inputs)[0]


model_path = "../data/gpt_ckpt"
model = GPT2Model(model_path)

epochs = 10
batch_size = 16
max_len = 30

tokenizer_path = "../data/gpt_ckpt/gpt2_kor_tokenizer.spiece"

tokenizer = SentencepieceTokenizer(tokenizer_path)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(
    tokenizer_path,
    mask_token=None,
    sep_token=None,
    cls_token=None,
    unknown_token="<unk>",
    padding_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)


def tf_top_filtering(logits, top_k=0, top_p=0.0, filter_value=99999):
    _logits = logits.numpy()
    top_k = min(top_k, logits.shape[-1])
    if top_k > 0:
        idx_to_remove = logits < tf.math.top_k(logits, top_k)[0][..., -1, None]
        _logits[idx_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits = tf.sort(logits, direction="DESCENDING")
        sorted_indices = tf.argsort(logits, direction="DESCENDING")
        cumulative_probs = tf.math.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = tf.concat([[False], sorted_indices_to_remove[..., :-1]], axis=0)
        indices_to_remove = sorted_indices[sorted_indices_to_remove].numpy().tolist()
        _logits[indices_to_remove] = filter_value

    return tf.constant([_logits])


def generate_sentence(words, model, max_step=100, greedy=False, top_k=0, top_p=0.0):
    sentence = words
    tokenized_sentence = tokenizer(sentence)

    for _ in range(max_step):
        input_ids = tf.constant([vocab[vocab.bos_token]] + vocab[tokenized_sentence])[None, :]
        outputs = model(input_ids)[:, -1, :]
        if greedy is True:
            generated = vocab.to_tokens(tf.argmax(outputs, axis=-1).numpy().tolist()[0])
        else:
            output_logit = tf_top_filtering(outputs[0], top_k=top_k, top_p=top_p)
            generated = vocab.to_tokens(tf.random.categorical(output_logit, 1).numpy().tolist()[0])[
                0
            ]
        if generated == "</s>":
            break
        sentence += generated.replace("▁", " ")
        tokenized_sentence = tokenizer(sentence)

    return sentence


# print(generate_sentence("방금", model, greedy=True))
# print(generate_sentence("오늘", model, greedy=True))
# print(generate_sentence("오늘", model, greedy=False))
# print(generate_sentence("언제나", model, top_k=0, top_p=0.15))
# print(generate_sentence("일부", model, greedy=True))

sentences = [sent[:-1] for sent in open("../data/finetune_data.txt", encoding="utf-8").readlines()]

inputs, outputs = [], []
for sentence in sentences:
    tokens = [vocab[vocab.bos_token]] + vocab[tokenizer(sentence)] + [vocab[vocab.eos_token]]
    inputs.append(tokens[:-1])
    outputs.append(tokens[1:])

inputs = np.array(
    pad_sequences(inputs, maxlen=max_len, value=vocab[vocab.padding_token]), dtype=np.int64
)
outputs = np.array(
    pad_sequences(outputs, maxlen=max_len, value=vocab[vocab.padding_token]), dtype=np.int64
)

loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
train_accuracy = keras.metrics.SparseCategoricalAccuracy(name="accuracy")


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, vocab[vocab.padding_token]))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def accuracy_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, vocab[vocab.padding_token]))
    mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis=-1)
    pred *= mask
    acc = train_accuracy(real, pred)
    return tf.reduce_mean(acc)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss_function,
    metrics=[accuracy_function],
)

# history = model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, validation_split=0.1)

model_path = "../data/tf_gpt2_finetuned"
if not os.path.exists(model_path):
    os.makedirs(model_path)

model.gpt2.save_pretrained(model_path)
model = GPT2Model(model_path)

print(generate_sentence("방금", model, greedy=True))
# print(generate_sentence("언제나", model, greedy=True))
# print(generate_sentence("오늘", model, greedy=True))


epochs = 3
batch_size = 32
max_len = 40

tokenizer_path = "../data/gpt_ckpt/gpt2_kor_tokenizer.spiece"

tokenizer = SentencepieceTokenizer(tokenizer_path)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(
    tokenizer_path,
    mask_token=None,
    sep_token="<sep>",
    cls_token=None,
    unknown_token="<unk>",
    padding_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

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


def make_review_data(data_df):
    texts, labels = [], []
    for data, label in data_df[["document", "label"]].values:
        tokenized_sentence = vocab[tokenizer(data)]
        tokens = [vocab[vocab.bos_token]]
        tokens += pad_sequences(
            [tokenized_sentence], maxlen=max_len, value=vocab[vocab.padding_token], padding="post"
        ).tolist()[0]
        tokens += [vocab[vocab.eos_token]]
        texts.append(tokens)
        labels.append(label)
    texts = np.array(texts, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)
    return texts, labels


train_texts, train_labels = make_review_data(train_ds)
test_texts, test_labels = make_review_data(test_ds)


class TFGPT2Classification(keras.Model):
    def __init__(self, path, num_class):
        super().__init__()
        self.gpt2 = TFGPT2Model.from_pretrained(path)
        self.num_class = num_class

        self.dropout = Dropout(self.gpt2.config.summary_first_dropout)
        self.classifier = Dense(
            self.num_class,
            kernel_initializer=TruncatedNormal(stddev=self.gpt2.config.initializer_range),
        )

    def call(self, inputs):
        outputs = self.gpt2(inputs)[0][:, -1]
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        return outputs


model_path = "../data/gpt_ckpt"
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=bool)

model = TFGPT2Classification(path=model_path, num_class=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

file_path = model_path + "/weights.h5"
callbacks = [
    EarlyStopping(monitor="val_accuracy", min_delta=0.0001, patience=2),
    ModelCheckpoint(
        filepath=file_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    ),
]

history = model.fit(
    train_texts,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=callbacks,
)

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
steps = range(1, len(accuracy) + 1)
plt.figure(figsize=(10, 6))
plt.plot(steps, accuracy, "bo", label="Training accuracy")
plt.plot(steps, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig("images/gpt2_accuracy", dpi=300)
plt.figure(figsize=(10, 6))
plt.plot(steps, loss, "bo", label="Training loss")
plt.plot(steps, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig("images/gpt2_loss", dpi=300)

model.load_weights(file_path)
model.evaluate(test_texts, test_labels, batch_size=1024)
