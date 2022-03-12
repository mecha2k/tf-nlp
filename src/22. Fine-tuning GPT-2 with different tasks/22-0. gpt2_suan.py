# import numpy as np
import numpy as np
import transformers
import os

import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer
from nltk.tokenize import sent_tokenize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFGPT2LMHeadModel

print(transformers.__version__)


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
