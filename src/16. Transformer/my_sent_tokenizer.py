import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import string, re, random

from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    Layer,
    TextVectorization,
    Dropout,
    Dense,
    Embedding,
    MultiHeadAttention,
    LayerNormalization,
)
from tensorflow.keras.utils import plot_model


def normalize_sentence(data):
    sentences = []
    for sentence in data:
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        sentences.append(sentence)
    return sentences


def text_vectorization(max_len, vocab_size, batch_size):
    train_data = pd.read_csv("../data/ChatBotData.csv")
    print(f"chatbot samples: {len(train_data):,}")

    questions = normalize_sentence(train_data["Q"])
    answers = normalize_sentence(train_data["A"])

    text_pairs = []
    for question, answer in zip(questions, answers):
        answer = "[start] " + answer + " [end]"
        text_pairs.append((question, answer))
    print(random.choice(text_pairs))

    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    valid_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]

    strip_chars = string.punctuation
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    print(f"[{re.escape(strip_chars)}]")

    def custom_standardization(input_string):
        # lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(input_string, f"[{re.escape(strip_chars)}]", "")

    print(custom_standardization("[start] 정말 다행이에요 . [end]"))

    source_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
    )
    target_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len + 1,
        standardize=custom_standardization,
    )

    questions, answers = zip(*train_pairs)
    source_vectorization.adapt(questions + answers)
    target_vectorization.adapt(questions + answers)
    vocabulary = source_vectorization.get_vocabulary()
    idx_to_word = dict(zip(range(len(vocabulary)), vocabulary))
    word_to_idx = dict(zip(vocabulary, range(len(vocabulary))))
    print(len(word_to_idx), len(idx_to_word))

    def format_dataset(source, target):
        source = source_vectorization(source)
        target = target_vectorization(target)
        return (
            {
                "source": source,
                "target": target[:, :-1],
            },
            target[:, 1:],
        )

    def make_dataset(pairs):
        sources, targets = zip(*pairs)
        sources = list(sources)
        targets = list(targets)
        dataset = tf.data.Dataset.from_tensor_slices((sources, targets)).batch(batch_size)
        dataset = dataset.map(format_dataset, num_parallel_calls=4)
        return dataset.shuffle(2048).prefetch(16)

    train_ds = make_dataset(train_pairs)
    valid_ds = make_dataset(valid_pairs)

    for sources, targets in train_ds.take(1):
        print(f"sources['source'].shape: {sources['source'].shape}")
        print(f"sources['target'].shape: {sources['target'].shape}")
        print(f"targets.shape: {targets.shape}")

        encoded_sent = sources["target"][0].numpy()
        decoded_sent = [idx_to_word[idx] for idx in encoded_sent]
        print(encoded_sent)
        print(decoded_sent)

    for i in range(10):
        print(idx_to_word[i])

    return (
        train_ds,
        valid_ds,
        idx_to_word,
        word_to_idx,
        test_pairs,
        source_vectorization,
        target_vectorization,
    )


if __name__ == "__main__":
    text_vectorization(max_len=20, vocab_size=15000, batch_size=128)
