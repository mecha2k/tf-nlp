import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import string, re, random

from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences

from konlpy.tag import Okt, Mecab
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm

okt = Okt()
mecab = Mecab(dicpath="C:/mecab/mecab-ko-dic")
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

# fmt: off
stopwords = [
    "의", "가", "이", "은", "들", "는", "좀", "잘", "걍", "과", "도", "를", "으로", "자", "에", "와", "한", "하다"
]
# fmt: on


def morphs_text_okt(data):
    sentences = []
    for sent in tqdm(data):
        sentence = okt.morphs(sent, stem=False)
        sentence = " ".join(sentence)
        sentences.append(sentence)
    return sentences


def morphs_text_mecab(data):
    sentences = []
    for sent in tqdm(data):
        sentence = mecab.morphs(sent)
        sentence = " ".join(sentence)
        sentences.append(sentence)
    return sentences


def text_vectorization(max_len, vocab_size, batch_size):
    train_data = pd.read_csv("../data/ChatBotData.csv")
    print(f"chatbot samples: {len(train_data):,}")

    # questions = normalize_sentence(train_data["Q"])
    # answers = normalize_sentence(train_data["A"])

    # questions = morphs_text_okt(train_data["Q"])
    # answers = morphs_text_okt(train_data["A"])

    questions = morphs_text_mecab(train_data["Q"])
    answers = morphs_text_mecab(train_data["A"])

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
    print(strip_chars)
    print(f"[{re.escape(strip_chars)}]")

    def custom_standardization(input_string):
        lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

    source_vectorization = TextVectorization(
        max_tokens=vocab_size, output_mode="int", output_sequence_length=max_len
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
    source_vocab = source_vectorization.get_vocabulary()
    src_idx_to_word = dict(zip(range(len(source_vocab)), source_vocab))
    src_word_to_idx = dict(zip(source_vocab, range(len(source_vocab))))
    target_vocab = target_vectorization.get_vocabulary()
    tar_idx_to_word = dict(zip(range(len(target_vocab)), target_vocab))
    tar_word_to_idx = dict(zip(target_vocab, range(len(target_vocab))))

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
        decoded_sent = [tar_idx_to_word[idx] for idx in encoded_sent]
        print(encoded_sent)
        print(decoded_sent)

    print(f"source vocabulary : {len(source_vocab)}")
    print(f"target vocabulary : {len(target_vocab)}")
    for i in range(10):
        print(tar_idx_to_word[i], end=" ")

    return (
        train_ds,
        valid_ds,
        tar_idx_to_word,
        tar_word_to_idx,
        test_pairs,
        source_vectorization,
        target_vectorization,
    )


def normalize_sentence(data):
    sentences = []
    for sentence in data:
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        sentences.append(sentence)
    return sentences


def text_vectorization_tfds(max_len, vocab_size, batch_size):
    train_data = pd.read_csv("../data/ChatBotData.csv")
    print(f"chatbot samples: {len(train_data):,}")

    questions = normalize_sentence(train_data["Q"])
    answers = normalize_sentence(train_data["A"])

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**13
    )
    print(tokenizer.subwords[:10])

    print(questions[20])
    encode_question = tokenizer.encode(questions[20])
    print(f"Tokenized sample question(encode): {encode_question}")
    print(f"Tokenized sample question(decode): {tokenizer.decode(encode_question)}")

    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    vocab_size = tokenizer.vocab_size + 2

    inputs, outputs = [], []
    for (sent1, sent2) in zip(questions, answers):
        sentence1 = start_token + tokenizer.encode(sent1) + end_token
        sentence2 = start_token + tokenizer.encode(sent2) + end_token
        inputs.append(sentence1)
        outputs.append(sentence2)

    questions = keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_len, padding="post")
    answers = tf.keras.preprocessing.sequence.pad_sequences(
        outputs, maxlen=max_len + 1, padding="post"
    )
    print(start_token, end_token)
    print("vocabulary size :", vocab_size)
    print("questions (shape) :", questions.shape)
    print("answers (shape) :", answers.shape)

    text_pairs = []
    for question, answer in zip(questions, answers):
        text_pairs.append((question, answer))
    print(random.choice(text_pairs))

    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    valid_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]

    def format_dataset(source, target):
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
        return dataset.shuffle(20480).prefetch(16)

    train_ds = make_dataset(train_pairs)
    valid_ds = make_dataset(valid_pairs)

    for sources, targets in train_ds.take(1):
        print(f"sources['source'].shape: {sources['source'].shape}")
        print(f"sources['target'].shape: {sources['target'].shape}")
        print(f"targets.shape: {targets.shape}")

    return train_ds, valid_ds, test_pairs, vocab_size, tokenizer, start_token, end_token


def char_vectorization(batch_size=64):
    train_data = pd.read_csv("../data/ChatBotData.csv")
    print(f"chatbot samples: {len(train_data):,}")

    buffer_size = 20000

    questions = normalize_sentence(train_data["Q"])
    answers = normalize_sentence(train_data["A"])
    answers = ["\t" + answer + "\n" for answer in answers]
    print(answers[0])

    vocab_set = set()
    for line in questions + answers:
        for char in line:
            vocab_set.add(char)

    vocab_size = len(vocab_set) + 2
    print("문장의 char 집합 :", vocab_size)

    vocab_set = sorted(list(vocab_set))
    print(vocab_set[:10])

    char_to_idx = dict([(word, i + 1) for i, word in enumerate(vocab_set)])
    idx_to_char = dict([(i + 1, word) for i, word in enumerate(vocab_set)])

    def char_encoding(inputs):
        sentences = []
        for sentence in inputs:
            encoded_char = []
            for char in sentence:
                encoded_char.append(char_to_idx[char])
            sentences.append(encoded_char)
        return sentences

    sources = char_encoding(questions)
    targets = char_encoding(answers)
    print("source 문장의 정수 인코딩 :", sources[:2])
    print("target 문장의 정수 인코딩 :", targets[:2])
    print("Decoding [0] : ", str([idx_to_char[word] for word in sources[0]]))

    max_src_len = max([len(line) for line in sources])
    max_tar_len = max([len(line) for line in targets])
    max_len = max(max_src_len, max_tar_len) + 10
    print("문장의 최대 길이 :", max_len)

    print("vocabulary size :", vocab_size)
    print("questions (shape) :", np.asarray(sources, dtype=object).shape)
    print("answers (shape) :", np.asarray(targets, dtype=object).shape)

    sources = pad_sequences(sources, maxlen=max_len, padding="post")
    targets = pad_sequences(targets, maxlen=max_len + 1, padding="post")

    text_pairs = []
    for source, target in zip(sources, targets):
        text_pairs.append((source, target))

    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    valid_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]

    def format_dataset(source, target):
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
        return dataset.shuffle(buffer_size=buffer_size).prefetch(16)

    train_ds = make_dataset(train_pairs)
    valid_ds = make_dataset(valid_pairs)

    for sources, targets in train_ds.take(1):
        print(f"sources['source'].shape: {sources['source'].shape}")
        print(f"sources['target'].shape: {sources['target'].shape}")
        print(f"targets.shape: {targets.shape}")

    return train_ds, valid_ds, test_pairs, vocab_size, max_len


if __name__ == "__main__":
    # char_vectorization(batch_size=64)
    text_vectorization_tfds(max_len=40, vocab_size=15000, batch_size=64)
    # text_vectorization(max_len=20, vocab_size=15000, batch_size=128)

    # train_data = pd.read_csv("../data/ChatBotData.csv")
    # print(f"chatbot samples: {len(train_data):,}")
    #
    # with open("../data/chatbot_questions.txt", "w", encoding="utf8") as f:
    #     f.write("\n".join(train_data["Q"]))
    #
    # print(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))
    # print(tokenizer.tokenize("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))
    # print(tokenizer.decode(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역")))
    #
    # from tokenizers import (
    #     ByteLevelBPETokenizer,
    #     CharBPETokenizer,
    #     SentencePieceBPETokenizer,
    #     BertWordPieceTokenizer,
    # )
    #
    # bert_wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=True, strip_accents=True)
    # tokenizer.train(
    #     files="../data/chatbot_questions.txt",
    #     vocab_size=30000,
    #     limit_alphabet=6000,
    #     min_frequency=5,
    # )
    # tokenizer.save_model("../data/bert_tokenizer")
