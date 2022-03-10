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


def tfds_text_encoder(questions, answers):
    # 서브워드텍스트인코더를 사용하여 질문과 답변을 모두 포함한 단어 집합(Vocabulary) 생성
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2 ** 13
    )

    # 시작 토큰과 종료 토큰에 대한 정수 부여.
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
    VOCAB_SIZE = tokenizer.vocab_size + 2

    print("시작 토큰 번호 :", START_TOKEN)
    print("종료 토큰 번호 :", END_TOKEN)
    print("단어 집합의 크기 :", VOCAB_SIZE)

    # 서브워드텍스트인코더 토크나이저의 .encode()와 decode() 테스트해보기
    # 임의의 입력 문장을 sample_string에 저장
    sample_string = questions[20]

    # encode() : 텍스트 시퀀스 --> 정수 시퀀스
    tokenized_string = tokenizer.encode(sample_string)
    print("정수 인코딩 후의 문장 {}".format(tokenized_string))

    # decode() : 정수 시퀀스 --> 텍스트 시퀀스
    original_string = tokenizer.decode(tokenized_string)
    print("기존 문장: {}".format(original_string))

    # 각 정수는 각 단어와 어떻게 mapping되는지 병렬로 출력
    # 서브워드텍스트인코더는 의미있는 단위의 서브워드로 토크나이징한다. 띄어쓰기 단위 X 형태소 분석 단위 X
    for ts in tokenized_string:
        print("{} ----> {}".format(ts, tokenizer.decode([ts])))

    # 최대 길이를 40으로 정의
    MAX_LENGTH = 40

    # 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
    def tokenize_and_filter(inputs, outputs):
        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(inputs, outputs):
            # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
            sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

        # 패딩
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=MAX_LENGTH, padding="post"
        )
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=MAX_LENGTH, padding="post"
        )

        return tokenized_inputs, tokenized_outputs

    questions, answers = tokenize_and_filter(questions, answers)

    print("질문 데이터의 크기(shape) :", questions.shape)
    print("답변 데이터의 크기(shape) :", answers.shape)

    # 0번째 샘플을 임의로 출력
    print(questions[0])
    print(answers[0])

    print("단어 집합의 크기(Vocab size): {}".format(VOCAB_SIZE))
    print("전체 샘플의 수(Number of samples): {}".format(len(questions)))

    # 텐서플로우 dataset을 이용하여 셔플(shuffle)을 수행하되, 배치 크기로 데이터를 묶는다.
    # 또한 이 과정에서 교사 강요(teacher forcing)을 사용하기 위해서 디코더의 입력과 실제값 시퀀스를 구성한다.
    BATCH_SIZE = 64
    BUFFER_SIZE = 20000

    # 디코더의 실제값 시퀀스에서는 시작 토큰을 제거해야 한다.
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"inputs": questions, "dec_inputs": answers[:, :-1]},  # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
            {"outputs": answers[:, 1:]},  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
        )
    )

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # 임의의 샘플에 대해서 [:, :-1]과 [:, 1:]이 어떤 의미를 가지는지 테스트해본다.
    print(answers[0])  # 기존 샘플
    print(answers[:1][:, :-1])  # 마지막 패딩 토큰 제거하면서 길이가 39가 된다.
    print(answers[:1][:, 1:])  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다. 길이는 역시 39가 된다.

    return dataset


def normalize_sentence(data):
    sentences = []
    for sentence in data:
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        sentences.append(sentence)
    return sentences


if __name__ == "__main__":
    train_data = pd.read_csv("../data/ChatBotData.csv")
    print(f"chatbot samples: {len(train_data):,}")

    questions = normalize_sentence(train_data["Q"])
    answers = normalize_sentence(train_data["A"])

    # dataset = tfds_text_encoder(questions, answers)
    # for encoder, decoder in dataset.take(1):
    #     print(encoder["inputs"][0])
    #     print(encoder["dec_inputs"][0])
    #     print(decoder["outputs"][0])

    text_pairs = []
    for question, answer in zip(questions, answers):
        answer = "[start]" + answer + "[end]"
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

    def custom_standardization(input_string):
        lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

    vocab_size = 15000
    sequence_length = 40
    batch_size = 128

    source_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    target_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
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

        encoded_sent = sources["source"][0].numpy()
        decoded_sent = [idx_to_word[idx] for idx in encoded_sent]
        print(encoded_sent)
        print(decoded_sent)
