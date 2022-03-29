import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
plt.style.use("seaborn")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


lines = pd.read_table("../data/fra-eng/fra.txt", names=["src", "tar", "lic"], sep="\t")
del lines["lic"]

lines = lines[:2002]
lines["tar"] = lines["tar"].apply(lambda x: "\t" + x + "\n")
print(lines.head())


def make_vocab(lines):
    vocab = set()
    for line in lines:
        for char in line:
            vocab.add(char)
    return sorted(list(vocab))


src_vocab = make_vocab(lines["src"])
tar_vocab = make_vocab(lines["tar"])

src_vocab_size = len(src_vocab) + 1
tar_vocab_size = len(tar_vocab) + 1

src_to_idx = dict((word, i + 1) for i, word in enumerate(src_vocab))
tar_to_idx = dict((word, i + 1) for i, word in enumerate(tar_vocab))
print(list(src_to_idx.keys())[:5])

idx_to_src = dict([(value, key) for key, value in src_to_idx.items()])
idx_to_tar = dict([(value, key) for key, value in tar_to_idx.items()])

encoder_inputs = []
for line in lines["src"]:
    encoder_inputs.append([src_to_idx[char] for char in line])

decoder_inputs = []
for line in lines["tar"]:
    decoder_inputs.append([tar_to_idx[char] for char in line])

decoder_targets = []
for line in lines["tar"]:
    decoder_targets.append([tar_to_idx[char] for char in line if char != "\t"])
for idx in encoder_inputs[100]:
    print(idx_to_src[idx], end="")
print("\n", decoder_targets[:3])


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

max_src_len = max([len(x) for x in encoder_inputs])
max_tar_len = max([len(x) for x in decoder_inputs])

encoder_inputs = pad_sequences(encoder_inputs, maxlen=max_src_len, padding="post")
decoder_inputs = pad_sequences(decoder_inputs, maxlen=max_tar_len, padding="post")
decoder_targets = pad_sequences(decoder_targets, maxlen=max_tar_len, padding="post")

encoder_in_data = to_categorical(encoder_inputs)
decoder_in_data = to_categorical(decoder_inputs)
decoder_tar_data = to_categorical(decoder_targets)


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import plot_model

encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]
print(encoder_inputs.shape)
print(type(state_h))

decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_softmax = Dense(tar_vocab_size, activation="softmax")
decoder_outputs = decoder_softmax(decoder_outputs)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
plot_model(model, to_file="images/02-seq2seq.png", show_shapes=True)
model.summary()

model.fit(
    x=[encoder_in_data, decoder_in_data],
    y=decoder_tar_data,
    batch_size=128,
    epochs=1,
    validation_split=0.2,
)

encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))

decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)

decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax(decoder_outputs)

decoder_model = Model(
    inputs=[decoder_inputs] + decoder_state_inputs,
    outputs=[decoder_outputs] + decoder_states,
)
encoder_model.summary()
decoder_model.summary()


def decode_sequences(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros(shape=(1, 1, tar_vocab_size))
    target_seq[0, 0, tar_to_idx["\t"]] = 1

    stop = False
    decoded_sentence = ""

    while not stop:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = idx_to_tar[sampled_token_index]

        decoded_sentence += sampled_char
        if sampled_char == "\n" or len(decoded_sentence) > max_tar_len:
            stop = True

        target_seq = np.zeros(shape=(1, 1, tar_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1

        states_value = [h, c]

    return decoded_sentence


for seq_index in [10, 100, 1000]:
    input_seq = encoder_in_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequences(input_seq)
    print("-" * 60)
    print("Input sentence:", lines["src"][seq_index])
    print("Target sentence:", lines["tar"][seq_index][2 : len(lines["tar"][seq_index]) - 1])
    print("Decode sentence:", decoded_sentence[1 : len(decoded_sentence) - 1])
