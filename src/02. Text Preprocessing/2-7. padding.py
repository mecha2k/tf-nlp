import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

preprocessed_sentences = [
    ["barber", "person"],
    ["barber", "good", "person"],
    ["barber", "huge", "person"],
    ["knew", "secret"],
    ["secret", "kept", "huge", "secret"],
    ["huge", "secret"],
    ["barber", "kept", "word"],
    ["barber", "kept", "word"],
    ["barber", "kept", "secret"],
    ["keeping", "keeping", "huge", "secret", "driving", "barber", "crazy"],
    ["barber", "went", "huge", "mountain"],
]

tokenizer = Tokenizer()
# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성.
tokenizer.fit_on_texts(preprocessed_sentences)

encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)

max_len = max(len(item) for item in encoded)
print(max_len)

for sentence in encoded:  # 각 문장에 대해서
    while len(sentence) < max_len:  # max_len보다 작으면
        sentence.append(0)

padded_np = np.array(encoded)
print(padded_np)

from tensorflow.keras.preprocessing.sequence import pad_sequences

encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)

padded = pad_sequences(encoded)
print(padded)

padded = pad_sequences(encoded, padding="post")
print(padded)

print((padded == padded_np).all())

padded = pad_sequences(encoded, padding="post", maxlen=5)

padded = pad_sequences(encoded, padding="post", truncating="post", maxlen=5)
print(padded)


# 단어 집합의 크기보다 1 큰 숫자를 사용
last_value = len(tokenizer.word_index) + 1
print(last_value)

padded = pad_sequences(encoded, padding="post", value=last_value)
print(padded)
