import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import imdb

vocab_size = 20000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print("훈련용 리뷰 개수 :", len(X_train))
print("테스트용 리뷰 개수 :", len(X_test))
print(X_train[0])
print(y_train[0])
print("훈련용 리뷰의 평규 길이: {}".format(np.mean(list(map(len, X_train)), dtype=int)))
print("테스트용 리뷰의 평균 길이: {}".format(np.mean(list(map(len, X_test)), dtype=int)))

max_len = 400
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

print("x_train의 크기(shape) :", X_train.shape)
print("x_test의 크기(shape) :", X_test.shape)

# 2. 모델 설계하기
embedding_dim = 64

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
# 모든 단어 벡터의 평균을 구한다.
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation="sigmoid"))

es = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=4)
mc = ModelCheckpoint(
    "../data/embedding_average_model.h5",
    monitor="val_acc",
    mode="max",
    verbose=0,
    save_best_only=True,
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(X_train, y_train, batch_size=32, epochs=10, callbacks=[es, mc], validation_split=0.2)

loaded_model = load_model("../data/embedding_average_model.h5")
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
