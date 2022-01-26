import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=vocab_size)
print(X_train[:5])
print(y_train[:5])

max_len = 200
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

print("X_train의 크기(shape) :", X_train.shape)
print("X_test의 크기(shape) :", X_test.shape)


model = Sequential()
model.add(Embedding(vocab_size, 256))
model.add(Dropout(0.3))
model.add(Conv1D(256, 3, padding="valid", activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.summary()

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
mc = ModelCheckpoint(
    "../data/11_3_best_model.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[es, mc]
)

loaded_model = load_model("../data/11_3_best_model.h5")
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

epochs = range(1, len(history.history["acc"]) + 1)
plt.plot(epochs, history.history["loss"])
plt.plot(epochs, history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig("images/03-01", dpi=300)
