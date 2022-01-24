import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 1. 아이리스 품종 데이터에 대한 이해
# iris_url = (
#     "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/"
#     "main/06.%20Machine%20Learning/dataset/Iris.csv"
# )
# urllib.request.urlretrieve(iris_url, filename="../data/Iris.csv")

data = pd.read_csv("../data/Iris.csv", encoding="latin1")
print("샘플의 개수 :", len(data))
print(data[:5])

# 중복을 허용하지 않고, 있는 데이터의 모든 종류를 출력
print("품종 종류:", data["Species"].unique(), sep="\n")

sns.set(style="ticks", color_codes=True)
g = sns.pairplot(data=data, hue="Species", palette="husl")
plt.savefig("images/10-01")

# 각 종과 특성에 대한 연관 관계
fig = plt.figure(figsize=(10, 6))
sns.barplot(x=data["Species"], y=data["SepalWidthCm"], ci=None)
plt.savefig("images/10-02")

fig = plt.figure(figsize=(10, 6))
data["Species"].value_counts().plot(kind="bar")
plt.savefig("images/10-03")

# Iris-virginica는 0, Iris-setosa는 1, Iris-versicolor는 2가 됨.
data["Species"] = data["Species"].replace(
    ["Iris-virginica", "Iris-setosa", "Iris-versicolor"], [0, 1, 2]
)
fig = plt.figure(figsize=(10, 6))
data["Species"].value_counts().plot(kind="bar")
plt.savefig("images/10-04")

data_X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
data_y = data["Species"].values
print(data_X[:5])
print(data_y[:5])

# 훈련 데이터와 테스트 데이터를 8:2로 나눕니다. 또한 데이터의 순서를 섞습니다.
(X_train, X_test, y_train, y_test) = train_test_split(
    data_X, data_y, train_size=0.8, random_state=42
)

# 원-핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train[:5])
print(y_test[:5])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

model = Sequential()
model.add(Dense(3, input_dim=4, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# 옵티마이저는 경사하강법의 일종인 adam을 사용합니다.
# 손실 함수(Loss function)는 크로스 엔트로피 함수를 사용합니다.
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 200번 시도합니다.
history = model.fit(
    X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test), verbose=0
)

epochs = range(1, len(history.history["accuracy"]) + 1)

fig = plt.figure(figsize=(10, 6))
plt.plot(epochs, history.history["loss"])
plt.plot(epochs, history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("images/10-05")

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
