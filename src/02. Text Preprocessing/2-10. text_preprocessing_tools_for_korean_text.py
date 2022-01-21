import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 지도 학습(Supervised Learning)
# 지도 학습의 훈련 데이터는 문제지를 연상케 합니다. 지도 학습의 훈련 데이터는 정답이 무엇인지 맞춰 하는 '문제'에 해당되는 데이터와 레이블이라고 부르는
# '정답'이 적혀있는 데이터로 구성되어 있습니다. 쉽게 비유하면, 기계는 정답이 적혀져 있는 문제지를 문제와 정답을 함께 보면서 열심히 공부하고, 향후에
# 정답이 없는 문제에 대해서도 정답을 잘 예측해야 합니다.

# 2. X와 y분리하기
## 1) zip 함수를 이용하여 분리하기


X, y = zip(["a", 1], ["b", 2], ["c", 3])
print("X 데이터 :", X)
print("y 데이터 :", y)

# 리스트의 리스트 또는 행렬 또는 뒤에서 배울 개념인 2D 텐서.
sequences = [["a", 1], ["b", 2], ["c", 3]]
X, y = zip(*sequences)
print("X 데이터 :", X)
print("y 데이터 :", y)

## 2) 데이터프레임을 이용하여 분리하기

values = [
    ["당신에게 드리는 마지막 혜택!", 1],
    ["내일 뵐 수 있을지 확인 부탁드...", 0],
    ["철수씨. 잘 지내시죠? 오랜만입...", 0],
    ["(광고) AI로 주가를 예측할 수 있다!", 1],
]

columns = ["메일 본문", "스팸 메일 유무"]

df = pd.DataFrame(values, columns=columns)
print(df)

X = df["메일 본문"]
y = df["스팸 메일 유무"]

print("X 데이터 :", X.to_list())
print("y 데이터 :", y.to_list())

## 3) Numpy를 이용하여 분리하기

np_array = np.arange(0, 16).reshape((4, 4))
print("전체 데이터 :")
print(np_array)

X = np_array[:, :3]
y = np_array[:, 3]

print("X 데이터 :")
print(X)
print("y 데이터 :", y)

# 3. 테스트 데이터 분리하기

## 1) 사이킷 런을 이용하여 분리하기


"""
예시 코드
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)
"""

# 임의로 X와 y 데이터를 생성
X, y = np.arange(10).reshape((5, 2)), range(5)

print("X 전체 데이터 :")
print(X)
print("y 전체 데이터 :")
print(list(y))

# 7:3의 비율로 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

print("X 훈련 데이터 :")
print(X_train)
print("X 테스트 데이터 :")
print(X_test)

print("y 훈련 데이터 :")
print(y_train)
print("y 테스트 데이터 :")
print(y_test)

# random_state의 값을 변경
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print("y 훈련 데이터 :")
print(y_train)
print("y 테스트 데이터 :")
print(y_test)

# random_state을 이전의 값이었던 1234로 변경
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

print("y 훈련 데이터 :")
print(y_train)
print("y 테스트 데이터 :")
print(y_test)

## 2) 수동으로 분리하기

# 임의로 X와 y 데이터를 생성
X, y = np.arange(0, 24).reshape((12, 2)), range(12)

print("X 전체 데이터 :")
print(X)
print("y 전체 데이터 :")
print(list(y))

num_of_train = int(len(X) * 0.8)  # 데이터의 전체 길이의 80%에 해당하는 길이값을 구한다.
num_of_test = int(len(X) - num_of_train)  # 전체 길이에서 80%에 해당하는 길이를 뺀다.
print("훈련 데이터의 크기 :", num_of_train)
print("테스트 데이터의 크기 :", num_of_test)

X_test = X[num_of_train:]  # 전체 데이터 중에서 20%만큼 뒤의 데이터 저장
y_test = y[num_of_train:]  # 전체 데이터 중에서 20%만큼 뒤의 데이터 저장
X_train = X[:num_of_train]  # 전체 데이터 중에서 80%만큼 앞의 데이터 저장
y_train = y[:num_of_train]  # 전체 데이터 중에서 80%만큼 앞의 데이터 저장

print("X 테스트 데이터 :")
print(X_test)
print("y 테스트 데이터 :")
print(list(y_test))
