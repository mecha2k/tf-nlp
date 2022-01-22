# Original file is located at
#     https://colab.research.google.com/drive/147HoS3GOKwDzFDNx2pF7z70XWuo06n9z
# 이 자료는 위키독스 딥 러닝을 이용한 자연어 처리 입문의 판다스, 넘파이, 맷플롯립의 튜토리얼 자료입니다.


import pandas as pd

sr = pd.Series([17000, 18000, 1000, 5000], index=["피자", "치킨", "콜라", "맥주"])

print("시리즈 출력 :")
print("-" * 15)
print(sr)

print("시리즈의 값 : {}".format(sr.values))
print("시리즈의 인덱스 : {}".format(sr.index))

## 2) 데이터프레임

values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ["one", "two", "three"]
columns = ["A", "B", "C"]

df = pd.DataFrame(values, index=index, columns=columns)

print("데이터프레임 출력 :")
print("-" * 18)
print(df)

print("데이터프레임의 인덱스 : {}".format(df.index))
print("데이터프레임의 열이름: {}".format(df.columns))
print("데이터프레임의 값 :")
print("-" * 18)
print(df.values)

# 리스트로 생성하기
data = [
    ["1000", "Steve", 90.72],
    ["1001", "James", 78.09],
    ["1002", "Doyeon", 98.43],
    ["1003", "Jane", 64.19],
    ["1004", "Pilwoong", 81.30],
    ["1005", "Tony", 99.14],
]

df = pd.DataFrame(data)
print(df)

df = pd.DataFrame(data, columns=["학번", "이름", "점수"])
print(df)

# 딕셔너리로 생성하기
data = {
    "학번": ["1000", "1001", "1002", "1003", "1004", "1005"],
    "이름": ["Steve", "James", "Doyeon", "Jane", "Pilwoong", "Tony"],
    "점수": [90.72, 78.09, 98.43, 64.19, 81.30, 99.14],
}

df = pd.DataFrame(data)
print(df)

# 앞 부분을 3개만 보기
print(df.head(3))

# 뒷 부분을 3개만 보기
print(df.tail(3))

# '학번'에 해당되는 열을 보기
print(df["학번"])

# csv 파일을 사용하는 경우가 많습니다. csv 파일을 데이터프레임으로 로드 할 때는 다음과 같이 합니다.
# df = pd.read_csv('csv 파일의 경로')

# 2. Numpy

import numpy as np

## 1) np.array()

# 1차원 배열
vec = np.array([1, 2, 3, 4, 5])
print(vec)

# 2차원 배열
mat = np.array([[10, 20, 30], [60, 70, 80]])
print(mat)

print("vec의 타입 :", type(vec))
print("mat의 타입 :", type(mat))

print("vec의 차원 :", vec.ndim)  # 차원 출력
print("vec의 크기(shape) :", vec.shape)  # 크기 출력

print("mat의 차원 :", mat.ndim)  # 차원 출력
print("mat의 크기(shape) :", mat.shape)  # 크기 출력

## 2) ndarray의 초기화

# 모든 값이 0인 2x3 배열 생성.
zero_mat = np.zeros((2, 3))
print(zero_mat)

# 모든 값이 1인 2x3 배열 생성.
one_mat = np.ones((2, 3))
print(one_mat)

# 모든 값이 특정 상수인 배열 생성. 이 경우 7.
same_value_mat = np.full((2, 2), 7)
print(same_value_mat)

# 대각선 값이 1이고 나머지 값이 0인 2차원 배열을 생성.
eye_mat = np.eye(3)
print(eye_mat)

random_mat = np.random.random((2, 2))  # 임의의 값으로 채워진 배열 생성
print(random_mat)

## 3) np.arange()

# 0부터 9까지
range_vec = np.arange(10)
print(range_vec)

# 1부터 9까지 +2씩 적용되는 범위
n = 2
range_n_step_vec = np.arange(1, 10, n)
print(range_n_step_vec)

## 4) reshape()

reshape_mat = np.array(np.arange(30)).reshape((5, 6))
print(reshape_mat)

## 5) Numpy 슬라이싱

mat = np.array([[1, 2, 3], [4, 5, 6]])
print(mat)

# 첫번째 행 출력
slicing_mat = mat[0, :]
print(slicing_mat)

# 두번째 열 출력
slicing_mat = mat[:, 1]
print(slicing_mat)

## 6) Numpy 정수 인덱싱(integer indexing)

mat = np.array([[1, 2], [4, 5], [7, 8]])
print(mat)

# 1행 0열의 원소
# => 0부터 카운트하므로 두번째 행 첫번째 열의 원소.
print(mat[1, 0])

# mat[[2행, 1행],[0열, 1열]]
# 각 행과 열의 쌍을 매칭하면 2행 0열, 1행 1열의 두 개의 원소.
indexing_mat = mat[[2, 1], [0, 1]]
print(indexing_mat)

## 7) Numpy 연산

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# result = np.add(x, y)와 동일.
result = x + y
print(result)

# result = np.subtract(x, y)와 동일.
result = x - y
print(result)

# result = np.multiply(result, x)와 동일.
result = result * x
print(result)

# result = np.divide(result, x)와 동일.
result = result / x
print(result)

mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])

mat3 = np.dot(mat1, mat2)
print(mat3)

# 3. Matplotlib

import matplotlib.pyplot as plt

plt.title("test")
plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
plt.savefig("images/01-01")

plt.title("test")
plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
plt.xlabel("hours")
plt.ylabel("score")
plt.savefig("images/01-02")


plt.title("students")
plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
plt.plot([1.5, 2.5, 3.5, 4.5], [3, 5, 8, 10])  # 라인 신규 추가
plt.xlabel("hours")
plt.ylabel("score")
plt.legend(["A student", "B student"])  # 범례 삽입
plt.savefig("images/01-03")
