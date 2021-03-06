# 1. 벡터와 행렬과 텐서
# 벡터는 크기와 방향을 가진 양입니다. 숫자가 나열된 형상이며 파이썬에서는 1차원 배열 또는 리스트로 표현합니다. 반면, 행렬은 행과 열을 가지는 2차원
# 형상을 가진 구조입니다. 파이썬에서는 2차원 배열로 표현합니다. 가로줄을 행(row)라고 하며, 세로줄을 열(column)이라고 합니다. 3차원부터는 주로
# 텐서라고 부릅니다. 텐서는 파이썬에서는 3차원 이상의 배열로 표현합니다.

# 2. 텐서(Tensor)

import numpy as np

## 0차원 텐서 (스칼라)

d = np.array(5)
print("차원 :", d.ndim)
print("텐서의 크기(shape) :", d.shape)

## 1차원 텐서 (벡터)

d = np.array([1, 2, 3, 4])
print("차원 :", d.ndim)
print("텐서의 크기(shape) :", d.shape)

## 2차원 텐서(행렬)

d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("차원 :", d.ndim)
print("텐서의 크기(shape) :", d.shape)

## 3차원 텐서(다차원 배열)

d = np.array(
    [
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [19, 20, 21, 22, 23], [23, 24, 25, 26, 27]],
    ]
)
print("차원 :", d.ndim)
print("텐서의 크기(shape) :", d.shape)

# 3. 벡터와 행렬의 연산

import numpy as np

A = np.array([8, 4, 5])
B = np.array([1, 2, 3])
print("두 벡터의 합 :", A + B)
print("두 벡터의 차 :", A - B)

A = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
B = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])
print("두 행렬의 합 :")
print(A + B)
print("두 행렬의 차 :")
print(A - B)

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
print("두 벡터의 내적 :", np.dot(A, B))

A = np.array([[1, 3], [2, 4]])
B = np.array([[5, 7], [6, 8]])
print("두 행렬의 행렬곱 :")
print(np.matmul(A, B))
print(A @ B)
