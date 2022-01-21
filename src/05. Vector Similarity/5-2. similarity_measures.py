import numpy as np


def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


doc1 = np.array((2, 3, 0, 1))
doc2 = np.array((1, 2, 3, 1))
doc3 = np.array((2, 1, 2, 2))
docQ = np.array((1, 1, 0, 1))

print("문서1과 문서Q의 거리 :", dist(doc1, docQ))
print("문서2과 문서Q의 거리 :", dist(doc2, docQ))
print("문서3과 문서Q의 거리 :", dist(doc3, docQ))

"""# 2. 자카드 유사도"""

doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()

print("문서1 :", tokenized_doc1)
print("문서2 :", tokenized_doc2)

union = set(tokenized_doc1).union(set(tokenized_doc2))
print("문서1과 문서2의 합집합 :", union)

intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
print("문서1과 문서2의 교집합 :", intersection)

print("자카드 유사도 :", len(intersection) / len(union))
