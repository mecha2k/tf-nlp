import nltk
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


def build_bag_of_words(document):
    # 온점 제거
    document = document.replace(".", "")
    tokenized_document = okt.morphs(document)
    word_to_index = {}
    bow = []

    for word in tokenized_document:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)
            # BoW에 전부 기본값 1을 넣는다.
            bow.insert(len(word_to_index) - 1, 1)
        else:
            # 재등장하는 단어의 인덱스
            index = word_to_index.get(word)
            # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
            bow[index] = bow[index] + 1

    return word_to_index, bow


okt = Okt()
nltk.download("stopwords", quiet=True)

doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
vocab, bow = build_bag_of_words(doc1)

print("vocabulary :", vocab)
print("bag of words vector :", bow)

# 2. 다양한 예제들

doc2 = "소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다."
vocab, bow = build_bag_of_words(doc2)
print("vocabulary :", vocab)
print("bag of words vector :", bow)

doc3 = doc1 + " " + doc2
print(doc3)

vocab, bow = build_bag_of_words(doc3)
print("vocabulary :", vocab)
print("bag of words vector :", bow)

# 3. CountVectorizer 클래스로 BoW 만들기
corpus = ["you know I want your love. because I love you."]
vector = CountVectorizer()

# 코퍼스로부터 각 단어의 빈도수를 기록
print("bag of words vector :", vector.fit_transform(corpus).toarray())

# 각 단어의 인덱스가 어떻게 부여되었는지를 출력
print("vocabulary :", vector.vocabulary_)

# 4. 불용어를 제거한 BoW
text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])
print("bag of words vector :", vect.fit_transform(text).toarray())
print("vocabulary :", vect.vocabulary_)

text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
print("bag of words vector :", vect.fit_transform(text).toarray())
print("vocabulary :", vect.vocabulary_)

text = ["Family is not an important thing. It's everything."]
stop_words = stopwords.words("english")
vect = CountVectorizer(stop_words=stop_words)
print("bag of words vector :", vect.fit_transform(text).toarray())
print("vocabulary :", vect.vocabulary_)
