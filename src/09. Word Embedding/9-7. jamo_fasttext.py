# Commented out IPython magic to ensure Python compatibility.
# Colab에 Mecab 설치
# !git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab
# !bash install_mecab-ko_on_colab190912.sh

# hgtk : https://github.com/bluedisk/hangul-toolkit

# 한글 자모 단위 처리 패키지 설치
# pip install hgtk

# Commented out IPython magic to ensure Python compatibility.
# fasttext 설치
# !git clone https://github.com/facebookresearch/fastText.git
# %cd fastText
# !make
# !pip install .

# 1. 데이터 로드

import re
import pandas as pd
import urllib.request
from tqdm import tqdm
import hgtk
import fasttext
from konlpy.tag import Mecab
from icecream import ic
import platform

osname = platform.system()
if osname == "Windows":
    mecab = Mecab(dicpath="C:/mecab/mecab-ko-dic")
else:
    mecab = Mecab()


def word_to_jamo(token):
    def to_special_token(jamo):
        if not jamo:
            return "-"
        else:
            return jamo

    decomposed_token = ""
    for char in token:
        try:
            # char(음절)을 초성, 중성, 종성으로 분리
            cho, jung, jong = hgtk.letter.decompose(char)

            # 자모가 빈 문자일 경우 특수문자 -로 대체
            cho = to_special_token(cho)
            jung = to_special_token(jung)
            jong = to_special_token(jong)
            decomposed_token = decomposed_token + cho + jung + jong

        # 만약 char(음절)이 한글이 아닐 경우 자모를 나누지 않고 추가
        except Exception as exception:
            if type(exception).__name__ == "NotHangulException":
                decomposed_token += char

    # 단어 토큰의 자모 단위 분리 결과를 추가
    return decomposed_token


def tokenize_by_jamo(s):
    return [word_to_jamo(token) for token in mecab.morphs(s)]


def jamo_to_word(jamo_sequence):
    tokenized_jamo = []
    index = 0

    # 1. 초기 입력
    # jamo_sequence = 'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ'
    while index < len(jamo_sequence):
        # 문자가 한글(정상적인 자모)이 아닐 경우
        if not hgtk.checker.is_hangul(jamo_sequence[index]):
            tokenized_jamo.append(jamo_sequence[index])
            index = index + 1

        # 문자가 정상적인 자모라면 초성, 중성, 종성을 하나의 토큰으로 간주.
        else:
            tokenized_jamo.append(jamo_sequence[index : index + 3])
            index = index + 3

    # 2. 자모 단위 토큰화 완료
    # tokenized_jamo : ['ㄴㅏㅁ', 'ㄷㅗㅇ', 'ㅅㅐㅇ']

    word = ""
    try:
        for jamo in tokenized_jamo:

            # 초성, 중성, 종성의 묶음으로 추정되는 경우
            if len(jamo) == 3:
                if jamo[2] == "-":
                    # 종성이 존재하지 않는 경우
                    word = word + hgtk.letter.compose(jamo[0], jamo[1])
                else:
                    # 종성이 존재하는 경우
                    word = word + hgtk.letter.compose(jamo[0], jamo[1], jamo[2])
            # 한글이 아닌 경우
            else:
                word = word + jamo

    # 복원 중(hgtk.letter.compose) 에러 발생 시 초기 입력 리턴.
    # 복원이 불가능한 경우 예시) 'ㄴ!ㅁㄷㅗㅇㅅㅐㅇ'
    except Exception as exception:
        if type(exception).__name__ == "NotHangulException":
            return jamo_sequence

    # 3. 단어로 복원 완료
    # word : '남동생'

    return word


# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt",
#     filename="../data/ratings_total.txt",
# )

# total_data = pd.read_table("../data/ratings_total.txt", names=["ratings", "reviews"])
# print("전체 리뷰 개수 :", len(total_data))  # 전체 리뷰 개수 출력
# print(total_data[:5])
#
# # 2. hgtk 튜토리얼
# # 한글인지 체크
# ic(hgtk.checker.is_hangul("ㄱ"))
# # 한글인지 체크
# ic(hgtk.checker.is_hangul("28"))
# # 음절을 초성, 중성, 종성으로 분해
# ic(hgtk.letter.decompose("남"))
# # 초성, 중성을 결합
# ic(hgtk.letter.compose("ㄴ", "ㅏ"))
# # 초성, 중성, 종성을 결합
# ic(hgtk.letter.compose("ㄴ", "ㅏ", "ㅁ"))
#
# # 한글이 아닌 입력에 대해서는 에러 발생.
# try:
#     ic(hgtk.letter.decompose("1"))
# except Exception as e:
#     print(e)
#
# # 결합할 수 없는 상황에서는 에러 발생
# try:
#     ic(hgtk.letter.compose("ㄴ", "ㅁ", "ㅁ"))
# except Exception as e:
#     print(e)
#
# # 3. 데이터 전처리
# ic(word_to_jamo("남동생"))
# ic(word_to_jamo("여동생"))
# ic(mecab.morphs("선물용으로 빨리 받아서 전달했어야 하는 상품이었는데 머그컵만 와서 당황했습니다."))
# ic(tokenize_by_jamo("선물용으로 빨리 받아서 전달했어야 하는 상품이었는데 머그컵만 와서 당황했습니다."))
#
# tokenized_data = []
# for sample in tqdm(total_data["reviews"].to_list()):
#     tokenzied_sample = tokenize_by_jamo(sample)  # 자소 단위 토큰화
#     tokenized_data.append(tokenzied_sample)
# print(len(tokenized_data))
# print(tokenized_data[0])
#
# ic(jamo_to_word("ㄴㅏㅁㄷㅗㅇㅅㅐㅇ"))
#
# # 4. FastText
# with open("../data/tokenized_data.txt", "w") as out:
#     for line in tqdm(tokenized_data, unit=" line"):
#         out.write(" ".join(line) + "\n")
#
# model = fasttext.train_unsupervised("../data/tokenized_data.txt", model="cbow")
# model.save_model("../data/fasttext.bin")

model = fasttext.load_model("../data/fasttext.bin")
print(model[word_to_jamo("남동생")])  # 'ㄴㅏㅁㄷㅗㅇㅅㅐㅇ'
model.get_nearest_neighbors(word_to_jamo("남동생"), k=10)


def transform(word_sequence):
    return [(jamo_to_word(word), similarity) for (similarity, word) in word_sequence]


ic(transform(model.get_nearest_neighbors(word_to_jamo("남동생"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("남동쉥"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("남동셍ㅋ"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("난동생"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("낫동생"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("납동생"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("냚동생"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("고품질"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("고품쥘"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("노품질"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("보품질"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("제품"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("제품ㅋ"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("제품^^"), k=10)))
ic(transform(model.get_nearest_neighbors(word_to_jamo("제푼ㅋ"), k=10)))
