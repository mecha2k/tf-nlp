# conda install -c conda-forge jpype1
# pip3 install konlpy, kss

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer

from konlpy.tag import Okt
from konlpy.tag import Kkma

# KSS(Korean Sentence Splitter)
import kss
import os

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from multiprocessing import freeze_support

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


if __name__ == "__main__":
    freeze_support()

    # 1. 단어 토큰화(Word Tokenization)
    # 토큰의 기준을 단어(word)로 하는 경우, 단어 토큰화(word tokenization)라고 합니다. 다만, 여기서 단어(word)는 단어 단위 외에도 단어구,
    # 의미를 갖는 문자열로도 간주되기도 합니다.
    # 2. 토큰화 중 생기는 선택의 순간

    print(
        "단어 토큰화1 :",
        word_tokenize(
            "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
        ),
    )

    print(
        "단어 토큰화2 :",
        WordPunctTokenizer().tokenize(
            "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
        ),
    )

    print(
        "단어 토큰화3 :",
        text_to_word_sequence(
            "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
        ),
    )

    # 3. 토큰화에서 고려해야할 사항
    tokenizer = TreebankWordTokenizer()

    text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
    print("트리뱅크 워드토크나이저 :", tokenizer.tokenize(text))

    # 4. 문장 토큰화(Sentence Tokenization)
    text = (
        "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, "
        "the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. "
        "He looked about, to make sure no one was near."
    )
    print("문장 토큰화1 :", sent_tokenize(text))

    text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
    print("문장 토큰화2 :", sent_tokenize(text))

    text = "딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?"
    print("한국어 문장 토큰화 :", kss.split_sentences(text))

    # 8. NLTK와 KoNLPy를 이용한 영어, 한국어 토큰화 실습
    text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
    tokenized_sentence = word_tokenize(text)

    print("단어 토큰화 :", tokenized_sentence)
    print("품사 태깅 :", pos_tag(tokenized_sentence))

    okt = Okt()
    kkma = Kkma()

    print("OKT 형태소 분석 :", okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
    print("OKT 품사 태깅 :", okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
    print("OKT 명사 추출 :", okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

    print("꼬꼬마 형태소 분석 :", kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
    print("꼬꼬마 품사 태깅 :", kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
    print("꼬꼬마 명사 추출 :", kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
