import tokenizers
import pandas as pd
import urllib.request
from tokenizers import BertWordPieceTokenizer

print(tokenizers.__version__)

naver_df = pd.read_table("../data/ratings.txt")
naver_df.head()

naver_df = naver_df.dropna(how="any")  # Null 값이 존재하는 행 제거
print(naver_df.isna().values.any())  # Null 값이 존재하는지 확인

with open("../data/naver_review.txt", "w", encoding="utf8") as f:
    f.write("\n".join(naver_df["document"]))

tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False)

# lowercase : True일 경우 토크나이저는 영어의 대문자와 소문자를 동일한 문자 취급.
# strip_accents : True일 경우 악센트 제거.
#  ex) é → e, ô → o
# wordpieces_prefix : 서브워드로 쪼개졌을 경우 뒤의 서브워드에는 ##를 부착하여 원래 단어에서 분리된 것임을 표시.
#   ex) 안녕하세요 -> [안녕, ##하세요]


data_file = "../data/naver_review.txt"
vocab_size = 30000
limit_alphabet = 6000
min_frequency = 5

tokenizer.train(
    files=data_file,
    vocab_size=vocab_size,
    limit_alphabet=limit_alphabet,
    min_frequency=min_frequency,
)

# vocab_size : 단어 집합의 크기
# limit_alphabet : merge가 되지 않은 초기 토큰(character 단위)의 허용 제한 개수
# min_frequency : merge가 되기 위한 pair의 최소 등장 횟수


# vocab 저장
tokenizer.save_model("../data")

# vocab 로드
df = pd.read_fwf("../data/vocab.txt", header=None)
print(df)

encoded = tokenizer.encode("아 배고픈데 짜장면먹고싶다")
print("토큰화 결과 :", encoded.tokens)
print("정수 인코딩 :", encoded.ids)
print("디코딩 :", tokenizer.decode(encoded.ids))

encoded = tokenizer.encode("커피 한잔의 여유를 즐기다")
print("토큰화 결과 :", encoded.tokens)
print("정수 인코딩 :", encoded.ids)
print("디코딩 :", tokenizer.decode(encoded.ids))

# 2. 기타 토크나이저

from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer

tokenizer = SentencePieceBPETokenizer()
tokenizer.train("../data/naver_review.txt", vocab_size=10000, min_frequency=5)

encoded = tokenizer.encode("이 영화는 정말 재미있습니다.")
print(encoded.tokens)
