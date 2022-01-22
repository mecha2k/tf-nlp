# 1. IMDB 리뷰
# pip install sentencepiece
# pip list | grep sentencepiece

import sentencepiece as spm
import pandas as pd
import urllib.request
import csv

# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv",
#     filename="../data/IMDb_Reviews.csv",
# )

train_df = pd.read_csv("../data/IMDb_Reviews.csv")
print(train_df["review"])
print("리뷰 개수 :", len(train_df))  # 리뷰 개수 출력

with open("../data/imdb_review.txt", "w", encoding="utf8") as f:
    f.write("\n".join(train_df["review"]))

spm.SentencePieceTrainer.Train(
    "--input=../data/imdb_review.txt --model_prefix=imdb --vocab_size=5000 --model_type=bpe --max_sentence_length=9999"
)

vocab_list = pd.read_csv("imdb.vocab", sep="\t", header=None, quoting=csv.QUOTE_NONE)
print(vocab_list.sample(10))
print(len(vocab_list))

sp = spm.SentencePieceProcessor()
vocab_file = "imdb.model"
sp.load(vocab_file)

lines = ["I didn't at all think of it this way.", "I have waited a long time for someone to film"]
for line in lines:
    print(line)
    print(sp.encode_as_pieces(line))
    print(sp.encode_as_ids(line))
    print()

sp.GetPieceSize()

sp.IdToPiece(430)

sp.PieceToId("▁character")

sp.DecodeIds([41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91])

sp.DecodePieces(
    ["▁I", "▁have", "▁wa", "ited", "▁a", "▁long", "▁time", "▁for", "▁someone", "▁to", "▁film"]
)

print(sp.encode("I have waited a long time for someone to film", out_type=str))
print(sp.encode("I have waited a long time for someone to film", out_type=int))

# 2. 네이버 영화 리뷰

import pandas as pd
import sentencepiece as spm
import urllib.request
import csv

# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt"
# )

naver_df = pd.read_table("../data/ratings.txt")
print(naver_df[:5])
print("리뷰 개수 :", len(naver_df))  # 리뷰 개수 출력
print(naver_df.isnull().values.any())

naver_df = naver_df.dropna(how="any")  # Null 값이 존재하는 행 제거
print(naver_df.isnull().values.any())  # Null 값이 존재하는지 확인
print("리뷰 개수 :", len(naver_df))  # 리뷰 개수 출력

with open("../data/naver_review.txt", "w", encoding="utf8") as f:
    f.write("\n".join(naver_df["document"]))

spm.SentencePieceTrainer.Train(
    "--input=../data/naver_review.txt --model_prefix=naver --vocab_size=5000 --model_type=bpe --max_sentence_length=9999"
)

vocab_list = pd.read_csv("naver.vocab", sep="\t", header=None, quoting=csv.QUOTE_NONE)
print(vocab_list[:10])
print(vocab_list.sample(10))
print(len(vocab_list))

sp = spm.SentencePieceProcessor()
vocab_file = "naver.model"
sp.load(vocab_file)

lines = [
    "뭐 이딴 것도 영화냐.",
    "진짜 최고의 영화입니다 ㅋㅋ",
]
for line in lines:
    print(line)
    print(sp.encode_as_pieces(line))
    print(sp.encode_as_ids(line))
    print()

sp.GetPieceSize()
sp.IdToPiece(4)
sp.PieceToId("영화")
sp.DecodeIds([54, 200, 821, 85])
sp.DecodePieces(["▁진짜", "▁최고의", "▁영화입니다", "▁ᄏᄏ"])

print(sp.encode("진짜 최고의 영화입니다 ㅋㅋ", out_type=str))
print(sp.encode("진짜 최고의 영화입니다 ㅋㅋ", out_type=int))
