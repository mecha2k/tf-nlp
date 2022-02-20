import pandas as pd
from transformers import BertTokenizer
from icecream import ic

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Bert-base의 토크나이저

result = tokenizer.tokenize("Here is the sentence I want embeddings for.")
ic(result)
ic(tokenizer.vocab["here"])
# ic(tokenizer.vocab["embeddings"])
ic(tokenizer.vocab["em"])
ic(tokenizer.vocab["##bed"])
ic(tokenizer.vocab["##ding"])
ic(tokenizer.vocab["##s"])

# BERT의 단어 집합을 vocabulary.txt에 저장
with open("../data/vocabulary.txt", "w", encoding="utf-8") as f:
    for token in tokenizer.vocab.keys():
        f.write(token + "\n")

df = pd.read_fwf("../data/vocabulary.txt", header=None, encoding="utf-8")
ic(df)
ic("단어 집합의 크기 :", len(df))
ic(df.loc[4667].values[0])

ic(df.loc[0].values[0])
ic(df.loc[100].values[0])
ic(df.loc[101].values[0])
ic(df.loc[102].values[0])
ic(df.loc[103].values[0])
