# pip install transformers

import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Bert-base의 토크나이저

result = tokenizer.tokenize("Here is the sentence I want embeddings for.")
print(result)
print(tokenizer.vocab["here"])
# print(tokenizer.vocab["embeddings"])
print(tokenizer.vocab["em"])
print(tokenizer.vocab["##bed"])
print(tokenizer.vocab["##ding"])
print(tokenizer.vocab["##s"])

# BERT의 단어 집합을 vocabulary.txt에 저장
with open("../data/vocabulary.txt", "w", encoding="utf-8") as f:
    for token in tokenizer.vocab.keys():
        f.write(token + "\n")

df = pd.read_fwf("../data/vocabulary.txt", header=None, encoding="utf-8")
print(df)
print("단어 집합의 크기 :", len(df))
print(df.loc[4667].values[0])
print(df.loc[102].values[0])
