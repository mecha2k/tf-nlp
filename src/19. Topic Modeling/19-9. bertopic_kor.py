from konlpy.tag import Mecab
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

text_file = "../data/2016-10-20.txt"
documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]

preprocessed_documents = []
for line in tqdm(documents):
    # 빈 문자열이거나 숫자로만 이루어진 줄은 제외
    if line and not line.replace(" ", "").isdecimal():
        preprocessed_documents.append(line)
print(preprocessed_documents[0])
print(preprocessed_documents[1])

# Mecab과 SBERT를 이용한 Bertopic
class CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger

    def __call__(self, sent):
        sent = sent[:1000000]
        word_tokens = self.tagger.morphs(sent)
        result = [word for word in word_tokens if len(word) > 1]
        return result


# mecab = Mecab(dicpath="c:/mecab/mecab-ko-dic")
# print(mecab.pos("아버지가방에들어가신다"))

custom_tokenizer = CustomTokenizer(Mecab(dicpath=r"c:/mecab/mecab-ko-dic"))
vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

model = BERTopic(
    embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
    language="multilingual",
    vectorizer_model=vectorizer,
    nr_topics=50,
    # nr_topics="auto",
    top_n_words=10,
    calculate_probabilities=True,
    verbose=True,
)

topics, probs = model.fit_transform(preprocessed_documents)
model.save("../data/bertopic_k_ex1")
print(f"{len(topics)}")

# BerTopic_model = BERTopic.load("../data/bertopic_k_ex1")

model.visualize_topics().write_html("images/topics_k.html")
model.visualize_distribution(probs[0]).write_html("images/dist_k.html")

for i in range(0, 5):
    print(i, "번째 토픽 :", model.get_topic(i))
