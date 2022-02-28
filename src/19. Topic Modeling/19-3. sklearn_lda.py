import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

data = pd.read_csv("../data/abcnews-date-text.csv")
print("뉴스 제목 개수 :", len(data))
print(data.head(5))

text = data.loc[:, ["headline_text"]]
print(text.head(5))

text["headline_text"] = text.apply(lambda row: nltk.word_tokenize(row["headline_text"]), axis=1)
print(text.head(5))

stop_words = stopwords.words("english")
text["headline_text"] = text.loc[:, "headline_text"].apply(
    lambda x: [word for word in x if word not in stop_words]
)
print(text.head(5))

text["headline_text"] = text.loc[:, "headline_text"].apply(
    lambda x: [WordNetLemmatizer().lemmatize(word, pos="v") for word in x]
)
print(text.head(5))

tokenized_doc = text.loc[:, "headline_text"].apply(lambda x: [word for word in x if len(word) > 3])
print(tokenized_doc[:5])

# 역토큰화 (토큰화 작업을 되돌림)
detokenized_doc = []
for i in range(len(text)):
    t = " ".join(tokenized_doc[i])
    detokenized_doc.append(t)

# 다시 text['headline_text']에 재저장
text["headline_text"] = detokenized_doc
print(text["headline_text"][:5])

# 상위 1,000개의 단어를 보존
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(text["headline_text"])

# TF-IDF 행렬의 크기 확인
print("TF-IDF 행렬의 크기 :", X.shape)

lda_model = LatentDirichletAllocation(
    n_components=10, learning_method="online", random_state=777, max_iter=1
)

lda_top = lda_model.fit_transform(X)

print(lda_model.components_)
print(lda_model.components_.shape)

# 단어 집합. 1,000개의 단어가 저장됨.
terms = vectorizer.get_feature_names_out()


def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print(
            "Topic %d:" % (idx + 1),
            [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[: -n - 1 : -1]],
        )


get_topics(lda_model.components_, terms)
