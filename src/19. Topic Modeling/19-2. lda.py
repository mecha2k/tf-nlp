import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
import gensim
from gensim import corpora

nltk.download("stopwords")

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=("headers", "footers", "quotes"))
documents = dataset.data
print("샘플의 수 :", len(documents))
print(documents[1])
print(dataset.target_names)

news_df = pd.DataFrame({"document": documents})
# 특수 문자 제거
news_df["clean_doc"] = news_df["document"].str.replace("[^a-zA-Z]", " ")
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df["clean_doc"] = news_df["clean_doc"].apply(
    lambda x: " ".join([w for w in x.split() if len(w) > 3])
)
# 전체 단어에 대한 소문자 변환
news_df["clean_doc"] = news_df["clean_doc"].apply(lambda x: x.lower())

print(news_df["clean_doc"][1])

# NLTK로부터 불용어를 받아온다.
stop_words = stopwords.words("english")
tokenized_doc = news_df["clean_doc"].apply(lambda x: x.split())  # 토큰화
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

print(tokenized_doc[:5])

dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]

# 수행된 결과에서 1번 인덱스 뉴스 출력
print(corpus[1])
print(dictionary[66])
print(len(dictionary))

NUM_TOPICS = 20
ldamodel = gensim.models.ldamodel.LdaModel(
    corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15
)

topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

# import pyLDAvis.gensim_models
#
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
# pyLDAvis.display(vis)

for i, topic_list in enumerate(ldamodel[corpus]):
    if i == 5:
        break
    print(i, "번째 문서의 topic 비율은", topic_list)


def make_topictable_per_doc(ldamodel, corpus):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%),
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc):  #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_list]), ignore_index=True
                )
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return topic_table


topictable = make_topictable_per_doc(ldamodel, corpus)
topictable = topictable.reset_index()  # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
topictable.columns = ["문서 번호", "가장 비중이 높은 토픽", "가장 높은 토픽의 비중", "각 토픽의 비중"]
print(topictable[:10])
