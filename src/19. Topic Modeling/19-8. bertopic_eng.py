from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]

print(docs[:5])
print("총 문서의 수 :", len(docs))

# model = BERTopic()
# topics, probabilities = model.fit_transform(docs)
#
# print("각 문서의 토픽 번호 리스트 :", len(topics))
# print("첫번째 문서의 토픽 번호 :", topics[0])
#
# print(model.get_topic_info())
# # Count 열의 값을 모두 합하면 총 문서의 수입니다.
# print(model.get_topic_info()["Count"].sum())
#
# # 위의 출력에서 Topic -1이 가장 큰 것으로 보입니다. -1은 토픽이 할당되지 않은 모든 이상치 문서(outliers)들을 나타냅니다. 현재 0번
# # 토픽부터 210번 토픽까지 있는데, 임의로 5번 토픽에 대해서 단어들을 출력해봅시다.
# print(model.get_topic(5))

# model = BERTopic(nr_topics=20)
# topics, probabilities = model.fit_transform(docs)
# model.visualize_topics()

model = BERTopic(nr_topics="auto")
topics, probabilities = model.fit_transform(docs)
model.get_topic_info()

model.visualize_topics().write_html("images/topics.html")
model.visualize_barchart().write_html("images/barchart.html")
model.visualize_heatmap().write_html("images/heatmap.html")

new_doc = docs[0]
print(new_doc)
topics, probs = model.transform([new_doc])
print("예측한 토픽 번호 :", topics)

model.save("../data/bertopic_ex1")
# BerTopic_model = BERTopic.load("../data/bertopic_ex1")
