from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]

# topic_model = BERTopic()
# topics, probs = topic_model.fit_transform(docs)
# topic_model.save("../data/bertopic_ex1")

topic_model = BERTopic.load("../data/bertopic_ex1")
print(topic_model.get_topic_info())
print(topic_model.get_topic(0))

topic_model.visualize_topics()
