from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import (
    TopicModelDataPreparation,
    bert_embeddings_from_list,
)
import contextualized_topic_models
import nltk
import pyLDAvis as vis
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Mecab
from tqdm import tqdm


def prepare_data():
    nltk.download("stopwords", quiet=True)
    text_file = "../data/2016-10-20.txt"

    documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]
    print(documents[:3])

    preprocessed_documents = []
    for line in tqdm(documents):
        if line and not line.replace(" ", "").isdecimal():
            preprocessed_documents.append(line)
    print(len(preprocessed_documents))

    class CustomTokenizer:
        def __init__(self, tagger):
            self.tagger = tagger

        def __call__(self, sent):
            word_tokens = self.tagger.morphs(sent)
            result = [word for word in word_tokens if len(word) > 1]
            return result

    custom_tokenizer = CustomTokenizer(Mecab(dicpath="c:/mecab/mecab-ko-dic"))
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)
    train_bow_embeddings = vectorizer.fit_transform(preprocessed_documents)
    print(train_bow_embeddings.shape)

    vocab = vectorizer.get_feature_names_out()
    id2token = {k: v for k, v in zip(range(0, len(vocab)), vocab)}
    print(len(vocab))

    train_contextualized_embeddings = bert_embeddings_from_list(
        preprocessed_documents,
        "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
    )

    qt = TopicModelDataPreparation()
    training_dataset = qt.load(train_contextualized_embeddings, train_bow_embeddings, id2token)

    return preprocessed_documents, vocab, training_dataset


# Combined TM 학습하기
def combined_topic_modeling(preprocessed_documents, vocab, training_dataset):
    ctm = CombinedTM(
        bow_size=len(vocab),
        contextual_size=768,
        n_components=50,
        num_epochs=20,
        num_data_loader_workers=0,
    )
    ctm.fit(training_dataset)

    print(ctm.get_topics(5))

    lda_vis_data = ctm.get_ldavis_data_format(vocab, training_dataset, n_samples=10)
    ctm_pd = vis.prepare(**lda_vis_data)
    # vis.display(ctm_pd)
    vis.save_html(ctm_pd, "images/comb_topic_k.html")

    # 이제 임의의 문서를 가져와서 어떤 토픽이 할당되었는지 확인할 수 있습니다.
    # 예를 들어, 반도(peninsula)에 대한 첫번째 전처리 된 문서의 토픽을 예측해 봅시다.
    topics_predictions = ctm.get_thetas(training_dataset, n_samples=5)

    # 전처리 문서의 첫번째 문서
    print(preprocessed_documents[0])

    # get the topic id of the first document and the topic should be about natural location related things
    topic_number = np.argmax(topics_predictions[0])
    print(ctm.get_topic_lists(5)[topic_number])

    # 차후 사용을 위해 모델 저장하기
    ctm.save(models_dir="../data/ctm_kor")


def load_ctm_models(vocab_len):
    ctm = CombinedTM(
        bow_size=vocab_len,
        contextual_size=768,
        num_epochs=100,
        n_components=50,
        num_data_loader_workers=0,
    )
    ctm.load(
        "../data/ctm_kor/contextualized_topic_model_nc_50_tpm_0.0_tpv_0.98_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99",
        epoch=19,
    )
    print(ctm.get_topic_lists(5))

    # ctm.get_wordcloud(topic_id=5, n_words=15)
    topic_id = 5
    n_words = 20
    word_score_list = ctm.get_word_distribution_by_topic_id(topic_id)[:n_words]
    word_score_dict = {key: value for (key, value) in word_score_list}

    word_cloud = wordcloud.WordCloud(
        font_path="../data/NanumBarunGothic.ttf",
        width=800,
        height=400,
        scale=2.0,
        max_font_size=250,
    )
    gen_word_cloud = word_cloud.generate_from_frequencies(word_score_dict)

    plt.figure()
    plt.imshow(gen_word_cloud)
    plt.axis("off")
    plt.title("Displaying Topic " + str(topic_id), loc="center", fontsize=24)
    word_cloud.to_file("images/word_cloud.png")


if __name__ == "__main__":
    # stable version: 2.2.0
    print(contextualized_topic_models.__version__)

    vocab_len = 3000
    # preprocessed_documents, vocab, training_dataset = prepare_data()
    # combined_topic_modeling(preprocessed_documents, vocab, training_dataset)
    # vocab_len = len(vocab)
    load_ctm_models(vocab_len)
