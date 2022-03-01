from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from multiprocessing import freeze_support
import contextualized_topic_models
import nltk

# stable version: 2.2.0
print(contextualized_topic_models.__version__)


def combined_topic_modeling():
    nltk.download("stopwords", quiet=True)
    text_file = "../data/dbpedia_sample_abstract_20k_unprep.txt"

    # 여기서 전처리 된 텍스트를 사용하는 이유는 무엇일까요? Bag of Words를 구축하려면 특수문자가 없는 텍스트가 필요하고,
    # 모든 단어를 사용하는 것보다는 빈번한 단어들만 사용하는 것이 좋습니다.
    documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]
    sp = WhiteSpacePreprocessing(documents)
    preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()

    # normalization 전처리 후 문서
    print(preprocessed_documents[:2])

    # 전처리 전 문서 == documnets와 동일
    print(unpreprocessed_corpus[:2])

    # 전체 단어 집합의 개수
    print("bag of words에 사용 될 단어 집합의 개수 :", len(vocab))
    print(vocab[:5])

    # 전처리 되지 않은 문서는 문맥을 반영한 문서 임베딩을 얻기 위한 입력으로 사용할 것이기 때문에 제거해서는 안 됩니다.
    # 전처리 전 문서와 전처리 후 문서를 TopicModelDataPreparation 객체에 넘겨줍니다. 이 객체는 bag of words와 문맥을 반영한
    # 문서의 BERT 임베딩을 얻습니다. 여기서 사용할 pretrained BERT는 paraphrase-distilroberta-base-v1입니다.
    #
    # tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v1")
    # training_dataset = tp.fit(
    #     text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents
    # )
    #
    # print(tp.vocab[:10])
    # print(len(tp.vocab))
    #
    # # 단어 집합의 상위 10개 단어를 출력해봅시다. 여기서 출력하는 tp.vocab과 앞에서의 vocab은 집합 관점에서는 같습니다.
    # print(set(vocab) == set(tp.vocab))
    #
    # # Combined TM 학습하기
    # # 이제 토픽 모델을 학습합니다. 여기서는 하이퍼파라미터에 해당하는 토픽의 개수(n_components)로는 50개를 선정합니다.
    # ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=50, num_epochs=20)
    # ctm.fit(training_dataset)  # run the model
    #
    # # 토픽들
    # # 학습 후에는 토픽 모델이 선정한 토픽들을 보려면 아래의 메소드를 사용합니다.
    # # get_topic_lists
    # # 해당 메소드에는 각 토픽마다 몇 개의 단어를 보고 싶은지에 해당하는 파라미터를 넣어즐 수 있습니다. 여기서는 5개를 선택했습니다. 아래의 토픽들은
    # # 위키피디아(일반적인 주제)으로부터 얻은 토픽을 보여줍니다. 우리는 영어 문서로 학습하였으므로 각 토픽에 해당하는 단어들도 영어 단어들입니다.
    #
    # print(ctm.get_topic_lists(5))
    #
    # # 우리의 토픽들을 시각화하기 위해서는 PyLDAvis를 사용합니다.
    # # lda_vis_data = ctm.get_ldavis_data_format(tp.vocab, training_dataset, n_samples=10)
    #
    # # import pyLDAvis as vis
    # #
    # # lda_vis_data = ctm.get_ldavis_data_format(tp.vocab, training_dataset, n_samples=10)
    # #
    # # ctm_pd = vis.prepare(**lda_vis_data)
    # # vis.display(ctm_pd)
    #
    # # 이제 임의의 문서를 가져와서 어떤 토픽이 할당되었는지 확인할 수 있습니다. 예를 들어, 반도(peninsula)에 대한 첫번째 전처리 된 문서의
    # # 토픽을 예측해 봅시다.
    # topics_predictions = ctm.get_thetas(
    #     training_dataset, n_samples=5
    # )  # get all the topic predictions
    #
    # # 전처리 문서의 첫번째 문서
    # print(preprocessed_documents[0])
    #
    # # import numpy as np
    # # topic_number = np.argmax(topics_predictions[0]) # get the topic id of the first document
    # #
    # # ctm.get_topic_lists(5)[topic_number] #and the topic should be about natural location related things
    # #
    # # """# 차후 사용을 위해 모델 저장하기"""
    # #
    # # ctm.save(models_dir="./")
    # #
    # # # let's remove the trained model
    # # del ctm
    # #
    # # ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, num_epochs=100, n_components=50)
    # #
    # # ctm.load("/content/contextualized_topic_model_nc_50_tpm_0.0_tpv_0.98_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99",
    # #                                                                                                       epoch=19)
    # #
    # # ctm.get_topic_lists(5)
    # #
    # # """참고 자료 : https://github.com/MilaNLProc/contextualized-topic-models"""


if __name__ == "__main__":
    freeze_support()
    combined_topic_modeling()
