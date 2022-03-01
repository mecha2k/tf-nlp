from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from konlpy.tag import Okt
from contextualized_topic_models.models.ctm import CombinedTM


with open("../data/대한민국헌법.txt", "r", encoding="utf-8") as f:
    text = f.read()

okt = Okt()
nouns = okt.nouns(text)  # 명사만 추출

words = [n for n in nouns if len(n) > 1]  # 단어의 길이가 1개인 것은 제외

c = Counter(words)  # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구함
wc = WordCloud(
    font_path="../data/NanumBarunGothic.ttf", width=400, height=400, scale=2.0, max_font_size=250
)
gen = wc.generate_from_frequencies(c)
plt.figure()
plt.imshow(gen)
wc.to_file("images/법전_워드클라우드.png")

ctm = CombinedTM(
    bow_size=3000,
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
n_words = 15
word_score_list = ctm.get_word_distribution_by_topic_id(topic_id)[:n_words]
word_score_dict = {key: value for (key, value) in word_score_list}

wc = WordCloud(
    font_path="../data/NanumBarunGothic.ttf", width=400, height=400, scale=2.0, max_font_size=250
)
gen = wc.generate_from_frequencies(word_score_dict)
plt.figure()
plt.imshow(gen)
wc.to_file("images/word_cloud.png")
