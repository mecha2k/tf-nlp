# pip install git+https://github.com/haven-jeon/PyKoSpacing.git

sent = (
    "김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 "
    "사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다."
)

new_sent = sent.replace(" ", "")  # 띄어쓰기가 없는 문장 임의로 만들기
print(new_sent)

# from pykospacing import Spacing
#
# spacing = Spacing()
# kospacing_sent = spacing(new_sent)
# print(sent)
# print(kospacing_sent)


# pip install git+https://github.com/ssut/py-hanspell.git

from hanspell import spell_checker

sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "
spelled_sent = spell_checker.check(sent)

hanspell_sent = spelled_sent.checked
print(hanspell_sent)

spelled_sent = spell_checker.check(new_sent)
hanspell_sent = spelled_sent.checked
print(hanspell_sent)
# print(kospacing_sent)  # 앞서 사용한 kospacing 패키지에서 얻은 결과

from konlpy.tag import Okt

tokenizer = Okt()
print(tokenizer.morphs("에이비식스 이대휘 1월 최애돌 기부 요정"))


# pip install soynlp

import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor


# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt",
#     filename="../data/2016-10-20.txt",
# )

# 훈련 데이터를 다수의 문서로 분리
corpus = DoublespaceLineCorpus("../data/2016-10-20.txt")
print(len(corpus))

i = 0
for document in corpus:
    if len(document) > 0:
        print(document)
        i = i + 1
    if i == 3:
        break

word_extractor = WordExtractor()
word_extractor.train(corpus)
word_score_table = word_extractor.extract()

print(word_score_table["반포한"].cohesion_forward)
print(word_score_table["반포한강"].cohesion_forward)
print(word_score_table["반포한강공"].cohesion_forward)
print(word_score_table["반포한강공원"].cohesion_forward)
print(word_score_table["반포한강공원에"].cohesion_forward)
print(word_score_table["디스"].right_branching_entropy)
print(word_score_table["디스플"].right_branching_entropy)
print(word_score_table["디스플레"].right_branching_entropy)
print(word_score_table["디스플레이"].right_branching_entropy)

from soynlp.tokenizer import LTokenizer

scores = {word: score.cohesion_forward for word, score in word_score_table.items()}
l_tokenizer = LTokenizer(scores=scores)
print(l_tokenizer.tokenize("국제사회와 우리의 노력들로 범죄를 척결하자", flatten=False))

from soynlp.tokenizer import MaxScoreTokenizer

maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
print(maxscore_tokenizer.tokenize("국제사회와우리의노력들로범죄를척결하자"))


# 4. SOYNLP를 이용한 반복되는 문자 정제
from soynlp.normalizer import *

print(emoticon_normalize("앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ", num_repeats=2))
print(emoticon_normalize("앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠ", num_repeats=2))
print(emoticon_normalize("앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠ", num_repeats=2))
print(emoticon_normalize("앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠㅠㅠ", num_repeats=2))

print(repeat_normalize("와하하하하하하하하하핫", num_repeats=2))
print(repeat_normalize("와하하하하하하핫", num_repeats=2))
print(repeat_normalize("와하하하하핫", num_repeats=2))


# pip install customized_konlpy

okt = Okt()
print(okt.morphs("은경이는 사무실로 갔습니다."))
# okt.add_dictionary("은경이", "Noun")
# okt.morphs("은경이는 사무실로 갔습니다.")
