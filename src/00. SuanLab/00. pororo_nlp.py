from unittest.mock import sentinel
from pororo import Pororo
import warnings

warnings.filterwarnings("ignore")
# print(Pororo.available_tasks())

ner = Pororo(task="ner", lang="ko")
results = ner("마이클 제프리 조던(영어: Michael Jeffrey Jordan, 1963년 2월 17일 ~ )은 미국의 은퇴한 농구 선수이다.")
# print(results)

mt = Pororo(task="translation", lang="multi")
print(mt("케빈은 아직도 일을 하고 있다.", src="ko", tgt="en"))

sentence = "Kevin is still working."
print(mt(sentence, src="en", tgt="ko"))

sentence = """
Yandex.Translate is a mobile and web service that translates words, phrases, whole texts, and entire
websites from English into Korean. The meanings of individual words come complete with examples of 
usage, transcription, and the possibility to hear pronunciation. In site translation mode, 
Yandex.Translate will translate the entire text content of the site at the URL you provide. 
Knows not just English and Korean, but 98 other languages as well.
"""
print(mt(sentence, src="en", tgt="ko"))
