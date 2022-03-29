from konlpy.tag import Mecab
import platform

if platform.system() == "Windows":
    mecab = Mecab(dicpath="c:/mecab/mecab-ko-dic")
else:
    mecab = Mecab()


sent = "언제나 현재에 집중하면 행복할 것이다."
print(mecab.pos(sent))
print(mecab.morphs(sent))
print(mecab.nouns(sent))
