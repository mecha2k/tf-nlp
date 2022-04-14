from pororo import Pororo
import docx
import warnings

warnings.filterwarnings("ignore")
mt = Pororo(task="translation", lang="multi")

doc = docx.Document("../data/Preface.docx")
print(len(doc.paragraphs))

for idx, paras in enumerate(doc.paragraphs):
    sentence = (paras.text).strip()
    if not sentence:
        continue
    print(idx)
    print(sentence)
    print(mt(sentence, src="en", tgt="ko"))
    print("-" * 30)