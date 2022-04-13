from pororo import Pororo
import docx
import warnings

warnings.filterwarnings("ignore")
mt = Pororo(task="translation", lang="multi")

doc = docx.Document("../data/Preface.docx")
print(len(doc.paragraphs))

for paras in doc.paragraphs:
    print(paras.text)
    print(mt(paras.text, src="en", tgt="ko"))
    print("-" * 50)
