import docx
import warnings
from googletrans import Translator

warnings.filterwarnings("ignore")

translator = Translator()
doc = docx.Document("../data/Preface.docx")
print(len(doc.paragraphs))

for idx, paras in enumerate(doc.paragraphs):
    sentence = (paras.text).strip()
    if not sentence:
        continue
    print(idx)
    print(sentence)
    result = translator.translate(sentence, src="en", dest="ko")
    print(result.text)
    print("-" * 30)
