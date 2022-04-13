import warnings

from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

warnings.filterwarnings("ignore")


output_string = StringIO()
with open("../data/Finding_Alphas.pdf", "rb") as F:
    parser = PDFParser(F)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr, output_string, laparams=LAParams(), codec="utf-8")
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
print(output_string.getvalue())

# with open("images/output.txt", mode="w") as out:
#     for line in tqdm(tokenized_data, unit=" line"):
#         out.write(" ".join(line) + "\n")
