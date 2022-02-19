from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer
from icecream import ic

model = TFBertForMaskedLM.from_pretrained("bert-large-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

inputs = tokenizer("Soccer is a really fun [MASK].", return_tensors="tf")

print(tokenizer.cls_token_id)
print(tokenizer.sep_token_id)
print(tokenizer.mask_token_id)
print(inputs)
print(inputs["input_ids"])
print(inputs["token_type_ids"])
print(inputs["attention_mask"])

from transformers import FillMaskPipeline

pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

print(pip("Soccer is a really fun [MASK]."))
print(pip("The Avengers is a really fun [MASK]."))
print(pip("I went to [MASK] this morning."))
