# !pip install transformers

from transformers import TFBertForMaskedLM

model = TFBertForMaskedLM.from_pretrained("bert-large-uncased")

from transformers import AutoTokenizer

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

pip("Soccer is a really fun [MASK].")
pip("The Avengers is a really fun [MASK].")
pip("I went to [MASK] this morning.")
