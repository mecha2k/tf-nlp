import transformers
from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer

print(transformers.__version__)

model = TFBertForMaskedLM.from_pretrained("klue/bert-base", from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

inputs = tokenizer("축구는 정말 재미있는 [MASK]다.", return_tensors="tf")

print(tokenizer.cls_token_id)
print(tokenizer.sep_token_id)
print(tokenizer.mask_token_id)
print(inputs["input_ids"])
print(inputs["token_type_ids"])
print(inputs["attention_mask"])

from transformers import FillMaskPipeline

pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

print(pip("축구는 정말 재미있는 [MASK]다."))
print(pip("어벤져스는 정말 재미있는 [MASK]다."))
print(pip("나는 오늘 아침에 [MASK]에 출근을 했다."))
print(pip("사람은 [MASK]히 살아야 한다."))
