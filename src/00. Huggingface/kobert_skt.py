import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer

model = BertModel.from_pretrained("skt/kobert-base-v1")
tokenizer = KoBERTTokenizer.from_pretrained(
    "skt/kobert-base-v1", sp_model_kwargs={"nbest_size": -1, "alpha": 0.6, "enable_sampling": True}
)

text = "한국어 모델을 공유합니다."
tokens = tokenizer.encode(text)
outputs = tokenizer.decode(tokens)
print(tokens)
print(outputs)

inputs = tokenizer.batch_encode_plus([text])
outputs = model(
    input_ids=torch.tensor(inputs["input_ids"]),
    attention_mask=torch.tensor(inputs["attention_mask"]),
)
print(inputs)
print(outputs.pooler_output.shape)
