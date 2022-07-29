import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.utils import logging
from torch.optim import AdamW

logging.set_verbosity_error()

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model.train()

no_decay = ["bias", "LayerNorm.weight"]
grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(grouped_parameters, lr=1e-5)
# optimizer = AdamW(model.parameters(), lr=1e-5)

max_seq_len = 16

text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = tokenizer(
    text_batch, return_tensors="pt", max_length=max_seq_len, padding="max_length", truncation=True
)
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
print(input_ids[0].shape)
print(input_ids[0])
print(tokenizer.decode(input_ids[0]))
print(attention_mask[0])


labels = torch.tensor([1, 0]).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs[0]
print(loss)
print(outputs[1])
print(len(outputs))
loss.backward()
optimizer.step()


labels = torch.tensor([1, 0])
outputs = model(input_ids, attention_mask=attention_mask)
loss = F.cross_entropy(outputs[0], labels)
loss.backward()
optimizer.step()
print(loss)

loss_fn = nn.CrossEntropyLoss()
inputs = torch.randn(2, 12, requires_grad=True)
target = torch.zeros(2, dtype=torch.long)
output = loss_fn(inputs, target)
output = F.cross_entropy(inputs, target)
output.backward()
print(outputs)
