import torch

from transformers import GPT2Model, GPT2LMHeadModel, GPT2ForSequenceClassification
from transformers import GPT2TokenizerFast, GPT2Config
from transformers.utils import logging

logging.set_verbosity_error()

config = GPT2Config()
model = GPT2Model(config)
config = model.config

special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", **special_tokens)
print(tokenizer.all_special_tokens)

max_length = 128
model = GPT2ForSequenceClassification.from_pretrained("gpt2", max_length=max_length)
print(model.config)

sentences = "Hello, my dog is cute!?"
inputs = tokenizer(sentences, max_length=max_length, padding="max_length", return_tensors="pt")
outputs = model(**inputs)
print(outputs)

sentences = [
    "Hello, my dog is cute!?",
    "GPT-2 huggingface transformers is awesome!",
    "Macbook is the best!",
]


inputs = tokenizer(sentences, max_length=max_length, padding="max_length", return_tensors="pt")
print(inputs.keys())
print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
print(inputs["input_ids"].shape)
print(inputs["attention_mask"].shape)
print(inputs["input_ids"][0])


outputs = model(inputs["input_ids"][0])
# print(outputs.logits.shape)
# print(outputs.attentions)
# print(outputs.hidden_states)

# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.eval()
# print(model.config)
#
# outputs = model(**inputs, labels=inputs["input_ids"])
# print(outputs.loss.item())
# print(outputs.logits.shape)
#
#
# model = GPT2ForSequenceClassification.from_pretrained("gpt2")
# with torch.no_grad():
#     logits = model(**inputs).logits
# predictions = logits.argmax().item()
# print(logits.shape)
# print(predictions)
# print(model.config.id2label[predictions])
#
#
# num_labels = len(model.config.id2label) + 5
# model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels)
#
# labels = torch.tensor(5)
# outputs = model(**inputs, labels=labels)
# print(num_labels)
# print(outputs.logits.shape)
# print(round(outputs.loss.item(), 2))
