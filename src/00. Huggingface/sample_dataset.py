from transformers import AutoTokenizer
from datasets import load_dataset, list_metrics, load_metric


dataset = load_dataset("rotten_tomatoes", split="train")
print(dataset[0])
print(len(dataset))
print(dataset[0]["text"])
print(dataset["text"][0])
print(dataset["label"][0])


max_seq_len = 128

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(tokenizer.tokenize("I love this movie."))
tokens = tokenizer(dataset[0]["text"], return_tensors="pt")
print(tokens["input_ids"].shape)

tokenization = lambda x: tokenizer(x["text"])
dataset = dataset.map(tokenization, batched=True)
print(dataset[0])
print(dataset.format["type"])

dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
print(dataset[0])
print(dataset.format["type"])


metrics_list = list_metrics()
len(metrics_list)
print(metrics_list)

metric = load_metric("glue", "mrpc")
print(metric.inputs_description)
