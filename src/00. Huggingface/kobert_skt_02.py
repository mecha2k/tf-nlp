import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import datasets
import os
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import BertModel, get_scheduler
from transformers.utils import logging
from transformers.optimization import get_cosine_schedule_with_warmup
from kobert_tokenizer import KoBERTTokenizer
from datasets import load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


logging.set_verbosity_error()
print(datasets.list_metrics())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")


epochs = 3
batch_size = 32
learning_rate = 1e-5
warmup_ratio = 0.1
max_seq_len = 64
model_file = "../data/models/kobert_skt_02.pt"

model = BertModel.from_pretrained("skt/kobert-base-v1")
tokenizer = KoBERTTokenizer.from_pretrained(
    "skt/kobert-base-v1", sp_model_kwargs={"nbest_size": -1, "alpha": 0.6, "enable_sampling": True}
)

# df = pd.read_table("../data/ratings_train.txt")
# df = df[:10]
# print(df.iloc[0]["document"])
#
# tokens = tokenizer(
#     df["document"].tolist(),
#     return_tensors="pt",
#     max_length=max_seq_len,
#     padding="max_length",
#     truncation=True,
# )
# print(tokens["input_ids"][0].shape)


class NsmcDataset(Dataset):
    def __init__(self, filename, tokenizer):
        self.df = pd.read_table(filename, encoding="utf-8")
        self.df = self.df.drop_duplicates(subset=["document"])
        self.df = self.df.dropna(how="any")
        self.num_labels = self.df["label"].value_counts().shape[0]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.df.iloc[idx]["document"],
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(tokens["token_type_ids"], dtype=torch.long),
            "labels": torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long),
        }


train_dataset = NsmcDataset("../data/ratings_train.txt", tokenizer)
test_dataset = NsmcDataset("../data/ratings_test.txt", tokenizer)
print(train_dataset[0]["input_ids"].shape)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

inputs = next(iter(train_loader))
labels = inputs.pop("labels")
print(inputs.keys())
print(inputs["input_ids"].shape)
print(inputs["attention_mask"][0])
print(inputs["token_type_ids"][0])
print(tokenizer.decode(inputs["input_ids"][0]))
print(labels.shape)

model.to(device)
training_steps = epochs * len(train_loader)
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_steps
)


class BertModel(nn.Module):
    def __init__(self, model, num_labels):
        super(BertModel, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_labels)

    def forward(self, **kwargs):
        _, output = self.model(**kwargs, return_dict=False)
        output = self.dropout(output)
        return self.fc(output)


if os.path.exists(model_file):
    bert_model = torch.load(model_file).to(device)
    print("Model loaded")
else:
    bert_model = BertModel(model, num_labels=train_dataset.num_labels).to(device)
    print("Model created")

optimizer = AdamW(bert_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

training_steps = epochs * len(train_loader)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(training_steps * warmup_ratio),
    num_training_steps=training_steps,
)


def compute_metrics(preds, labels):
    preds = torch.argmax(preds, dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# _, output = model(**inputs, return_dict=False)
# output = nn.Dropout(0.1)(output)
# output = nn.Linear(768, train_dataset.num_labels)(output)
# loss = loss_fn(output, labels)
# print(type(output))
# print(output.shape)
# print(loss)


for epoch in tqdm(range(epochs)):
    model.train()
    train_acc = 0
    for step, inputs in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.pop("labels")
        optimizer.zero_grad()
        outputs = bert_model(**inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        metric = compute_metrics(outputs, labels)
        train_acc += metric["accuracy"]
        if step % 100 == 0:
            print(
                f"epoch: {epoch:3d}, step: {step:5d}, loss: {loss.item():10.5f}, train_acc: {train_acc / (step + 1):10.5f}"
            )

    model.eval()
    metric = load_metric("accuracy")
    print(metric.inputs_description)
    for step, inputs in enumerate(test_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.pop("labels")
        outputs = bert_model(**inputs)
        predictions = torch.argmax(outputs, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
    score = metric.compute()
    print(f"epoch: {epoch:3d}, test_acc: {score['accuracy']*100:6.2f}%")

torch.save(bert_model, model_file)


def predict(sentence):
    tokens = tokenizer(sentence, max_length=max_seq_len, padding="max_length", truncation=True)
    with torch.no_grad():
        tokens = {k: torch.tensor(v).unsqueeze(dim=0).to(device) for k, v in tokens.items()}
        outputs = bert_model(**tokens)
        predictions = torch.argmax(outputs, dim=-1)
        return predictions.cpu().numpy()


bert_model = torch.load(model_file).to(device)
bert_model.eval()

sentiment_dict = {0: "부정", 1: "긍정"}
samples = DataLoader(train_dataset, batch_size=10, shuffle=True)
samples = next(iter(samples))["input_ids"]
print(samples.shape)
for sample in samples:
    sentence = tokenizer.decode(sample, skip_special_tokens=True)
    print(sentence, ",  판정 : ", sentiment_dict[predict(sentence)[0]])
    print("-" * 120)
