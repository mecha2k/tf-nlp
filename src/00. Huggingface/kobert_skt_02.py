import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import BertModel
from transformers.utils import logging
from transformers.optimization import get_cosine_schedule_with_warmup
from kobert_tokenizer import KoBERTTokenizer


logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")


epochs = 1
batch_size = 32
learning_rate = 1e-5
warmup_ratio = 0.1
max_seq_len = 64


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
        self.df = pd.read_table(filename)
        self.df = self.df[:32]
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
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

inputs = next(iter(train_loader))
labels = inputs.pop("labels")
print(inputs.keys())
print(inputs["input_ids"].shape)
print(inputs["attention_mask"][0])
print(inputs["token_type_ids"][0])
print(tokenizer.decode(inputs["input_ids"][0]))
print(labels.shape)

model.to(device)
traing_steps = epochs * len(train_loader)
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=int(traing_steps * warmup_ratio), num_training_steps=traing_steps
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


bert_model = BertModel(model, num_labels=train_dataset.num_labels).to(device)
optimizer = AdamW(bert_model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

training_steps = epochs * len(train_loader)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(training_steps * warmup_ratio),
    num_training_steps=training_steps,
)


def calc_accuracy(x, y):
    max_vals, max_indices = torch.max(x, 1)
    train_acc = (max_indices == y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


# _, output = model(**inputs, return_dict=False)
# output = nn.Dropout(0.1)(output)
# output = nn.Linear(768, train_dataset.num_labels)(output)
# loss = loss_fn(output, labels)
# print(type(output))
# print(output.shape)
# print(loss)


progress_bar = tqdm(range(training_steps))
for epoch in range(epochs):
    model.train()
    step, train_acc, test_acc = 0, 0, 0
    for step, inputs in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.pop("labels")
        optimizer.zero_grad()
        outputs = bert_model(**inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_acc += calc_accuracy(outputs, labels)
        progress_bar.update(1)
        if step % 2 == 0:
            print(
                f"epoch: {epoch}, step: {step}, loss: {loss.item()}, train_acc: {train_acc / (step + 1)}"
            )
    print(f"epoch: {epoch}, train_acc: {train_acc / (step + 1) * 100:.2f}%")

    model.eval()
    for step, inputs in enumerate(test_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.pop("labels")
        outputs = bert_model(**inputs)
        test_acc += calc_accuracy(outputs, labels)
    print(f"epoch: {epoch}, test_acc: {test_acc / (step + 1) * 100:.2f}%")
