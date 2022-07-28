import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel
from transformers.utils import logging
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


class NsmcDataset(Dataset):
    def __init__(self, filename, tokenizer):
        self.df = pd.read_table(filename)
        self.df = self.df[:1000]
        self.df = self.df.drop_duplicates(subset=["document"])
        self.df = self.df.dropna(how="any")
        self.num_labels = self.df["label"].value_counts().shape[0]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode_plus(
            self.df.iloc[idx]["document"],
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
        )
        inputs = {
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(tokens["token_type_ids"], dtype=torch.long),
        }
        label = torch.tensor(self.df.iloc[idx]["label"])
        return inputs, label


train_dataset = NsmcDataset("../data/ratings_train.txt", tokenizer)
test_dataset = NsmcDataset("../data/ratings_test.txt", tokenizer)
print(train_dataset[0][0]["input_ids"].shape)
print(train_dataset[0][1])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

inputs, labels = next(iter(train_loader))
print(inputs["input_ids"].shape)
print(tokenizer.decode(inputs["input_ids"][0]))


class BertModel(nn.Module):
    def __init__(self, model, num_labels):
        super(BertModel, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, output = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        output = self.dropout(output)
        return self.fc(output)


bert_model = BertModel(model, num_labels=train_dataset.num_labels).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
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

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

optimizer = AdamW(grouped_parameters, lr=learning_rate)
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


for epoch in range(epochs):
    model.train()
    for step, (inputs, labels) in enumerate(train_loader):
        input_ids, attention_mask, token_type_ids = tuple(x.to(device) for x in inputs.values())
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = bert_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")
    # model.eval()
#     # with torch.no_grad():
#     #     train_acc = calc_accuracy(model(**inputs), labels)
#     #     print(f"epoch: {epoch}, train_acc: {train_acc}")
#     #     test_acc = calc_accuracy(model(**inputs), labels)
#     #     print(f"epoch: {epoch}, test_acc: {test_acc}")
#     # torch.save(model.state_dict(), f"model/model_{epoch}.pt")
#     # torch.save(tokenizer.state_dict(), f"model/tokenizer_{epoch}.pt")
#     # torch.save(scheduler.state_dict(), f"model/scheduler_{epoch}.pt")
#     # torch.save(optimizer.state_dict(), f"model/optimizer_{epoch}.pt")
#     # torch.save(loss_fn.state_dict(), f"model/loss_fn_{epoch}.pt")
#     # torch.save(train_acc, f"model/train_acc_{epoch}.pt")
#     # torch.save(test_acc, f"model/test_acc_{epoch}.pt")
#     # torch.save(model, f"model/model_{epoch}.pt")
#     # torch.save(tokenizer, f"model/tokenizer_{epoch}.pt")
#     # torch.save(scheduler, f"model/scheduler_{epoch}.pt")
#     # torch.save(optimizer, f"model/optimizer_{epoch}.pt")
#     # torch.save(loss_fn,

#
# for e in range(num_epochs):
#     train_acc = 0.0
#     test_acc = 0.0
#     model.train()
#     for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(
#         tqdm_notebook(train_dataloader)
#     ):
#         optimizer.zero_grad()
#         token_ids = token_ids.long().to(device)
#         segment_ids = segment_ids.long().to(device)
#         valid_length = valid_length
#         label = label.long().to(device)
#         out = model(token_ids, valid_length, segment_ids)
#         loss = loss_fn(out, label)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#         optimizer.step()
#         scheduler.step()  # Update learning rate schedule
#         train_acc += calc_accuracy(out, label)
#         if batch_id % log_interval == 0:
#             print(
#                 "epoch {} batch id {} loss {} train acc {}".format(
#                     e + 1, batch_id + 1, loss.data.cpu().numpy(), train_acc / (batch_id + 1)
#                 )
#             )
#     print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
#     model.eval()
#     for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(
#         tqdm_notebook(test_dataloader)
#     ):
#         token_ids = token_ids.long().to(device)
#         segment_ids = segment_ids.long().to(device)
#         valid_length = valid_length
#         label = label.long().to(device)
#         out = model(token_ids, valid_length, segment_ids)
#         test_acc += calc_accuracy(out, label)
#     print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))

# X_train = tokenizer(X_train_list, max_length=max_seq_len, padding="max_length")
# # max_length=max_seq_len, padding="max_length", truncation=True
# X_test = tokenizer(X_test_list, max_length=max_seq_len, padding="max_length")


# text = "한국어 모델을 공유합니다."
# tokens = tokenizer.encode(text)
# outputs = tokenizer.decode(tokens)
# print(tokens)
# print(outputs)
#
# inputs = tokenizer.batch_encode_plus([text])
# outputs = model(
#     input_ids=torch.tensor(inputs["input_ids"]),
#     attention_mask=torch.tensor(inputs["attention_mask"]),
# )
# print(inputs)
# print(outputs.pooler_output.shape)
