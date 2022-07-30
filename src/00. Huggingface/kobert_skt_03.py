import torch
import torch.nn as nn
import pandas as pd
import os
from transformers.utils import logging
from kobert_tokenizer import KoBERTTokenizer


logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")


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


max_seq_len = 64
model_file = "../data/models/kobert_skt_02.pt"
bert_model = torch.load(model_file).to(device)
bert_model.eval()
print("Model loaded")
tokenizer = KoBERTTokenizer.from_pretrained(
    "skt/kobert-base-v1", sp_model_kwargs={"nbest_size": -1, "alpha": 0.6, "enable_sampling": True}
)
print("Tokenizer loaded")


def predict(sentence):
    tokens = tokenizer(sentence, max_length=max_seq_len, padding="max_length", truncation=True)
    with torch.no_grad():
        tokens = {k: torch.tensor(v).unsqueeze(dim=0).to(device) for k, v in tokens.items()}
        outputs = bert_model(**tokens)
        predictions = torch.argmax(outputs, dim=-1)
        return predictions.cpu().numpy()


sentiment_dict = {0: "부정", 1: "긍정"}
sentences = pd.read_table("../data/ratings_train.txt", encoding="utf-8")
sentences = sentences.drop_duplicates(subset=["document"])
sentences = sentences.dropna(how="any")
sentences = sentences["document"].sample(n=10).tolist()
for sentence in sentences:
    print(sentence, ",  판정 : ", sentiment_dict[predict(sentence)[0]])
    print("-" * 120)
