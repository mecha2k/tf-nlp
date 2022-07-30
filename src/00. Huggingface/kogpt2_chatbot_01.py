import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import datasets
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, get_scheduler
from transformers.utils import logging
import re
import os


epochs = 0
batch_size = 64
learning_rate = 1e-5
warmup_ratio = 0.1
s_neg = -1e18
max_seq_len = 32
model_file = "../data/models/kogpt2.pt"


logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")

special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", **special_tokens)

print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)
print("-" * 150)
print(tokenizer.decode(0))
print(tokenizer.decode(1))
print(tokenizer.decode(3))
print("-" * 150)

sentences = [
    "But what about second breakfast?",
    "15일 한국부동산원에 따르면 올해 2월 주택종합(아파트·연립주택·단독주택) 매매가격 동향을 조사한 결과",
    "서울은 전월 대비 0.04%, 수도권은 0.03% 하락했다.",
    "코로나가 심각합니다.",
]
encoded = tokenizer(sentences, padding="max_length", truncation=True)
print(encoded)
print(tokenizer.decode(encoded["input_ids"][1]))
print(encoded[3].tokens)
print(tokenizer.tokenize(sentences[3]))
print("-" * 150)

vocab = tokenizer.get_vocab()
print(sorted(vocab.items(), key=lambda x: x[1])[:10])
print(len(vocab))

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print("-" * 150)


usr_token = "<usr>"
sys_token = "<sys>"
bos_token = tokenizer.bos_token
eos_token = tokenizer.eos_token
mask_token = tokenizer.mask_token

df = pd.read_csv("../data/ChatBotData.csv", encoding="utf-8")
df.drop_duplicates(subset=["Q", "A"], inplace=True)
df.dropna(how="any", inplace=True)

df = df.sample(n=1)
question = re.sub(r"([?.!,])", r" ", df["Q"].values[0])
answer = re.sub(r"([?.!,])", r" ", df["A"].values[0])
sentiment = df["label"].values[0]
qus_token = tokenizer.tokenize(bos_token + question + usr_token)
ans_token = tokenizer.tokenize(sys_token + answer + eos_token)
qus_len = len(qus_token)
ans_len = len(ans_token)
print(qus_token)
print(ans_token)

if qus_len + ans_len > max_seq_len:
    ans_len = max_seq_len - qus_len
    if ans_len <= 0:
        qus_token = qus_token[-(int(max_seq_len / 2)) :]
        qus_len = len(qus_token)
        ans_len = max_seq_len - qus_len
        assert ans_len > 0
    ans_token = ans_token[:ans_len]
    ans_len = len(ans_token)
    assert ans_len == len(ans_token), f"{a_len} ==? {len(a_toked)}"

labels = [mask_token] * qus_len + ans_token[1:]
labels_ids = tokenizer.convert_tokens_to_ids(labels)
while len(labels_ids) < max_seq_len:
    labels_ids += [tokenizer.pad_token_id]
attention_mask = [0] * qus_len + [1] * ans_len + [0] * (max_seq_len - qus_len - ans_len)
input_ids = tokenizer.convert_tokens_to_ids(qus_token + ans_token)
while len(input_ids) < max_seq_len:
    input_ids += [tokenizer.pad_token_id]

print(f"question : {question}")
print(f"qus_token : {qus_token}")
print(f"answer : {answer}")
print(f"ans_token : {ans_token}")
print(f"labels : {labels}")


class ChatDataset(Dataset):
    def __init__(self, filename, tokenizer):
        self.df = pd.read_csv(filename, encoding="utf-8")
        self.df.drop_duplicates(subset=["Q", "A"], inplace=True)
        self.df.dropna(how="any", inplace=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df = self.df.iloc[idx]
        question = re.sub(r"([?.!,])", r" ", df.Q)
        answer = re.sub(r"([?.!,])", r" ", df.A)
        sentiment = df.label
        qus_token = tokenizer.tokenize(bos_token + question + usr_token)
        ans_token = tokenizer.tokenize(sys_token + answer + eos_token)
        qus_len = len(qus_token)
        ans_len = len(ans_token)

        if qus_len + ans_len > max_seq_len:
            ans_len = max_seq_len - qus_len
            if ans_len <= 0:
                qus_token = qus_token[-(int(max_seq_len / 2)) :]
                qus_len = len(qus_token)
                ans_len = max_seq_len - qus_len
                assert ans_len > 0
            ans_token = ans_token[:ans_len]
            ans_len = len(ans_token)
            assert ans_len == len(ans_token), f"{a_len} ==? {len(a_toked)}"

        labels = [mask_token] * qus_len + ans_token[1:]
        labels_ids = tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < max_seq_len:
            labels_ids += [tokenizer.pad_token_id]
        attention_mask = [0] * qus_len + [1] * ans_len + [0] * (max_seq_len - qus_len - ans_len)
        input_ids = tokenizer.convert_tokens_to_ids(qus_token + ans_token)
        while len(input_ids) < max_seq_len:
            input_ids += [tokenizer.pad_token_id]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(labels_ids, dtype=torch.long),
            # sentiment : 일상다반사 0, 이별(부정) 1, 사랑(긍정) 2
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
        }


dataset = ChatDataset("../data/ChatBotData.csv", tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

inputs = next(iter(dataloader))
print(inputs.keys())
print(inputs["input_ids"][0])
print(inputs["attention_mask"][0])
print(inputs["token_type_ids"][0])
print(tokenizer.decode(inputs["input_ids"][0]))


if os.path.exists(model_file):
    model = torch.load(model_file).to(device)
    print("Model loaded")
else:
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2").to(device)
    print("Model created")


text = "근육이 커지기 위해서는"
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
gen_ids = model.generate(
    input_ids,
    max_length=128,
    repetition_penalty=2.0,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    use_cache=True,
)
generated = tokenizer.decode(gen_ids[0])
print(generated)


training_steps = epochs * len(dataloader)
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(reduction="none")
scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_steps
)


def compute_metrics(preds, labels):
    preds = torch.argmax(preds, dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# inputs = next(iter(dataloader))
# inputs = {k: v.to(device) for k, v in inputs.items()}
# outputs = model(inputs["input_ids"])
# outputs = outputs.logits
# mask_3d = (
#     inputs["attention_mask"].unsqueeze(dim=2).repeat_interleave(repeats=outputs.shape[2], dim=2)
# )
# mask_out = torch.where(mask_3d == 1, outputs, s_neg * torch.ones_like(outputs))
# loss = loss_fn(mask_out.transpose(2, 1), inputs["token_type_ids"])
# # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
# loss = loss.sum() / inputs["attention_mask"].sum()
# loss.backward()
# optimizer.step()


model.train()
for epoch in tqdm(range(epochs)):
    loss_sum = 0
    for step, inputs in enumerate(dataloader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        sentiment = inputs.pop("sentiment")
        optimizer.zero_grad()
        outputs = model(inputs["input_ids"])
        outputs = outputs.logits
        mask_3d = (
            inputs["attention_mask"]
            .unsqueeze(dim=2)
            .repeat_interleave(repeats=outputs.shape[2], dim=2)
        )
        mask_out = torch.where(mask_3d == 1, outputs, s_neg * torch.ones_like(outputs))
        loss = loss_fn(mask_out.transpose(2, 1), inputs["token_type_ids"])
        loss = loss.sum() / inputs["attention_mask"].sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += loss.item()
        if step % 10 == 0:
            print(f"[epoch: {epoch:>2}, step: {step:5d}], cost = {loss_sum/(step+1):>.9}", end="\n")

torch.save(model, model_file)

#             while 1:
#                 input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
#                 pred = self(input_ids)
#                 gen = tok.convert_ids_to_tokens(
#                     torch.argmax(
#                         pred,
#                         dim=-1).squeeze().numpy().tolist())[-1]
#                 if gen == EOS:
#                     break
#                 a += gen.replace('▁', ' ')
#             print("Simsimi > {}".format(a.strip()))

# #
# q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
# q_len = len(q_toked)
#
# a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
# a_len = len(a_toked)
#     input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)

# usr_token = "<usr>"
# sys_token = "<sys>"
# bos_token = tokenizer.bos_token
# eos_token = tokenizer.eos_token
# mask_token = tokenizer.mask_token
# qus_token = tokenizer.tokenize(bos_token + question + usr_token)
# ans_token = tokenizer.tokenize(sys_token + answer + eos_token)


def chatbot(text):
    question = tokenizer.tokenize(text)
    question = " ".join(question)
    with torch.no_grad():
        answer = ""
        while True:
            input_ids = bos_token + str(question) + usr_token + sys_token + str(answer)
            input_ids = tokenizer.encode(input_ids, return_tensors="pt").unsqueeze(dim=0)
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            outputs = model(input_ids)
            outputs = outputs.logits
            generated = tokenizer.convert_ids_to_tokens(
                torch.argmax(outputs, dim=-1).squeeze().cpu().numpy()
            )[-1]
            if generated == eos_token:
                break
            answer += generated.replace("▁", "")
    return answer.strip()


model = torch.load(model_file).to(device)
model.eval()

# print(chatbot("안녕하세요"))

# sentiment_dict = {0: "부정", 1: "긍정"}
# samples = DataLoader(train_dataset, batch_size=10, shuffle=True)
# samples = next(iter(samples))["input_ids"]
# print(samples.shape)
# for sample in samples:
#     sentence = tokenizer.decode(sample, skip_special_tokens=True)
#     print(sentence, ",  판정 : ", sentiment_dict[predict(sentence)[0]])
#     print("-" * 120)

text = "오늘도 좋은 하루!"
# question = "<usr>" + text + "<sys>"
# input_ids = [tokenizer.bos_token_id] + tokenizer.encode(question)
# input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
#
#
# output = model.generate(
#     input_ids, max_length=50, early_stopping=True, eos_token_id=tokenizer.eos_token_id
# )
#
# decoded_sentence = tokenizer.decode(output[0].numpy().tolist())
#
# decoded_sentence.split("<sys> ")[1].replace("</s>", "")
#
# output = model.generate(input_ids, max_length=50, do_sample=True, top_k=10)
# tokenizer.decode(output[0].numpy().tolist())

# text = "근육이 커지기 위해서는"
# input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
# gen_ids = model.generate(
#     input_ids,
#     max_length=128,
#     repetition_penalty=2.0,
#     pad_token_id=tokenizer.pad_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     bos_token_id=tokenizer.bos_token_id,
#     use_cache=True,
# )
# generated = tokenizer.decode(gen_ids[0])
# print(generated)
