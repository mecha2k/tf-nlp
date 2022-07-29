import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    logging,
    get_scheduler,
)
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm

logging.set_verbosity_error()

dataset = load_dataset("yelp_review_full")
print(dataset)
print(dataset["train"][100])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

tokenization = lambda x: tokenizer(x["text"], padding="max_length", truncation=True)
token_datasets = dataset.map(tokenization, batched=True)
print(token_datasets["train"][0].keys())

small_train_dataset = token_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = token_datasets["test"].shuffle(seed=42).select(range(1000))

metric = load_metric("accuracy")
training_args = TrainingArguments(output_dir="../data/hugging_trainer", evaluation_strategy="epoch")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
# trainer.train()


del model
del trainer
torch.cuda.empty_cache()

token_datasets = token_datasets.remove_columns(["text"])
token_datasets = token_datasets.rename_column("label", "labels")

token_datasets.set_format("torch")
small_train_dataset = token_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = token_datasets["test"].shuffle(seed=42).select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, shuffle=False, batch_size=8)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

num_epochs = 1
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.train()
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
