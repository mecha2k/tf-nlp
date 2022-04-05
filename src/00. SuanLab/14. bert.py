import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from transformers import AutoTokenizer, TFAutoModel, logging
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tqdm import tqdm


np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)

plt.style.use("seaborn")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False

logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


train_data = pd.read_table("../data/ratings_train.txt")
test_data = pd.read_table("../data/ratings_test.txt")

train_data.drop_duplicates(subset=["document"], inplace=True)
test_data.drop_duplicates(subset=["document"], inplace=True)

train_data.dropna(how="any", inplace=True)
test_data.dropna(how="any", inplace=True)

print(train_data["label"].value_counts())
print(train_data.isna().sum())
print(train_data.head())


epochs = 3
batch_size = 64
max_len = 40
num_classes = 2


model_name = "klue/bert-base"
cache_dir = "../data/klue-bert-base"
weights_file = "../data/bert_nsmc_klue.h5"
train_np_file = "../data/bert_klue_train.npy"
test_np_file = "../data/bert_klue_test.npy"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, do_lower_case=False)
model = TFAutoModel.from_pretrained(model_name, cache_dir=cache_dir, from_pt=True)


# model_name = "bert-base-multilingual-cased"
# cache_dir = "../data/bert_multi"
# weights_file = "../data/bert_base_multi.h5"
# train_np_file = "../data/bert_base_train.npy"
# test_np_file = "../data/bert_base_test.npy"

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, do_lower_case=False)
# model = TFAutoModel.from_pretrained(model_name, cache_dir=cache_dir)


def bert_tokenizer(sentence, max_len=max_len):
    encoded_sentence = tokenizer.encode_plus(
        text=sentence,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    input_ids = encoded_sentence["input_ids"]
    attention_masks = encoded_sentence["attention_mask"]
    token_type_ids = encoded_sentence["token_type_ids"]
    return input_ids, attention_masks, token_type_ids


def prepare_nsmc_tokens(data, filename):
    input_ids, attention_masks, token_type_ids = [], [], []
    for sentence in tqdm(data["document"], total=len(data)):
        input_id, attention_mask, token_type_id = bert_tokenizer(sentence)
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)

    input_ids = np.array(input_ids, dtype=np.int32)
    attention_masks = np.array(attention_masks, dtype=np.int32)
    token_type_ids = np.array(token_type_ids, dtype=np.int32)
    np.save(filename, (input_ids, attention_masks, token_type_ids))


filename = train_np_file
if not os.path.exists(filename):
    prepare_nsmc_tokens(train_data, filename)
input_ids, attention_masks, token_type_ids = np.load(filename)

train_inputs = (input_ids, attention_masks, token_type_ids)
train_labels = train_data["label"].to_numpy()
print(f"Sentences: {len(input_ids)}\tLabels: {len(train_labels)}")

index = 5
input_id = input_ids[index]
attention_mask = attention_masks[index]
token_type_id = token_type_ids[index]
print(input_id)
print(attention_mask)
print(token_type_id)
print(tokenizer.decode(input_id))

input_ids = Input(shape=(max_len,), dtype="int32", name="input_ids")
attention_masks = Input(shape=(max_len,), dtype="int32", name="attention_masks")
token_type_ids = Input(shape=(max_len,), dtype="int32", name="token_type_ids")
embeddings = model(input_ids, attention_masks, token_type_ids)[1]
outputs = Dropout(model.config.hidden_dropout_prob)(embeddings)
kernel_initializer = tf.keras.initializers.TruncatedNormal(model.config.initializer_range)
outputs = Dense(num_classes, kernel_initializer=kernel_initializer)(outputs)

classification_model = Model(inputs=[input_ids, attention_masks, token_type_ids], outputs=outputs)

for layer in classification_model.layers:
    print(layer)
print(classification_model.layers[3])
# classification_model.layers[3].trainable = False
classification_model.summary()
plot_model(classification_model, to_file="images/14-cls_bert.png", show_shapes=True)


# class TFBertClassification(tf.keras.Model):
#     def __init__(self, num_class):
#         super().__init__()
#         self.bert = TFAutoModel.from_pretrained(
#             "bert-base-multilingual-cased", cache_dir="../data/bert_multi"
#         )
#         self.dropout = Dropout(self.bert.config.hidden_dropout_prob)
#         self.classifier = Dense(
#             num_class,
#             kernel_initializer=tf.keras.initializers.TruncatedNormal(
#                 self.bert.config.initializer_range
#             ),
#         )
#
#     def call(self, input_ids, attention_mask=None, token_type_ids=None):
#         outputs = self.bert(input_ids, attention_mask, token_type_ids)
#         outputs = outputs[1]
#         outputs = self.dropout(outputs, training=False)
#         outputs = self.classifier(outputs)
#         return outputs
#
#
# classification_model = TFBertClassification(num_class=num_classes)


optimizer = tf.keras.optimizers.Adam(5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
classification_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


callbacks = [
    EarlyStopping(monitor="val_accuracy", min_delta=0.0001, patience=2),
    ModelCheckpoint(
        weights_file,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    ),
]

history = classification_model.fit(
    x=train_inputs,
    y=train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=callbacks,
)

names = ["loss", "accuracy"]
plt.figure(figsize=(10, 5))
for i, name in enumerate(names):
    plt.subplot(1, 2, i + 1)
    plt.plot(history.history[name])
    plt.plot(history.history[f"val_{name}"])
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.legend([name, f"val_{name}"])
plt.savefig("images/14-bert_nsmc_loss", dpi=300)


filename = test_np_file
if not os.path.exists(filename):
    prepare_nsmc_tokens(test_data, filename)
input_ids, attention_masks, token_type_ids = np.load(filename)

test_inputs = (input_ids, attention_masks, token_type_ids)
test_labels = test_data["label"].to_numpy()
print(f"Test Sentences: {len(input_ids)}\tLabels: {len(test_labels)}")

classification_model.load_weights(weights_file)

results = classification_model.evaluate(x=test_inputs, y=test_labels, batch_size=1024)
print("test loss, test acc: ", results)
