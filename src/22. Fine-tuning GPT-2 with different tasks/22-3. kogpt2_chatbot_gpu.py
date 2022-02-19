import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained(
    "skt/kogpt2-base-v2", bos_token="</s>", eos_token="</s>", pad_token="<pad>"
)
model = TFGPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2", from_pt=True)

print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)
print("-" * 10)
print(tokenizer.decode(1))
print(tokenizer.decode(2))
print(tokenizer.decode(3))
print(tokenizer.decode(4))

import pandas as pd
from tqdm import tqdm

# import urllib.request
# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
#     filename="../data/ChatBotData.csv",
# )
train_data = pd.read_csv("../data/ChatBotData.csv")
print(len(train_data))

batch_size = 32


def get_chat_data():
    for question, answer in zip(train_data.Q.to_list(), train_data.A.to_list()):
        bos_token = [tokenizer.bos_token_id]
        eos_token = [tokenizer.eos_token_id]
        sent = tokenizer.encode("<usr>" + question + "<sys>" + answer)
        yield bos_token + sent + eos_token


dataset = tf.data.Dataset.from_generator(get_chat_data, output_types=tf.int32)

dataset = dataset.padded_batch(
    batch_size=batch_size, padded_shapes=(None,), padding_values=tokenizer.pad_token_id
)

for batch in dataset:
    print(batch)
    tokenizer.decode(batch[0])
    print(batch[0])
    break

print(
    tokenizer.encode(
        "</s><usr> 12시 땡!<sys> 하루가 또 가네요.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>"
        "<pad><pad><pad>"
    )
)

adam = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)

steps = len(train_data) // batch_size + 1
print(steps)

EPOCHS = 3

for epoch in range(EPOCHS):
    epoch_loss = 0

    for batch in tqdm(dataset, total=steps):
        with tf.GradientTape() as tape:
            result = model(batch, labels=batch)
            loss = result[0]
            batch_loss = tf.reduce_mean(loss)

        grads = tape.gradient(batch_loss, model.trainable_variables)
        adam.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += batch_loss / steps

    print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, epoch_loss))

text = "오늘도 좋은 하루!"

sent = "<usr>" + text + "<sys>"

input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
input_ids = tf.convert_to_tensor([input_ids])

output = model.generate(
    input_ids, max_length=50, early_stopping=True, eos_token_id=tokenizer.eos_token_id
)

decoded_sentence = tokenizer.decode(output[0].numpy().tolist())

decoded_sentence.split("<sys> ")[1].replace("</s>", "")

output = model.generate(input_ids, max_length=50, do_sample=True, top_k=10)
tokenizer.decode(output[0].numpy().tolist())


def return_answer_by_chatbot(user_text):
    sent = "<usr>" + user_text + "<sys>"
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
    input_ids = tf.convert_to_tensor([input_ids])
    output = model.generate(input_ids, max_length=50, do_sample=True, top_k=20)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    chatbot_response = sentence.split("<sys> ")[1].replace("</s>", "")
    return chatbot_response


print(return_answer_by_chatbot("안녕! 반가워~"))
print(return_answer_by_chatbot("너는 누구야?"))
print(return_answer_by_chatbot("사랑해"))
print(return_answer_by_chatbot("나랑 영화보자"))
print(return_answer_by_chatbot("너무 심심한데 나랑 놀자"))
print(return_answer_by_chatbot("영화 해리포터 재밌어?"))
print(return_answer_by_chatbot("너 딥 러닝 잘해?"))
print(return_answer_by_chatbot("너 취했어?"))
print(return_answer_by_chatbot("커피 한 잔 할까?"))
