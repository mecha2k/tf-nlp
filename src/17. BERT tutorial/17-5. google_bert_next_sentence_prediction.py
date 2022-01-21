import transformers

"""# 1. 모델 로드"""

from transformers import TFBertForNextSentencePrediction

model = TFBertForNextSentencePrediction.from_pretrained("bert-base-uncased")

"""# 2. 토크나이저 로드"""

from transformers import AutoTokenizer

import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

"""# 3. 테스트

인덱스 0 : 실제 다음 문장  
인덱스 1 : 서로 상관없는 문장
"""

prompt = (
    "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
)
next_sentence = (
    "pizza is eaten with the use of a knife and fork. In casual settings, however, it is cut into wedges"
    " to be eaten while held in the hand."
)

encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

print(encoding["input_ids"])

print(tokenizer.cls_token, ":", tokenizer.cls_token_id)
print(tokenizer.sep_token, ":", tokenizer.sep_token_id)

print(tokenizer.decode(encoding["input_ids"][0]))

print(encoding["token_type_ids"])

logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
print(logits)

softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print(probs)
print("최종 예측 레이블 :", tf.math.argmax(probs, axis=-1).numpy())

prompt = (
    "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
)
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
print(logits)

softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print(probs)
print(tf.math.argmax(probs, axis=-1).numpy())
