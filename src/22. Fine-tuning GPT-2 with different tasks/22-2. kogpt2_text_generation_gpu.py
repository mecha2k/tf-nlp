# !pip install transformers

"""# 1. KoGPT2로 문장 생성하기"""

import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
model = TFGPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2", from_pt=True)

sent = "근육이 커지기 위해서는"

input_ids = tokenizer.encode(sent)
input_ids = tf.convert_to_tensor([input_ids])
print(input_ids)

output = model.generate(input_ids, max_length=128, repetition_penalty=2.0, use_cache=True)
output_ids = output.numpy().tolist()[0]
print(output_ids)

tokenizer.decode(output_ids)

"""# 2. Numpy로 Top 5 뽑기"""

import numpy as np
import random

output = model(input_ids)
print(output.logits)
print(output.logits.shape)

top5 = tf.math.top_k(output.logits[0, -1], k=5)
tokenizer.convert_ids_to_tokens(top5.indices.numpy())

"""# 3. Numpy Top 5로 문장 생성하기"""

sent = "근육이 커지기 위해서는"
input_ids = tokenizer.encode(sent)


while len(input_ids) < 50:
    output = model(np.array([input_ids]))
    top5 = tf.math.top_k(output.logits[0, -1], k=5)
    token_id = random.choice(top5.indices.numpy())
    input_ids.append(token_id)

print(tokenizer.decode(input_ids))
