# -*- coding: utf-8 -*-
"""한국어 BERT를 이용한 다음 문장 예측.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dt35ns6r_iK9HDVnruK8uBicoOuLSGND
"""

pip install transformers

import transformers
transformers.__version__

"""# 1. 모델 로드"""

from transformers import TFBertForNextSentencePrediction

model = TFBertForNextSentencePrediction.from_pretrained('klue/bert-base', from_pt=True)

"""# 2. 토크나이저 로드"""

from transformers import AutoTokenizer

import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

"""# 3. 테스트

인덱스 0 : 실제 다음 문장  
인덱스 1 : 서로 상관없는 문장
"""

# 이어지는 두 개의 문장
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "여행을 가보니 한국의 2002년 월드컵 축구대회의 준비는 완벽했습니다."
encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]

softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print('최종 예측 레이블 :', tf.math.argmax(probs, axis=-1).numpy())

# 상관없는 두 개의 문장
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "극장가서 로맨스 영화를 보고싶어요"
encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]

softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print('최종 예측 레이블 :', tf.math.argmax(probs, axis=-1).numpy())