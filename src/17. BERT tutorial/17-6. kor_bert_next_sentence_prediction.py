import transformers
from transformers import TFBertForNextSentencePrediction
from transformers import AutoTokenizer
import tensorflow as tf

model = TFBertForNextSentencePrediction.from_pretrained("klue/bert-base", from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 3. 테스트
# 인덱스 0 : 실제 다음 문장
# 인덱스 1 : 서로 상관없는 문장

# 이어지는 두 개의 문장
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "여행을 가보니 한국의 2002년 월드컵 축구대회의 준비는 완벽했습니다."
encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]

softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print("최종 예측 레이블 :", tf.math.argmax(probs, axis=-1).numpy())

# 상관없는 두 개의 문장
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "극장가서 로맨스 영화를 보고싶어요"
encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]

softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print("최종 예측 레이블 :", tf.math.argmax(probs, axis=-1).numpy())
