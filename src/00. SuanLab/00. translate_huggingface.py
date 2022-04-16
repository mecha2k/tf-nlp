from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")

result = translator(
    "방황하는 삼성전자 주가는 언제쯤 우상향 궤적을 그릴지 다들 궁금하실 겁니다. 단기 전망은 어두워보입니다. 증권사들은 삼성전자 목표 주가를 낮춰 잡기 시작했습니다."
)
print(result)
