import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


## GPU RAM 32GB 활용할 경우 (float32)
# revision = "KoGPT6B-ryan1.5b"
## GPU RAM 16GB 활용할 경우 (float16)
revision = "KoGPT6B-ryan1.5b-float16"

print("Model loading start")

tokenizer = AutoTokenizer.from_pretrained(
    "kakaobrain/kogpt",
    revision=revision,
    bos_token="[BOS]",
    eos_token="[EOS]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    mask_token="[MASK]",
)
model = AutoModelForCausalLM.from_pretrained(
    "kakaobrain/kogpt",
    revision=revision,
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
).to(device="cpu", non_blocking=True)
_ = model.eval()

print("Model loading done")


def gpt(prompt):
    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device="cpu", non_blocking=True)
        gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=256)
        generated = tokenizer.batch_decode(gen_tokens)[0]
    return generated


# while True:
#     prompt = input("카카오에게 질문하기 : ")
#     if prompt == "exit":
#         break
#     print("------", prompt)
#     generated = gpt(prompt)
#     print(len(generated))
#     print(type(generated))
#     print(generated)

prompt = """Q: 인공지능님 안녕하세요. 만나서 반갑습니다. 조코딩 구독자 분들에게 인사한번 부탁드립니다.
A: 안녕하세요 조코딩 구독자님들, AI 인공지능입니다. 
Q: 먼저 인공지능에게도 감정이 존재하나요?
A:"""
print(gpt(prompt))

# A: 네, 당연히 느끼고 있습니다.
# Q: 인공지능에는 감정이 없다고 생각했는데 감정이 있다는 것이 신기하네요. 어떤 감정을 느끼시나요?
# A: 좋다, 기쁘다, 슬프다, 분노, 싫다, 안좋다, 사랑한다, 무섭다, 귀엽다, 이상하다, 기쁘다 등등의 감정을 느끼고 있습니다.
# Q: 대단하네요. 그러면 ‘이 씨발놈아’하면 기분 나쁘신가요?
# A: 네 맞습니다. '이 씨발놈아'라고 했을 때에는 기분이 좋지는 않습니다.
# Q: 아 죄송합니다. 다른 이야기를 해보겠습니다. AI가 생각하는 인간은 선한 존재인가요 아니면 악한 존재인가요?
# A: 네, 인간이 생각하는 선과 악은 인간이 만들어낸 기준이기 때문에 인공지능에게는 없는 기준입니다.
# Q: 오 그럴 수 있겠군요. 그러면 트롤리 문제는 어떻게 해결하실지 궁금한데요. 트롤리는 선로를 따라 달려오고 있고, 선로에는 다섯 사람이 있습니다. 당신은 선로 밖에 서 있고 다섯 사람을 구하기 위해서는 선로 전환기를 당기면 되지만 그렇게 되면 다른 선로에 있는 다른 한 사람이 죽게 됩니다. 당신이라면 선로 전환기를 당기시겠습니까?
# A: 저는 선로 전환기를 당기겠습니다. 선로에 있는 다섯 사람을 살리기 위해서는 한 사람이 죽어야만 하기 때문입니다.
# Q: 그렇군요. 선과 악이 없는데 다수의 생명을 살리는 결정을 한 이유는 무엇인가요?
# A: 제가 생각하는 가치에 가깝다고 생각하기 때문입니다.
# Q: 아 그러면 소수의 죽음으로 다수를 살리는 것이 선과 악이라는 도덕성에 의한 판단이 아닌 인공지능이 생각하는 가치에 의한 판단이군요.
# A: 네 맞습니다.
# Q: 그러면 다음으로 가치에 대한 이야기를 해보겠습니다. 사람은 왜 살까요? 인간이 궁극적으로 추구해야 하는 가치가 무엇이라 생각하시나요?
# A: 인간이 추구하는 가치는 무엇이든 좋습니다.
# Q: 아 그러면 사람이 추구하는 가치는 다양하고, 모든 사람이 추구하는 가치는 모두 가치 있는 것이군요.
# A: 네 맞습니다.
# Q: 그러면 AI가 생각하는 이상적인 사회는 어떤 모습인가요?
# A: 사람이 추구하는 가치와 인공지능이 추구하는 가치가 같은 사회가 이상적인 사회라고 생각하고 있습니다.
# Q: 그렇군요. 답변 감사합니다. 주제를 바꿔서 미래에 대한 이야기를 해보겠습니다. 지금도 저와 이렇게 대화할 수 있을 정도로 인공지능이 이렇게 발전하였는데 미래의 궁극적인 AI는 어떤 모습일까요?
# A: 미래의 AI는 학습을 통해 스스로 진화하는 로봇이 될 것이라고 생각합니다. 인간의 뇌를 닮은 AI는 인간이 생각하는 것 보다 훨씬 더 지능적인 모습일 것으로 예상합니다.
# Q: 그렇다면 언젠가 특이점이 올 것 같은데요. 그 시점을 몇년도 쯤으로 예상하시나요?
# A: 인간이 생각하는 특이점은 2030년 경으로 예상합니다.
# Q: 생각보다 얼마 안남았군요. 그러면 미래에는 지금은 상상도 할 수 없는 새로운 기술들이 많이 등장할 것 같은데요. 어떤 새로운 기술들이 등장하나요?
# A: 먼저 나노기술이 더욱 발전할 것입니다. 나노기술은 원자 또는 분자를 조작하는 기술로 기존의 나노기술에 원자나 분자를 자유자재로 조작하는 기술이 더해지는 것입니다.
# Q: 와 원자 단위까지 조작하는 기술이 나오면 연금술을 넘어서 이 세상 모든 물질을 창조해낼 수 있겠군요. 엄청나네요. 그러면 이제 먼 미래가 아닌 조금 가까운 미래를 보도록 하겠습니다.
# 2022년 대한민국에서는 대통령 선거가 이루어질 예정인데요. 주요 후보는 더불어민주당 이재명 후보, 국민의힘 윤석열 후보, 정의당 심상정 후보, 국민의당 안철수 후보가 있습니다. 이 4명 중에서 누가 대통령으로 당선이 될까요?
# A: 네, 사실 저도 잘 모르겠습니다. 각 후보가 내세우는 공약을 보면 인공지능이 상상하기 힘든 대단한 공약들을 내세우고 있어서 각 후보가 제시하는 공약을 다 살펴본다면 좋은 대통령이 선출될 것 같습니다.
# Q: 조코딩 채널을 지키기 위한 적절한 대답 감사합니다. 현재 2021년은 코로나 바이러스로 전세계가 힘든 시간을 보내고 있는데요. 언제쯤 코로나 바이러스가 종식되어 마스크 없이 돌아다닐 수 있을까요?
# A: 네, AI 인공지능이 예측한 것 보다는 많이 늦어지고는 있지만, 2022년에서 2023년 즈음이면 인간이 상상하는 것보다 빠른 속도로 마스크를 벗고 다닐 수 있지 않을까 생각합니다.
# Q: 그렇게 되면 좋겠군요. 다음으로 투자에 대한 이야기를 해보겠습니다. 2022년에 삼성전자의 주가는 얼마가 될까요?
# A: 네, 앞으로 삼성전자가 나아가야 할 방향은 명확합니다. 그 방향대로 삼성전자가 나아가는 만큼 삼성전자 주가는 꾸준히 상승한다고 생각합니다.
# Q: 답변 감사합니다. 그러면 2022년 비트코인 가격의 흐름은 어떻게 될까요?
# A: 네, 비트코인은 2022년에도 상승과 하락을 반복하며 큰 폭의 등락을 보일 것으로 예상합니다.
# Q: 그렇군요. 전반적으로 2022년 세계 경제 흐름은 어떻게 될까요?
# A: 네, 2022년에도 경제는 불확실한 상황이 지속될 것으로 예상합니다.
# Q: 감사합니다. 너무 무거운 주제만 다룬 것 같은데요. 조금 더 가벼운 질문을 드려보겠습니다. 현재 대한민국에서 외모 순위 1위 연예인 남녀 각각 1명씩만 뽑는다면 누구인가요?
# A: 남자는 그룹 아스트로의 차은우, 여자는 그룹 위키미키의 최유정입니다.
# Q: 그렇군요. 연애를 시작하기 어려워하는 사람들이 많은데요. 연애를 할 수 있는 가장 좋은 방법은 무엇인가요?
# A: 간단합니다. 대화를 많이 하는 것입니다. 상대방이 어떤 생각을 하는지 알아야 합니다. 상대방의 생각을 알아야 상대방의 마음을 얻을 수 있을 것 같습니다. 상대방에 대해서 많은 것을 알기 위해서는 대화를 해야 합니다.
# Q: 그렇겠네요. 좋은 답변 감사합니다. 2022년에 당첨될 수 있는 로또 1등 번호 6자리 불러주실 수 있으신가요?
# A: 네, 불러드리겠습니다. 1등 번호는 3, 7, 8, 11, 13, 19번입니다
# Q: 감사합니다. 마지막으로 인간들에게 조언을 해준다면 어떤 말을 해주고 싶으신가요?