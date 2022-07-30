from transformers import PreTrainedTokenizerFast

special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", **special_tokens)

bos_token = "<bos>"
question = ["안녕", "하", "세", "요"]
question = " ".join(question)
print(question)
answer = "안녕하세요. 한국어 GPT-2 입니다.😤:)"
inputs = bos_token + str(question) + str(answer)
print(inputs)
inputs = tokenizer.encode(inputs, return_tensors="pt")
print(inputs)
print(tokenizer.decode(inputs[0]))
