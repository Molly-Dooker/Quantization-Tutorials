import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer



device = "cuda"
dtype = torch.bfloat16
model_id = "meta-llama/Llama-3.2-1B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)


def moderate(messages):
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

print(moderate(messages))