from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch.distributed as dist
import ipdb
# 모델 저장 경로 설정

model_id = "neuralmagic/Llama-3.2-1B-Instruct-FP8"
number_gpus = 8
max_model_len = 8192

# Sampling parameters
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

# 토크나이저 로드 (cache_dir 적용)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 프롬프트 생성
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# vLLM에서 모델을 /dataset에 저장
llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len)

# 모델 생성
outputs = llm.generate(prompts, sampling_params)

# 출력 결과
generated_text = outputs[0].outputs[0].text
print(generated_text)

if dist.is_initialized():
    dist.destroy_process_group()