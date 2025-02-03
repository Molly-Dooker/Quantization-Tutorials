#Large Languase Model, such as llama2-7b.
from calflops import calculate_flops
from transformers import AutoModelForCausalLM, AutoTokenizer
import ipdb
model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# model_id = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"


batch_size, max_seq_length = 1, 1

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
flops, macs, params = calculate_flops(model=model,
                                      input_shape=(batch_size, max_seq_length),
                                      transformer_tokenizer=tokenizer)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#Llama2(7B) FLOPs:1.7 TFLOPS   MACs:850.00 GMACs   Params:6.74 B 