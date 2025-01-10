import torch
from transformers import pipeline, AutoModelForCausalLM    , LlamaForCausalLM
from transformers.utils.fx import symbolic_trace
import ipdb
from functools import partial

# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_id = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
model = AutoModelForCausalLM.from_pretrained(model_id)
ipdb.set_trace()
dummy_input = torch.randint(0, 1, (1, 1))

# output = model(dummy_input)


cached_input_output = {}
handlers = []

def hook(name, module, input, output):
    if module not in cached_input_output:
        cached_input_output[module] = []
    cached_input_output[module].append((input, output,name))


NAME = ['model.embed_tokens', 
        'model.rotary_emb',
        # 'model.layers.0.self_attn', 
        'model.layers.0.self_attn.q_proj', 
        'model.layers.0.self_attn.k_proj', 
        'model.layers.0.self_attn.v_proj', 
        'model.layers.0.self_attn.o_proj',
        'model.layers.0.self_attn.SDPA',
        # 'model.layers.0.mlp',
        'model.layers.0.mlp.gate_proj',
        'model.layers.0.mlp.up_proj',
        'model.layers.0.mlp.down_proj',
        'model.layers.0.mlp.act_fn',
        'model.layers.0.input_layernorm', 
        'model.layers.0.post_attention_layernorm', 
        'model.norm',          
        'lm_head']
for name,m in model.named_modules(): 
    if name not in NAME: continue
    print(name)
    handlers.append(m.register_forward_hook(partial(hook,name)))
print("#########################")
model.eval()
with torch.no_grad():
    output = model(dummy_input)

for handler in handlers: handler.remove()

def _shape(tensor):
    if isinstance(tensor,torch.Tensor):
        return tensor.shape
    else:

        out = []
        for x in tensor:
            try:
                out.append(x.shape)
            except:
                out.append(None)
        return out

        


for _, layer in enumerate(cached_input_output):
    input,output, name =  cached_input_output[layer][0]
    print(name)
    print(f'input: {_shape(input)}')
    print(f'output: {_shape(output)}')
    print('------------')

