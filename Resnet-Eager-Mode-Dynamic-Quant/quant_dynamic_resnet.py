import torch
from model.resnet import resnet18

from evaluate import evaluate

model = resnet18(pretrained=True)
print(model)

# Step 1: architecture changes
# QuantStubs (we will do FloatFunctionals later)
# Done

# Step 2: fuse modules (recommended but not necessary)
modules_to_list = model.modules_to_fuse()

# It will keep Batchnorm
model.eval()
# fused_model = torch.ao.quantization.fuse_modules_qat(model, modules_to_list)

# This will fuse BatchNorm weights into the preceding Conv
fused_model = torch.ao.quantization.fuse_modules(model, modules_to_list)

# Step 3: Assign qconfigs
backend = 'fbgemm'
qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

for name, module in fused_model.named_modules():
    module.qconfig = qconfig

# Step 4: Prepare for fake-quant
fused_model.train()
fake_quant_model = torch.ao.quantization.prepare_qat(fused_model)

# Step 4b: Try dynamic quantization
fake_quant_model_dynamic = torch.quantization.quantize_dynamic(model)

# Evaluate
print('\noriginal')
evaluate(model, 'cpu')
print('\nfused')
evaluate(fused_model, 'cpu')

print('\ndynamic')
evaluate(fake_quant_model_dynamic, 'cpu')


# Step 5: convert (true int8 model)
fake_quant_model.to('cpu')
converted_model = torch.quantization.convert(fake_quant_model)

print('\nfake quant')
evaluate(fake_quant_model, 'cpu')


print('\nconverted')
evaluate(converted_model, 'cpu')


# ## Torch compile
# compiled_model = torch.compile(model)
# print(compiled_model)