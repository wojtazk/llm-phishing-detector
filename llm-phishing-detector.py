import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# specify model path
model_path = 'openlm-research/open_llama_3b'

# set device to execute query on
cuda_available = torch.cuda.is_available()
device = 'cuda' if cuda_available else 'cpu'

# print info about CUDA device
print(f'CUDA device available: {cuda_available}')
print(f'CUDA devices available: {torch.cuda.device_count()} ')
print(f'CUDA current device: {torch.cuda.current_device()}')
print(f'CUDA current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')

# initialize model
tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=False)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16
)

# move model to device
model.to(device)

# get message from the user
message = 'Click this link to win an Iphone'

# create prompt
prompt = f'Q: Is this message a phishing attempt: {message}\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids  # 'pt' - pytorch tensors

# generate output
generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=16, do_sample=True
)

# print model output
print(tokenizer.decode(generation_output[0], skip_special_tokens=True))
