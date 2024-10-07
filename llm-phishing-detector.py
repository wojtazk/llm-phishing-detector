import torch
from transformers import BertTokenizer, BertForSequenceClassification

# specify model path
model_path = 'ealvaradob/bert-finetuned-phishing'

# set device to execute query on
cuda_available = torch.cuda.is_available()
device = 'cuda' if cuda_available else 'cpu'

# initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# move model to gpu if it is available
model.to(device)

# print info about CUDA device
print(f'CUDA device available: {cuda_available}')
print(f'CUDA devices available: {torch.cuda.device_count()} ')
print(f'CUDA current device: {torch.cuda.current_device()}')
print(f'CUDA current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')

# get user input
message = input('\nEnter potential phishing message:\n')

# tokenize the input
tokenized_input = tokenizer(message, return_tensors='pt', truncation=True, padding=True).to(device)

# dont caculate gradient for the output
with torch.no_grad():
    model_output = model(**tokenized_input)

# get raw scores from a model
logits = model_output.logits

# get distributed probability
probabilities = torch.softmax(logits, dim=-1)  # tensor([normal_message_probability, phishing_message_probability])

# get the index of the maximum value element
predicted_class = torch.argmax(probabilities, dim=-1).item()  # 0 -> normal message, 1 -> phishing

# get probabilities from tensor
[_not_phishing_probability, phishing_probability] = probabilities.tolist()[0]

# print info about the message
print(f'\nPredicted class: {'phishing' if predicted_class == 1 else 'normal message'}')
print(f'Phishing probability: {phishing_probability * 100:.5f}%')
