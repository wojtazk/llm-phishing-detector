from helpers import print_device_info, get_model_prediction, OutputColors
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# specify model path
MODEL_PATH = 'ealvaradob/bert-finetuned-phishing'

# check for available cuda devices
cuda_available = torch.cuda.is_available()
# set device to use
device = 'cuda' if cuda_available else 'cpu'

# initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# move model to gpu if it is available
model.to(device)

# print info about the device
print_device_info(cuda_available)

# get user input
message = input(f'\n{OutputColors.BLUE}Enter potential phishing message (One line only):\n{OutputColors.RESET}')

# tokenize the input
tokenized_input = tokenizer(message, return_tensors='pt', truncation=True, padding=True).to(device)

# get model output
with torch.no_grad(): # don't calculate gradient for the output
    model_output = model(**tokenized_input)

# get model prediction -> (label, phishing_probability)
predicted_label, phishing_probability = get_model_prediction(model_output)

# print info about the message
print(f'\nPrediction: '
      f'{OutputColors.GREEN + 'normal message' if predicted_label == 0
      else OutputColors.RED + 'phishing message'}{OutputColors.RESET}')
print(f'Phishing probability: {OutputColors.RED if phishing_probability >= 0.5 else OutputColors.GREEN}'
      f'{phishing_probability * 100:.5f}%{OutputColors.RESET}')
