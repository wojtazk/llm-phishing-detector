from helpers import print_device_info, get_model_prediction, OutputColors
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class LlmPhishingDetector:
    # specify model path
    model_path = 'ealvaradob/bert-finetuned-phishing'

    # check for available cuda devices
    cuda_available = torch.cuda.is_available()
    # set device to use
    device = 'cuda' if cuda_available else 'cpu'

    def __init__(self):
        # initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)

        # move model to gpu if it is available
        self.model.to(self.device)

    def detect_phishing(self, message):
        # tokenize the input
        tokenized_input = self.tokenizer(message, return_tensors='pt', truncation=True, padding=True).to(self.device)

        # get model output
        with torch.no_grad(): # don't calculate gradient for the output
            model_output = self.model(**tokenized_input)

        # get model prediction -> (label, phishing_probability)
        label, probability = get_model_prediction(model_output)

        return label, probability


# print device info
print_device_info(LlmPhishingDetector.cuda_available)
# get user input
user_input = input(f'\n{OutputColors.BLUE}Enter potential phishing message (One line only):\n{OutputColors.RESET}')
# get model prediction
predicted_label, predicted_phishing_probability = LlmPhishingDetector().detect_phishing(user_input)

# print info about the message
print(f'\nPrediction: '
      f'{OutputColors.GREEN + 'normal message' if predicted_label == 0
      else OutputColors.RED + 'phishing message'}{OutputColors.RESET}')
print(f'Phishing probability: {OutputColors.RED if predicted_phishing_probability >= 0.5 else OutputColors.GREEN}'
      f'{predicted_phishing_probability * 100:.5f}%{OutputColors.RESET}')
