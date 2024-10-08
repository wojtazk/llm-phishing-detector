import torch
from transformers import BertTokenizer, BertForSequenceClassification
from helpers import get_model_prediction


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

    def detect_phishing(self, message) -> (int, float):
        # tokenize the input
        tokenized_input = self.tokenizer(message, return_tensors='pt', truncation=True, padding=True).to(self.device)

        # get model output
        with torch.no_grad(): # don't calculate gradient for the output
            model_output = self.model(**tokenized_input)

        # get model prediction -> (label, phishing_probability)
        label, probability = get_model_prediction(model_output)

        return label, probability