import torch
from transformers import BertTokenizer, BertForSequenceClassification
from helpers import OutputColors


class LlmPhishingDetector:
    # specify model path
    model_path = 'ealvaradob/bert-finetuned-phishing'

    # check for available cuda devices
    cuda_available = torch.cuda.is_available()
    # set device to use
    device = 'cuda' if cuda_available else 'cpu'

    @classmethod
    def print_device_info(cls) -> None:
        if cls.cuda_available:
            print(f'{OutputColors.GREY}'
                  f'CUDA device available: {cls.cuda_available}')
            print(f'CUDA devices available: {torch.cuda.device_count()} ')
            print(f'CUDA current device: {torch.cuda.current_device()}')
            print(f'Running on: {torch.cuda.get_device_name(torch.cuda.current_device())}'
                  f'{OutputColors.RESET}')
        else:
            print(f'Running on CPU')

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
        with torch.no_grad():  # don't calculate gradient for the output
            model_output = self.model(**tokenized_input)

        # get raw scores from a model
        logits = model_output.logits

        # get distributed probability
        probabilities = torch.softmax(logits,
                                      dim=-1)  # tensor([[normal_message_probability, phishing_message_probability]])

        # get the index of the maximum value element
        label = torch.argmax(probabilities, dim=-1).item()  # 0 -> normal message, 1 -> phishing

        # get probabilities from tensor
        [_not_phishing_probability, phishing_probability] = probabilities.tolist()[0]

        return label, phishing_probability
