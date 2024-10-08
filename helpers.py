import torch
from transformers.modeling_outputs import SequenceClassifierOutput
import os

import csv
import sys


class OutputColors:
    RESET = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    GREY = '\33[90m'


def print_device_info(cuda_available: bool) -> None:
    if cuda_available:
        print(f'{OutputColors.GREY}'
              f'CUDA device available: {cuda_available}')
        print(f'CUDA devices available: {torch.cuda.device_count()} ')
        print(f'CUDA current device: {torch.cuda.current_device()}')
        print(f'Running on: {torch.cuda.get_device_name(torch.cuda.current_device())}'
              f'{OutputColors.RESET}')
    else:
        print(f'Running on CPU')

def get_model_prediction(model_output: SequenceClassifierOutput) -> (int, float):
    # get raw scores from a model
    logits = model_output.logits

    # get distributed probability
    probabilities = torch.softmax(logits, dim=-1)  # tensor([normal_message_probability, phishing_message_probability])

    # get the index of the maximum value element
    predicted_label = torch.argmax(probabilities, dim=-1).item()  # 0 -> normal message, 1 -> phishing

    # get probabilities from tensor
    [_not_phishing_probability, phishing_probability] = probabilities.tolist()[0]

    return predicted_label, phishing_probability


def load_evaluation_data() -> (str, int):
    # increase the csv max field size limit
    csv.field_size_limit(sys.maxsize)

    eval_data_dir = 'evaluation_data'

    # increase the csv max field size limit
    csv.field_size_limit(sys.maxsize)

    message_bodies: [str] = []
    labels: [int] = []

    # get all files in eval_data_dir
    evaluation_files = os.listdir(eval_data_dir)

    print()
    for file in evaluation_files:
        # skip files with extensions other than .csv
        if not file.endswith('.csv'):
            continue

        print(f'{OutputColors.BLUE}loading data from: {file}')

        with open(f'{eval_data_dir}/{file}', newline='') as csv_file:
            try:
                reader = csv.DictReader(csv_file)

                for row in reader:
                    body = row['body']
                    label = row['label']

                    # in case of bad label or body -> skip
                    if (label is None) or (body is None):
                        continue

                    # append body and label
                    message_bodies.append(body)
                    labels.append(int(label))
            except csv.Error as err:
                print(err)

    print(f'{OutputColors.GREEN}loading complete{OutputColors.RESET}')
    return message_bodies, labels