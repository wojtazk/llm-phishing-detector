import os

import csv
import sys


class OutputColors:
    RESET = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    GREY = '\33[90m'


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