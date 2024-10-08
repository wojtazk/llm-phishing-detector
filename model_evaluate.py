from helpers import load_evaluation_data
from LlmPhishingDetector import LlmPhishingDetector

# get data from files
message_bodies, labels = load_evaluation_data()

model_prediction_labels = []

# initialize detector
detector = LlmPhishingDetector()
# get predictions
loop_counter = 0
for message in message_bodies:
    label, _ = detector.detect_phishing(message)
    model_prediction_labels.append(label)

    loop_counter += 1
    if loop_counter % 200 == 0:
        # print model progress
        print(f'progress: {len(model_prediction_labels)} / {len(labels)}')

# count correct labels
num_correct_labels = len(labels)
for i in range(len(labels)):
    if labels[i] != model_prediction_labels[i]:
        num_correct_labels -= 1

total_messages = len(message_bodies)
print(f'Total number of messages: {total_messages}')
print(f'Number of phishing messages: {labels.count(1)}')
print(f'Model prediction accuracy: {(num_correct_labels / total_messages) * 100:.5f}%')


