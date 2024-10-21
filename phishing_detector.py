from helpers import OutputColors
from llm_phishing_detector import LlmPhishingDetector

detector = LlmPhishingDetector()

# print device info
detector.print_device_info()

# get user input
inputs = [input(f'{OutputColors.BLUE}\nEnter potential phishing message (type <<EOF to end input):'
                f'\n{OutputColors.RESET}')]

while True:
    line = input()
    if line == '<<EOF':
        break
    inputs.append(line)

# join inputs
user_input = '\n'.join(inputs)

# get model prediction
predicted_label, predicted_phishing_probability = detector.detect_phishing(user_input)

# print info about the message
print(f'\nPrediction: '
      f'{OutputColors.GREEN + 'normal' if predicted_label == 0 else OutputColors.RED + 'phishing'}'
      f'{OutputColors.RESET}')
print(f'Phishing probability: {OutputColors.RED if predicted_phishing_probability >= 0.5 else OutputColors.GREEN}'
      f'{predicted_phishing_probability * 100:.5f}%{OutputColors.RESET}')
