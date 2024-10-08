from helpers import print_device_info, OutputColors
from LlmPhishingDetector import LlmPhishingDetector

detector = LlmPhishingDetector()

# print device info
print_device_info(detector.cuda_available)
# get user input
user_input = input(f'\n{OutputColors.BLUE}Enter potential phishing message (One line only):\n{OutputColors.RESET}')
# get model prediction
predicted_label, predicted_phishing_probability = detector.detect_phishing(user_input)

# print info about the message
print(f'\nPrediction: '
      f'{OutputColors.GREEN + 'normal message' if predicted_label == 0
      else OutputColors.RED + 'phishing message'}{OutputColors.RESET}')
print(f'Phishing probability: {OutputColors.RED if predicted_phishing_probability >= 0.5 else OutputColors.GREEN}'
      f'{predicted_phishing_probability * 100:.5f}%{OutputColors.RESET}')
