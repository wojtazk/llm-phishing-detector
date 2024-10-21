# LLM Phishing Detector

## LLM Model
model used: [ealvaradob/bert-finetuned-phishing](https://huggingface.co/ealvaradob/bert-finetuned-phishing)\
model license: `Apache license 2.0`

## LlmPhishingDetector class usage example

```python
from llm_phishing_detector import LlmPhishingDetector

# initialize detector
detector = LlmPhishingDetector()

message = "Hello There!"

# get label(0 - normal message, 1 - phishing) and phishing probability
# label: int, phishing_probability: float
label, phishing_probability = detector.detect_phishing(message)
```

## Running locally
Install project requirements:
```shell
pip install -r ./requirements.txt
```
Run the program:
```shell
python phishing_detector.py
```
\
Type the content to be checked for phishing.\
Type `<<EOF` as the last line to signal the end of input.

Get prediction from the model.

![image](https://github.com/user-attachments/assets/27557da8-6bf6-4c03-81a3-26826db36982)


## Running locally with pipenv
> [!WARNING]
> It doesn't really work on Windows, due to issues with installing PyTorch

Get into pipenv environment:
```shell
pipenv shell
```
Install dependencies
```shell
pipenv install
```
Run the program:
```shell
python phishing_detector.py
```

## Evaluating the model
> [!WARNING]
> For evaluation to work properly you need to put `csv` files into `evaluation_data` directory.\
> `body` & `label` column headers should be present in `csv` files.

```shell
python model_evaluate.py
```

### Data used for evaluation
<!-- - Non-phishing emails - [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
    - https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz -->
- [Phishing Email Curated Datasets](https://figshare.com/articles/dataset/Phishing_Email_Curated_Datasets/24899943)\
\
    `license`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
      
    >A. I. Champa, M. F. Rabbi, and M. F. Zibran, “Curated datasets and feature analysis for phishing email detection with machine learning,” in 3rd IEEE International Conference on Computing and Machine Intelligence (ICMI), 2024, pp. 1–7 (to appear).

### Evaluation Score
On `Phishing Email Curated Database` the model labeled `92.34%` messages correctly.\
\
![evaluation score screenshot](https://github.com/user-attachments/assets/e263503b-5cbf-4c1e-a643-acc69539b850)

## Running in Docker
> [!NOTE]
> Docker image is about 11GB (sry, dependencies)

Build image:
```shell
docker image build -t llm-phishing-detector:v1 .
```
Run the container (not GPU accelerated):
```shell
docker container run -it --rm llm-phishing-detector:v1
```
Run on CUDA device:
```shell
docker container run --gpus all -it --rm llm-phishing-detector:v1
```
