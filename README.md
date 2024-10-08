# LLM Phishing Detector

## LLM Model
model used: [ealvaradob/bert-finetuned-phishing](https://huggingface.co/ealvaradob/bert-finetuned-phishing)\
model license: `Apache license 2.0`

## Running locally
Install project requirements:
```shell
pip install -r ./requirements.txt
```
Run the program:
```shell
python llm_phishing_detector.py
```

## Running locally with pipenv
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
python llm_phishing_detector.py
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
