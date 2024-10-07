# LLM Phishing Detector
LLM model: [ealvaradob/bert-finetuned-phishing](https://huggingface.co/ealvaradob/bert-finetuned-phishing)

## Running locally
Install project requirements:
```shell
pip install -r ./requirements.txt
```
Run the program:
```shell
python llm-phishing-detector.py
```

## Running in Docker
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