FROM python:3.12-bookworm

LABEL description="Detect phishing leveraging LLM's"
LABEL authors="wojtazk"

WORKDIR /llm-phishing-detector

COPY . .

# install project dependencies
RUN pip install -r ./requirements.txt

# download the model from HuggingFace
RUN python -c "from transformers import BertTokenizer, BertForSequenceClassification;\
model_path = 'ealvaradob/bert-finetuned-phishing';\
BertTokenizer.from_pretrained(model_path);\
BertForSequenceClassification.from_pretrained(model_path)"

CMD ["python", "phishing_detector.py"]