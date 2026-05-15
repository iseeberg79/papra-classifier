FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir flask fasttext-wheel "numpy<2" requests anthropic

COPY papra_model.bin .
COPY classify.py .

EXPOSE 5000

CMD ["python3", "classify.py"]

