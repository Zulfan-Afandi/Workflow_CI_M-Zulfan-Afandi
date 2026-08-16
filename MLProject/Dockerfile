FROM python:3.12-slim

WORKDIR /app

COPY conda.yaml .
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.1 \
    mlflow==2.19.0

COPY . .

# Contoh serve model MLflow (sesuaikan path run_id/model kalau perlu)
CMD ["python", "modelling.py"]
