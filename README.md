# Workflow-CI — Heart Disease Model

Repo ini berisi MLflow Project + GitHub Actions CI yang otomatis:
1. Menjalankan training model (`mlflow run`) setiap ada push ke `main`.
2. Build image Docker dan push ke Docker Hub (kalau secrets sudah diisi).

## Docker Hub
Image: https://hub.docker.com/r/<DOCKERHUB_USERNAME>/heart-disease-model
(ganti `<DOCKERHUB_USERNAME>` dengan username Docker Hub kamu setelah CI berhasil push)

## Setup yang perlu dilakukan sekali di GitHub (Settings > Secrets and variables > Actions):
- `DOCKERHUB_USERNAME` — username Docker Hub kamu
- `DOCKERHUB_TOKEN` — access token dari Docker Hub (Account Settings > Security > New Access Token)
- (opsional, kalau pakai DagsHub) `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`
