"""
modelling.py (Workflow-CI version)
===================================
Training model Heart Disease UCI dengan MLflow tracking, dipakai sebagai
entry point MLflow Project (dipanggil lewat `mlflow run .` di CI).

Cara pakai manual:
    python modelling.py
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "namadataset_preprocessing", "heart_train_preprocessed.csv")
TEST_PATH = os.path.join(BASE_DIR, "namadataset_preprocessing", "heart_test_preprocessed.csv")
TARGET_COL = "target"


def load_train_test():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test


def main():
    # Kalau MLFLOW_TRACKING_URI diset (misal ke DagsHub) dia akan dipakai,
    # kalau tidak, fallback ke folder lokal ./mlruns
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("Heart_Disease_Workflow_CI")

    X_train, X_test, y_train, y_test = load_train_test()

    mlflow.sklearn.autolog()

    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    }

    with mlflow.start_run(run_name="ci_random_forest"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)

        print("=== Hasil Model (Workflow-CI) ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")


if __name__ == "__main__":
    main()
