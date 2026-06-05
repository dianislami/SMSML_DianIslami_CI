"""
modelling.py (MLProject version)
=================================
Versi modelling untuk dijalankan via MLflow Project + GitHub Actions CI.
Support parameter CLI dan tracking ke DagsHub.
"""

# import library
import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators',      type=int,   default=200)
    parser.add_argument('--max_depth',         type=int,   default=20)
    parser.add_argument('--min_samples_split', type=int,   default=2)
    parser.add_argument('--max_features',      type=str,   default='sqrt')
    parser.add_argument('--data_dir',          type=str,   default='Dry_Bean_Dataset_preprocessing')
    return parser.parse_args()


def save_confusion_matrix(y_test, y_pred, path="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_feature_importance(model, feature_names, path="feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices], color='steelblue')
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def main():
    args = parse_args()

    # Setup MLflow
    dagshub_owner = os.environ.get("DAGSHUB_REPO_OWNER", "")
    dagshub_repo  = os.environ.get("DAGSHUB_REPO_NAME",  "")

    if dagshub_owner and dagshub_repo:
        mlflow.set_tracking_uri(
            f"https://dagshub.com/{dagshub_owner}/{dagshub_repo}.mlflow"
        )
        print(f"[MLflow] Tracking ke DagsHub: {dagshub_owner}/{dagshub_repo}")
    else:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        print("[MLflow] Tracking ke localhost")

    mlflow.set_experiment("Dry-Bean-CI")

    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, "dry_bean_train.csv"))
    test_df  = pd.read_csv(os.path.join(args.data_dir, "dry_bean_test.csv"))
    X_train  = train_df.drop(columns=["Class"])
    y_train  = train_df["Class"]
    X_test   = test_df.drop(columns=["Class"])
    y_test   = test_df["Class"]
    print(f"[Data] Train: {X_train.shape} | Test: {X_test.shape}")

    os.environ.pop("MLFLOW_RUN_ID", None)

    with mlflow.start_run(run_name="CI-RandomForest"):
        # Parameters
        params = {
            "n_estimators":      args.n_estimators,
            "max_depth":         args.max_depth,
            "min_samples_split": args.min_samples_split,
            "max_features":      args.max_features,
        }
        mlflow.log_params(params)

        # Train
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred    = model.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall    = recall_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("accuracy",              acc)
        mlflow.log_metric("f1_weighted",           f1)
        mlflow.log_metric("precision_weighted",    precision)
        mlflow.log_metric("recall_weighted",       recall)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Artefak tambahan
        cm_path = save_confusion_matrix(y_test, y_pred)
        fi_path = save_feature_importance(model, X_train.columns.tolist())
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(fi_path, artifact_path="plots")

        # Simpan model lokal untuk upload CI
        os.makedirs("artifacts", exist_ok=True)
        import joblib
        model_path = "artifacts/model.pkl"
        joblib.dump(model, model_path)

        run_id = mlflow.active_run().info.run_id
        print(f"[MLflow] Run ID: {run_id}")

    print(f"\n✅ Accuracy: {acc:.4f} | F1: {f1:.4f}")


if __name__ == '__main__':
    main()
