import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Catatan: TIDAK ada dagshub.init() di sini secara sengaja.
# Tanpa baris itu, MLflow otomatis menyimpan run ke folder lokal ./mlruns
# sehingga bisa dibuka lewat "mlflow ui" dan di-screenshot dari 127.0.0.1:5000

mlflow.set_experiment("Telco_Churn_Tuning_Local")

X_train = pd.read_csv("namadataset_preprocessing/X_train.csv")
X_test = pd.read_csv("namadataset_preprocessing/X_test.csv")
y_train = pd.read_csv("namadataset_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("namadataset_preprocessing/y_test.csv").values.ravel()

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

with mlflow.start_run(run_name="tuning_rf_advance_local"):
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=3, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Manual logging: params
    mlflow.log_params(grid.best_params_)

    # Manual logging: metrics setara autolog
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    # Artefak tambahan #1: confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix - Telco Churn")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")
    plt.close()

    # Artefak tambahan #2: metric_info.json
    metric_info = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_score": f1, "roc_auc": auc, "best_params": grid.best_params_
    }
    with open("metric_info.json", "w") as f:
        json.dump(metric_info, f, indent=2)
    mlflow.log_artifact("metric_info.json")

    # Artefak tambahan #3: feature importance
    importances = best_model.feature_importances_
    plt.figure(figsize=(6, 5))
    sns.barplot(x=importances, y=X_train.columns)
    plt.title("Feature Importance - Telco Churn")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    print(f"Best params: {grid.best_params_}")
    print(f"Test accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")
    print("Training selesai. Jalankan 'mlflow ui --port 5000' lalu buka http://127.0.0.1:5000 untuk melihat hasilnya.")