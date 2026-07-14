import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("Telco_Churn_Baseline")
mlflow.sklearn.autolog()

X_train = pd.read_csv("namadataset_preprocessing/X_train.csv")
X_test = pd.read_csv("namadataset_preprocessing/X_test.csv")
y_train = pd.read_csv("namadataset_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("namadataset_preprocessing/y_test.csv").values.ravel()

with mlflow.start_run(run_name="baseline_rf"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Test accuracy:", acc)
