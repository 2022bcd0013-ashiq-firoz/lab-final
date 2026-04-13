import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import argparse
import os

# -------------------------
# ARGUMENTS
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/data.csv")
parser.add_argument("--run_type", type=str, default="base")  # base / tuned
args = parser.parse_args()

# -------------------------
# MLFLOW SETUP (FIXED)
# -------------------------
tracking_uri = "http://localhost:5000"

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    # fallback for local runs (safe for Linux + Windows)
    mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment("test")

# -------------------------
# LOAD DATA
# -------------------------
if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Dataset not found at {args.data_path}")

df = pd.read_csv(args.data_path)

# Basic preprocessing
df = df.dropna()

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    if col != "Churn":
        df[col] = LabelEncoder().fit_transform(df[col])

# Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODEL CONFIG
# -------------------------
if args.run_type == "base":
    model = XGBClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=8,
        learning_rate=0.105,
        eval_metric="logloss",
        use_label_encoder=False
    )
else:
    model = XGBClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        use_label_encoder=False
    )

# -------------------------
# TRAIN + LOG
# -------------------------
with mlflow.start_run(run_name=args.run_type):

    # Params
    mlflow.log_param("dataset_version", "v1")
    mlflow.log_param("model", "xgboost")
    mlflow.log_param("run_type", args.run_type)

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log dataset (useful for reproducibility)
    mlflow.log_artifact(args.data_path)

    # ✅ FIXED model logging (no deprecated arg)
    mlflow.sklearn.log_model(model, name="model")

    print("=" * 40)
    print(f"Run Type : {args.run_type}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=" * 40)