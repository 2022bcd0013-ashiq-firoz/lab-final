import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import argparse
import os
import json
import joblib


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/data_v2.csv")
parser.add_argument("--run_type", type=str, default="base")  
args = parser.parse_args()

if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Dataset not found at {args.data_path}")

df = pd.read_csv(args.data_path)

df = df.dropna()

for col in df.select_dtypes(include="object").columns:
    if col != "Churn":
        df[col] = LabelEncoder().fit_transform(df[col])

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)


os.makedirs("artifacts", exist_ok=True)

model_path = f"artifacts/model_{args.run_type}.pkl"
joblib.dump(model, model_path)

metrics = {
    "Name": "Ashiq Firoz",
    "Roll": "2022BCD0013",
    "accuracy": acc,
    "f1_score": f1
}

metrics_path = f"artifacts/metrics_{args.run_type}.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)


print(f"Run Type : {args.run_type}")
print(f"Accuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Model saved at : {model_path}")
print(f"Metrics saved at : {metrics_path}")
