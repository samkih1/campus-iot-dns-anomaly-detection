from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


TEST_FILE = str(PROJECT_ROOT / "data/campus_dns_test_features.csv")
MODEL_FILE = str(PROJECT_ROOT / "models/isolation_forest_model.pkl")


FEATURES = [
    "subdomain_length",
    "query_entropy",
    "inter_query_interval",
    "interval_std_10q",
    "queries_last_5s",
    "nxdomain_ratio_1s",
    "repeated_query_ratio_1s"
]


def main():

    print("Loading independent test dataset...")

    data = pd.read_csv(TEST_FILE)

    print("Test dataset shape:", data.shape)

    X_test = data[FEATURES]

    # Ground truth for anomaly detection:
    # Benign = 0, any attack = 1
    y_true = data["label"].apply(
        lambda x: 0 if x == "Benign" else 1
    )

    model = joblib.load(MODEL_FILE)

    print("\nLoaded Isolation Forest model.")

    print("\nRunning predictions...")

    raw_predictions = model.predict(X_test)

    # Isolation Forest:
    #  1 = inlier/normal
    # -1 = outlier/anomaly
    y_pred = pd.Series(raw_predictions).apply(
        lambda x: 0 if x == 1 else 1
    )

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print("\nOverall Metrics")
    print("----------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report")
    print("---------------------")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Benign", "Anomalous"],
            digits=4,
            zero_division=0
        )
    )

    print("Confusion Matrix")
    print("----------------")
    print("[[TN FP]")
    print(" [FN TP]]")

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\nPrediction counts")
    print("-----------------")
    print(pd.Series(y_pred).value_counts().sort_index())


if __name__ == "__main__":
    main()
