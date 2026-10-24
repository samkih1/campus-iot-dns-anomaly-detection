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

THRESHOLD = -0.083149


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

    # Ground truth:
    # Benign = 0
    # DNS_Tunnel / NXDOMAIN_Flood = 1
    y_true = (data["label"] != "Benign").astype(int)

    model = joblib.load(MODEL_FILE)

    print("\nLoaded Isolation Forest model.")

    print(f"\nUsing calibrated threshold: {THRESHOLD:.6f}")

    # Higher decision score = more normal
    # Lower score = more anomalous
    scores = model.decision_function(X_test)

    # Apply training-derived 5th percentile threshold
    y_pred = (scores < THRESHOLD).astype(int)

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

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0.0

    print("\nOverall Metrics")
    print("----------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")

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
    print(cm)

    # Save predictions for later analysis
    results = data[
        [
            "scenario",
            "label"
        ]
    ].copy()

    results["anomaly_score"] = scores
    results["predicted_anomaly"] = y_pred

    OUTPUT_FILE = str(
        PROJECT_ROOT / "results/isolation_forest_calibrated_predictions.csv"
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved predictions to:\n{OUTPUT_FILE}")

    print("\nScenario-level anomaly rates")
    print("----------------------------")

    scenario_summary = (
        results
        .groupby("scenario")
        ["predicted_anomaly"]
        .agg(["count", "sum", "mean"])
    )

    scenario_summary.columns = [
        "records",
        "flagged_anomalous",
        "anomaly_rate"
    ]

    print(scenario_summary)


if __name__ == "__main__":
    main()
