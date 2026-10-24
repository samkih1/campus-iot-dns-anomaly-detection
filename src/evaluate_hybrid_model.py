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

RF_MODEL_FILE = str(PROJECT_ROOT / "models/random_forest_model.pkl")
RF_ENCODER_FILE = str(PROJECT_ROOT / "models/random_forest_label_encoder.pkl")
IF_MODEL_FILE = str(PROJECT_ROOT / "models/isolation_forest_model.pkl")

IF_THRESHOLD = -0.083149


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

    X_test = data[FEATURES]

    print("Test dataset shape:", data.shape)

    rf_model = joblib.load(RF_MODEL_FILE)
    label_encoder = joblib.load(RF_ENCODER_FILE)
    if_model = joblib.load(IF_MODEL_FILE)

    print("\nModels loaded successfully.")

    # -----------------------------
    # Random Forest predictions
    # -----------------------------

    rf_encoded = rf_model.predict(X_test)

    rf_predictions = label_encoder.inverse_transform(
        rf_encoded
    )

    # -----------------------------
    # Isolation Forest scores
    # -----------------------------

    if_scores = if_model.decision_function(X_test)

    if_anomaly = (
        if_scores < IF_THRESHOLD
    ).astype(int)

    # -----------------------------
    # Hybrid decision logic
    # -----------------------------

    hybrid_predictions = []

    for rf_pred, anomaly_flag in zip(
        rf_predictions,
        if_anomaly
    ):

        if rf_pred != "Benign":

            # Known supervised attack
            hybrid_predictions.append(rf_pred)

        else:

            if anomaly_flag == 1:

                # RF says benign but Isolation Forest
                # detects unusual behaviour
                hybrid_predictions.append(
                    "Anomalous_Unknown"
                )

            else:

                hybrid_predictions.append(
                    "Benign"
                )

    data["rf_prediction"] = rf_predictions
    data["if_anomaly"] = if_anomaly
    data["if_score"] = if_scores
    data["hybrid_prediction"] = hybrid_predictions

    # -----------------------------
    # Multiclass hybrid analysis
    # -----------------------------

    print("\nHybrid prediction counts")
    print("------------------------")

    print(
        data["hybrid_prediction"]
        .value_counts()
    )

    print("\nHybrid predictions by true label")
    print("--------------------------------")

    print(
        pd.crosstab(
            data["label"],
            data["hybrid_prediction"]
        )
    )

    # -----------------------------
    # Binary evaluation
    # -----------------------------
    # Ground truth:
    # Benign = 0
    # attacks = 1

    y_true_binary = (
        data["label"] != "Benign"
    ).astype(int)

    # Hybrid:
    # anything other than Benign
    # is considered anomalous/attack
    y_pred_binary = (
        data["hybrid_prediction"] != "Benign"
    ).astype(int)

    accuracy = accuracy_score(
        y_true_binary,
        y_pred_binary
    )

    precision = precision_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0
    )

    recall = recall_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0
    )

    f1 = f1_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0.0

    print("\nBinary Hybrid Metrics")
    print("---------------------")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")

    print("\nBinary Classification Report")
    print("----------------------------")

    print(
        classification_report(
            y_true_binary,
            y_pred_binary,
            target_names=[
                "Benign",
                "Anomalous"
            ],
            digits=4,
            zero_division=0
        )
    )

    print("Confusion Matrix")
    print("----------------")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(cm)

    print("\nScenario-level Hybrid Results")
    print("-----------------------------")

    scenario_summary = (
        data
        .groupby("scenario")
        .agg(
            records=("scenario", "size"),
            hybrid_flagged=(
                "hybrid_prediction",
                lambda x: (x != "Benign").sum()
            )
        )
    )

    scenario_summary["flag_rate"] = (
        scenario_summary["hybrid_flagged"]
        / scenario_summary["records"]
    )

    print(scenario_summary)

    OUTPUT_FILE = str(
        PROJECT_ROOT / "results/hybrid_model_predictions.csv"
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved hybrid predictions to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
