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

MODEL_FILE = str(PROJECT_ROOT / "models/random_forest_model.pkl")
ENCODER_FILE = str(PROJECT_ROOT / "models/random_forest_label_encoder.pkl")


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
    y_test = data["label"]

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)

    print("\nLoaded Random Forest model.")
    print("Classes:", list(encoder.classes_))

    y_test_encoded = encoder.transform(y_test)

    print("\nRunning predictions on independent test dataset...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)

    precision_macro = precision_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\nOverall Metrics")
    print("----------------")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall:    {recall_macro:.4f}")
    print(f"Macro F1-score:  {f1_macro:.4f}")

    print("\nClassification Report")
    print("---------------------")

    print(
        classification_report(
            y_test_encoded,
            y_pred,
            target_names=encoder.classes_,
            digits=4,
            zero_division=0
        )
    )

    print("Confusion Matrix")
    print("----------------")

    cm = confusion_matrix(y_test_encoded, y_pred)

    print(cm)


if __name__ == "__main__":
    main()
