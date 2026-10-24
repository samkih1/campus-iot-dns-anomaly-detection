from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest


TRAIN_FILE = str(PROJECT_ROOT / "data/campus_dns_features.csv")

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

    print("Loading training dataset...")

    data = pd.read_csv(TRAIN_FILE)

    print("Dataset shape:", data.shape)

    # Isolation Forest is trained only on benign DNS behaviour
    benign_data = data[data["label"] == "Benign"].copy()

    print("\nBenign training records:", len(benign_data))

    print("\nBenign scenario distribution:")
    print(benign_data["scenario"].value_counts())

    X_train = benign_data[FEATURES]

    print("\nSelected features:")
    for feature in FEATURES:
        print("-", feature)

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Isolation Forest on benign records...")

    model.fit(X_train)

    print("Isolation Forest training completed.")

    joblib.dump(model, MODEL_FILE)

    print("\nSaved model to:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()
