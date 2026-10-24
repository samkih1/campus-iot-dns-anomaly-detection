from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import joblib


TRAIN_FILE = str(PROJECT_ROOT / "data/campus_dns_features.csv")
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

    data = pd.read_csv(TRAIN_FILE)

    normal = data[
        data["scenario"] == "Normal"
    ].copy()

    X_normal = normal[FEATURES]

    model = joblib.load(MODEL_FILE)

    scores = model.decision_function(X_normal)

    normal["anomaly_score"] = scores
    normal["predicted_anomaly"] = (
        normal["anomaly_score"] < THRESHOLD
    ).astype(int)

    print("Training Normal records:", len(normal))

    print("\nPrediction counts")
    print("-----------------")
    print(
        normal["predicted_anomaly"]
        .value_counts()
        .sort_index()
    )

    print("\nAnomaly score summary")
    print("---------------------")
    print(
        normal["anomaly_score"]
        .describe()
        .round(6)
    )

    flagged = normal[
        normal["predicted_anomaly"] == 1
    ]

    accepted = normal[
        normal["predicted_anomaly"] == 0
    ]

    print("\nFLAGGED TRAINING NORMAL FEATURE MEANS")
    print("-------------------------------------")
    print(
        flagged[FEATURES]
        .mean()
        .round(6)
    )

    print("\nACCEPTED TRAINING NORMAL FEATURE MEANS")
    print("--------------------------------------")
    print(
        accepted[FEATURES]
        .mean()
        .round(6)
    )

    print("\nMEAN DIFFERENCE: FLAGGED - ACCEPTED")
    print("-----------------------------------")
    print(
        (
            flagged[FEATURES].mean()
            - accepted[FEATURES].mean()
        ).round(6)
    )


if __name__ == "__main__":
    main()
