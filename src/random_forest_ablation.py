from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


TRAIN_FILE = str(PROJECT_ROOT / "data/campus_dns_features.csv")
TEST_FILE = str(PROJECT_ROOT / "data/campus_dns_test_features.csv")


FEATURE_SETS = {
    "All_7_features": [
        "subdomain_length",
        "query_entropy",
        "inter_query_interval",
        "interval_std_10q",
        "queries_last_5s",
        "nxdomain_ratio_1s",
        "repeated_query_ratio_1s"
    ],

    "Without_subdomain_length": [
        "query_entropy",
        "inter_query_interval",
        "interval_std_10q",
        "queries_last_5s",
        "nxdomain_ratio_1s",
        "repeated_query_ratio_1s"
    ],

    "Without_nxdomain_ratio": [
        "subdomain_length",
        "query_entropy",
        "inter_query_interval",
        "interval_std_10q",
        "queries_last_5s",
        "repeated_query_ratio_1s"
    ],

    "Without_both": [
        "query_entropy",
        "inter_query_interval",
        "interval_std_10q",
        "queries_last_5s",
        "repeated_query_ratio_1s"
    ]
}


def main():

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(train["label"])
    y_test = encoder.transform(test["label"])

    results = []

    for experiment, features in FEATURE_SETS.items():

        print(f"\nRunning: {experiment}")
        print("Features:", features)

        X_train = train[features]
        X_test = test[features]

        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        results.append({
            "experiment": experiment,
            "features_used": len(features),
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        })

    results_df = pd.DataFrame(results)

    print("\nAblation Results")
    print("----------------")
    print(results_df.to_string(index=False))


    results_df.to_csv(
        str(PROJECT_ROOT / "results/random_forest_ablation_results.csv"),
        index=False
    )

    print(
        "\nSaved:",
        str(PROJECT_ROOT / "results/random_forest_ablation_results.csv")
    )


if __name__ == "__main__":
    main()

