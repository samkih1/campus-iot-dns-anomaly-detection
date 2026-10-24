from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest


TRAIN_FILE = str(
    Path(__file__).resolve().parent.parent / "data/campus_dns_features.csv"
)

TEST_FILE = str(
    Path(__file__).resolve().parent.parent / "data/campus_dns_test_features.csv"
)

FEATURES = [
    "subdomain_length",
    "query_entropy",
    "inter_query_interval",
    "interval_std_10q",
    "queries_last_5s",
    "nxdomain_ratio_1s",
    "repeated_query_ratio_1s"
]

BENIGN_CLASS = "Benign"

EXPERIMENTS = [
    {
        "experiment": "Hold_out_DNS_Tunnel",
        "known_attack": "NXDOMAIN_Flood",
        "held_out_attack": "DNS_Tunnel"
    },
    {
        "experiment": "Hold_out_NXDOMAIN_Flood",
        "known_attack": "DNS_Tunnel",
        "held_out_attack": "NXDOMAIN_Flood"
    }
]


def evaluate_experiment(
    development,
    test,
    experiment,
    known_attack,
    held_out_attack
):
    training_subset = development[
        development["label"].isin(
            [BENIGN_CLASS, known_attack]
        )
    ].copy()

    X_train = training_subset[FEATURES]
    y_train = training_subset["label"]

    X_test = test[FEATURES]
    y_test = test["label"]

    # Random Forest is fitted without the held-out attack class.
    random_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(X_train, y_train)

    rf_predictions = random_forest.predict(X_test)

    # Isolation Forest is fitted only on benign development records.
    benign_development = development[
        development["label"] == BENIGN_CLASS
    ][FEATURES]

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    isolation_forest.fit(benign_development)

    benign_training_scores = (
        isolation_forest.decision_function(
            benign_development
        )
    )

    threshold = np.percentile(
        benign_training_scores,
        5
    )

    final_predictions = rf_predictions.astype(
        object
    ).copy()

    rf_benign_mask = (
        rf_predictions == BENIGN_CLASS
    )

    if_scores = np.full(
        len(test),
        np.nan
    )

    if_scores[rf_benign_mask] = (
        isolation_forest.decision_function(
            X_test.loc[rf_benign_mask]
        )
    )

    anomalous_mask = (
        rf_benign_mask &
        (if_scores < threshold)
    )

    final_predictions[anomalous_mask] = (
        "Anomalous_Unknown"
    )

    held_out_mask = (
        y_test == held_out_attack
    )

    benign_test_mask = (
        y_test == BENIGN_CLASS
    )

    known_attack_mask = (
        y_test == known_attack
    )

    held_out_total = int(
        held_out_mask.sum()
    )

    held_out_outputs = pd.Series(
        final_predictions[held_out_mask]
    ).value_counts()

    result = {
        "experiment": experiment,
        "known_attack": known_attack,
        "held_out_attack": held_out_attack,
        "training_records": len(training_subset),
        "benign_training_records": int(
            (y_train == BENIGN_CLASS).sum()
        ),
        "known_attack_training_records": int(
            (y_train == known_attack).sum()
        ),
        "threshold": threshold,
        "held_out_test_records": held_out_total,
        "held_out_predicted_known_attack": int(
            (
                final_predictions[held_out_mask]
                == known_attack
            ).sum()
        ),
        "held_out_anomalous_unknown": int(
            (
                final_predictions[held_out_mask]
                == "Anomalous_Unknown"
            ).sum()
        ),
        "held_out_retained_benign": int(
            (
                final_predictions[held_out_mask]
                == BENIGN_CLASS
            ).sum()
        ),
        "held_out_unknown_detection_rate": (
            (
                final_predictions[held_out_mask]
                == "Anomalous_Unknown"
            ).sum() / held_out_total
        ),
        "held_out_total_flagged_rate": (
            (
                final_predictions[held_out_mask]
                != BENIGN_CLASS
            ).sum() / held_out_total
        ),
        "known_attack_correctly_classified": int(
            (
                final_predictions[known_attack_mask]
                == known_attack
            ).sum()
        ),
        "known_attack_test_records": int(
            known_attack_mask.sum()
        ),
        "benign_false_positives": int(
            (
                final_predictions[benign_test_mask]
                != BENIGN_CLASS
            ).sum()
        ),
        "benign_test_records": int(
            benign_test_mask.sum()
        )
    }

    detailed_results = test.copy()
    detailed_results["experiment"] = experiment
    detailed_results["rf_prediction"] = rf_predictions
    detailed_results["if_score"] = if_scores
    detailed_results["final_prediction"] = (
        final_predictions
    )

    detailed_results.to_csv(
        f"{experiment}_predictions.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print(experiment)
    print("=" * 70)
    print("Known attack:", known_attack)
    print("Held-out attack:", held_out_attack)
    print("Training records:", len(training_subset))
    print("Calculated threshold:", threshold)

    print("\nHeld-out attack outputs:")
    print(held_out_outputs.to_string())

    print(
        "\nAnomalous_Unknown detection rate:",
        f"{result['held_out_unknown_detection_rate']:.4f}"
    )

    print(
        "Total held-out attack flagging rate:",
        f"{result['held_out_total_flagged_rate']:.4f}"
    )

    print(
        "Benign false positives:",
        result["benign_false_positives"],
        "of",
        result["benign_test_records"]
    )

    return result


def main():
    development = pd.read_csv(
        TRAIN_FILE
    )

    test = pd.read_csv(
        TEST_FILE
    )

    print("Development records:", len(development))
    print("Independent test records:", len(test))

    print("\nDevelopment class distribution:")
    print(
        development["label"]
        .value_counts()
        .to_string()
    )

    print("\nTest class distribution:")
    print(
        test["label"]
        .value_counts()
        .to_string()
    )

    summary_results = []

    for settings in EXPERIMENTS:
        result = evaluate_experiment(
            development=development,
            test=test,
            **settings
        )

        summary_results.append(result)

    summary = pd.DataFrame(
        summary_results
    )

    output_file = str(
        Path(__file__).resolve().parent.parent / "results/held_out_attack_results.csv"
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("HELD-OUT ATTACK SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()

