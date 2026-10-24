from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd


TRAIN_FILE = str(PROJECT_ROOT / "data/campus_dns_features.csv")
TEST_FILE = str(PROJECT_ROOT / "data/campus_dns_test_features.csv")


FEATURES = [
    "subdomain_length",
    "query_entropy",
    "inter_query_interval",
    "interval_std_10q",
    "queries_last_5s",
    "nxdomain_ratio_1s",
    "repeated_query_ratio_1s"
]


def summarize(df, name):

    print(f"\n{name}")
    print("-" * len(name))

    summary = df[FEATURES].agg(
        ["mean", "median", "std", "min", "max"]
    ).T

    print(summary.round(6))


def main():

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    train_normal = train[
        train["scenario"] == "Normal"
    ].copy()

    test_normal = test[
        test["scenario"] == "Normal"
    ].copy()

    print("Training Normal records:", len(train_normal))
    print("Test Normal records:", len(test_normal))

    summarize(
        train_normal,
        "TRAINING NORMAL FEATURE SUMMARY"
    )

    summarize(
        test_normal,
        "TEST NORMAL FEATURE SUMMARY"
    )

    print("\nMEAN DIFFERENCE: TEST - TRAIN")
    print("-----------------------------")

    difference = (
        test_normal[FEATURES].mean()
        - train_normal[FEATURES].mean()
    )

    print(difference.round(6))


if __name__ == "__main__":
    main()
