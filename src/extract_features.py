from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent





import math
from collections import Counter

import pandas as pd


INPUT_FILE = str(PROJECT_ROOT / "raw_data/campus_dns_dataset.csv")
OUTPUT_FILE = str(PROJECT_ROOT / "data/campus_dns_features.csv")


def calculate_entropy(text):
    """Calculate Shannon entropy of a DNS query string."""

    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)

    entropy = 0.0

    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def get_subdomain(query):
    """
    Extract the portion before the registered research domain.

    Examples:
    cloud.campus.test -> cloud
    abc123.tunnel.campus.test -> abc123
    """

    query = str(query).rstrip(".")
    parts = query.split(".")

    if query.endswith(".tunnel.campus.test") and len(parts) > 3:
        return ".".join(parts[:-3])

    if query.endswith(".campus.test") and len(parts) > 2:
        return ".".join(parts[:-2])

    return ""


def main():

    df = pd.read_csv(INPUT_FILE)

    # Ensure DNS query is treated as text
    df["query"] = df["query"].fillna("").astype(str)
    # Convert timestamp to numeric
    df["ts"] = pd.to_numeric(df["ts"])

    # Sort records within each experimental scenario and source device
    df = df.sort_values(
        ["scenario", "id.orig_h", "ts"]
    ).reset_index(drop=True)

    # Time since the previous DNS query from the same device
    # within the same experimental scenario
    df["inter_query_interval"] = (
        df.groupby(
            ["scenario", "id.orig_h"]
        )["ts"]
        .diff()
        .fillna(0)
    )

# Variation in inter-query intervals over the
# most recent 10 DNS queries from the same source
    # Variation in inter-query intervals over the
    # most recent 10 DNS queries from the same source
    df["interval_std_10q"] = (
        df.groupby(
            ["scenario", "id.orig_h"]
        )["inter_query_interval"]
        .transform(
            lambda x: x.rolling(
                window=10,
                min_periods=2
            ).std()
        )
        .fillna(0)
    )
  # Number of DNS queries from the same device
    # within the current and preceding 1-second window
    df["queries_last_1s"] = 0

    for (scenario, source), group in df.groupby(
        ["scenario", "id.orig_h"],
        sort=False
    ):

        timestamps = group["ts"].to_list()
        counts = []

        left = 0

        for right in range(len(timestamps)):

            while (
                timestamps[right] - timestamps[left] > 1.0
            ):
                left += 1

            counts.append(right - left + 1)

        df.loc[
            group.index,
            "queries_last_1s"
        ] = counts

    # Number of DNS queries from the same device
    # within the current and preceding 5-second window
    df["queries_last_5s"] = 0

    for (scenario, source), group in df.groupby(
        ["scenario", "id.orig_h"],
        sort=False
    ):

        timestamps = group["ts"].to_list()
        counts = []

        left = 0

        for right in range(len(timestamps)):

            while (
                timestamps[right] - timestamps[left] > 5.0
            ):
                left += 1

            counts.append(right - left + 1)

        df.loc[
            group.index,
            "queries_last_5s"
        ] = counts
    # Variability of inter-query intervals over the
    # most recent 10 DNS queries from the same device
    df["interval_std_10q"] = (
        df.groupby(
            ["scenario", "id.orig_h"]
        )["inter_query_interval"]
        .transform(
            lambda x: x.rolling(
                window=10,
                min_periods=2
            ).std()
        )
        .fillna(0)
    )
    # Mark whether the current DNS response is NXDOMAIN
    df["is_nxdomain"] = (
        df["rcode_name"]
        .fillna("")
        .eq("NXDOMAIN")
        .astype(int)
    )

    # Proportion of NXDOMAIN responses within the current
    # and preceding 1-second window for the same device
    df["nxdomain_ratio_1s"] = 0.0

    for (scenario, source), group in df.groupby(
        ["scenario", "id.orig_h"],
        sort=False
    ):

        timestamps = group["ts"].to_list()
        nxdomain_flags = group["is_nxdomain"].to_list()

        ratios = []

        left = 0
        running_nxdomain = 0

        for right in range(len(timestamps)):

            running_nxdomain += nxdomain_flags[right]

            while (
                timestamps[right] - timestamps[left] > 1.0
            ):
                running_nxdomain -= nxdomain_flags[left]
                left += 1

            window_size = right - left + 1

            ratios.append(
                running_nxdomain / window_size
            )

        df.loc[
            group.index,
            "nxdomain_ratio_1s"
        ] = ratios
    # Ratio of unique DNS queries within the current
    # and preceding 1-second window for the same device
    df["unique_query_ratio_1s"] = 0.0

    for (scenario, source), group in df.groupby(
        ["scenario", "id.orig_h"],
        sort=False
    ):

        timestamps = group["ts"].to_list()
        queries = group["query"].to_list()

        ratios = []

        left = 0

        for right in range(len(timestamps)):

            while (
                timestamps[right] - timestamps[left] > 1.0
            ):
                left += 1

            window_queries = queries[left:right + 1]

            unique_count = len(set(window_queries))
            window_size = len(window_queries)

            ratios.append(
                unique_count / window_size
            )

        df.loc[
            group.index,
            "unique_query_ratio_1s"
        ] = ratios
    # Proportion of repeated DNS queries within
    # the trailing 1-second window
    df["repeated_query_ratio_1s"] = (
        1.0 - df["unique_query_ratio_1s"]
    )
    # Total DNS query length
    df["query_length"] = df["query"].apply(len)

    # Total DNS query length
    df["query_length"] = df["query"].apply(len)

    # Extract subdomain
    df["subdomain"] = df["query"].apply(get_subdomain)

    # Length of extracted subdomain
    df["subdomain_length"] = df["subdomain"].apply(len)

    # Number of DNS labels
    df["label_count"] = df["query"].apply(
        lambda query: len(query.rstrip(".").split("."))
    )

    # Shannon entropy of the extracted subdomain
    df["query_entropy"] = df["subdomain"].apply(
        calculate_entropy
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("Feature extraction completed.")
    print("Dataset shape:", df.shape)

    print("\nFeature summary by scenario:")

    print(
        df.groupby("scenario")[
            [
                "query_length",
                "subdomain_length",
                "label_count",
                "query_entropy"
            ]
        ].mean().round(3)
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
