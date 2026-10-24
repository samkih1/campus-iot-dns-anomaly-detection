from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import joblib
import time
import tracemalloc

from sklearn.metrics import accuracy_score


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


def measure_random_forest(model, encoder, X):

    tracemalloc.start()

    start = time.perf_counter()

    encoded_predictions = model.predict(X)

    end = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    predictions = encoder.inverse_transform(
        encoded_predictions
    )

    elapsed = end - start

    return predictions, elapsed, peak


def measure_isolation_forest(model, X):

    tracemalloc.start()

    start = time.perf_counter()

    scores = model.decision_function(X)

    predictions = (
        scores < IF_THRESHOLD
    ).astype(int)

    end = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    elapsed = end - start

    return predictions, elapsed, peak


def main():

    data = pd.read_csv(TEST_FILE)

    X_test = data[FEATURES]

    print("Test records:", len(X_test))

    rf_model = joblib.load(RF_MODEL_FILE)
    rf_encoder = joblib.load(RF_ENCODER_FILE)
    if_model = joblib.load(IF_MODEL_FILE)

    print("\nModels loaded successfully.")

    # Warm-up runs
    rf_model.predict(X_test.iloc[:10])
    if_model.decision_function(X_test.iloc[:10])

    print("\nMeasuring Random Forest...")

    rf_predictions, rf_time, rf_peak_memory = (
        measure_random_forest(
            rf_model,
            rf_encoder,
            X_test
        )
    )

    rf_per_record_ms = (
        rf_time / len(X_test)
    ) * 1000

    rf_throughput = (
        len(X_test) / rf_time
    )

    print(f"Total inference time: {rf_time:.6f} seconds")
    print(
        f"Average time per record: "
        f"{rf_per_record_ms:.6f} ms"
    )
    print(
        f"Throughput: "
        f"{rf_throughput:.2f} records/second"
    )
    print(
        f"Peak Python memory: "
        f"{rf_peak_memory / (1024 * 1024):.3f} MB"
    )

    print("\nMeasuring Isolation Forest...")

    if_predictions, if_time, if_peak_memory = (
        measure_isolation_forest(
            if_model,
            X_test
        )
    )

    if_per_record_ms = (
        if_time / len(X_test)
    ) * 1000

    if_throughput = (
        len(X_test) / if_time
    )

    print(f"Total inference time: {if_time:.6f} seconds")
    print(
        f"Average time per record: "
        f"{if_per_record_ms:.6f} ms"
    )
    print(
        f"Throughput: "
        f"{if_throughput:.2f} records/second"
    )
    print(
        f"Peak Python memory: "
        f"{if_peak_memory / (1024 * 1024):.3f} MB"
    )

    print("\nMeasuring Hybrid sequential inference...")

    tracemalloc.start()

    start = time.perf_counter()

    rf_encoded = rf_model.predict(X_test)

    rf_predictions = rf_encoder.inverse_transform(
        rf_encoded
    )

    if_scores = if_model.decision_function(X_test)

    if_flags = (
        if_scores < IF_THRESHOLD
    ).astype(int)

    hybrid_predictions = []

    for rf_pred, if_flag in zip(
        rf_predictions,
        if_flags
    ):

        if rf_pred != "Benign":
            hybrid_predictions.append(rf_pred)

        elif if_flag == 1:
            hybrid_predictions.append(
                "Anomalous_Unknown"
            )

        else:
            hybrid_predictions.append(
                "Benign"
            )

    end = time.perf_counter()

    current, hybrid_peak_memory = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    hybrid_time = end - start

    hybrid_per_record_ms = (
        hybrid_time / len(X_test)
    ) * 1000

    hybrid_throughput = (
        len(X_test) / hybrid_time
    )

    print(
        f"Total inference time: "
        f"{hybrid_time:.6f} seconds"
    )
    print(
        f"Average time per record: "
        f"{hybrid_per_record_ms:.6f} ms"
    )
    print(
        f"Throughput: "
        f"{hybrid_throughput:.2f} records/second"
    )
    print(
        f"Peak Python memory: "
        f"{hybrid_peak_memory / (1024 * 1024):.3f} MB"
    )


if __name__ == "__main__":
    main()
