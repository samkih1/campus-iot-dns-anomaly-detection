from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import os
import time
import threading

import joblib
import pandas as pd
import psutil


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


def monitor_memory(process, stop_event, memory_samples):

    while not stop_event.is_set():

        memory_samples.append(
            process.memory_info().rss
        )

        time.sleep(0.001)


def measure_model(name, prediction_function, repeats=100):

    process = psutil.Process(os.getpid())

    # Warm-up
    prediction_function()

    memory_before = process.memory_info().rss

    process.cpu_percent(interval=None)

    memory_samples = []
    stop_event = threading.Event()

    monitor = threading.Thread(
        target=monitor_memory,
        args=(
            process,
            stop_event,
            memory_samples
        )
    )

    monitor.start()

    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()

    for _ in range(repeats):
        prediction_function()

    wall_end = time.perf_counter()
    cpu_end = process.cpu_times()

    stop_event.set()
    monitor.join()

    memory_after = process.memory_info().rss

    wall_time = wall_end - wall_start

    cpu_time = (
        (cpu_end.user - cpu_start.user)
        + (cpu_end.system - cpu_start.system)
    )

    if wall_time > 0:
        cpu_percent_single_core = (
            cpu_time / wall_time
        ) * 100
    else:
        cpu_percent_single_core = 0.0

    if memory_samples:
        peak_rss = max(memory_samples)
    else:
        peak_rss = memory_after

    peak_increase = max(
        0,
        peak_rss - memory_before
    )

    print(f"\n{name}")
    print("-" * len(name))

    print(f"Repeated inference runs: {repeats}")

    print(
        f"Total wall time: "
        f"{wall_time:.6f} seconds"
    )

    print(
        f"Total process CPU time: "
        f"{cpu_time:.6f} seconds"
    )

    print(
        f"Process CPU utilisation "
        f"(single-core equivalent): "
        f"{cpu_percent_single_core:.2f}%"
    )

    print(
        f"RSS before inference: "
        f"{memory_before / (1024 ** 2):.3f} MB"
    )

    print(
        f"RSS after inference: "
        f"{memory_after / (1024 ** 2):.3f} MB"
    )

    print(
        f"Peak observed RSS: "
        f"{peak_rss / (1024 ** 2):.3f} MB"
    )

    print(
        f"Peak RSS increase during inference: "
        f"{peak_increase / (1024 ** 2):.3f} MB"
    )


def main():

    print("Loading test dataset and models...")

    data = pd.read_csv(TEST_FILE)

    X_test = data[FEATURES]

    rf_model = joblib.load(RF_MODEL_FILE)
    rf_encoder = joblib.load(RF_ENCODER_FILE)
    if_model = joblib.load(IF_MODEL_FILE)

    print("Test records:", len(X_test))
    print("Models loaded successfully.")

    def run_rf():

        encoded = rf_model.predict(X_test)

        rf_encoder.inverse_transform(
            encoded
        )

    def run_if():

        scores = if_model.decision_function(
            X_test
        )

        (
            scores < IF_THRESHOLD
        ).astype(int)

    def run_hybrid():

        rf_encoded = rf_model.predict(
            X_test
        )

        rf_predictions = (
            rf_encoder.inverse_transform(
                rf_encoded
            )
        )

        if_scores = (
            if_model.decision_function(
                X_test
            )
        )

        if_flags = (
            if_scores < IF_THRESHOLD
        ).astype(int)

        hybrid_predictions = []

        for rf_pred, if_flag in zip(
            rf_predictions,
            if_flags
        ):

            if rf_pred != "Benign":

                hybrid_predictions.append(
                    rf_pred
                )

            elif if_flag == 1:

                hybrid_predictions.append(
                    "Anomalous_Unknown"
                )

            else:

                hybrid_predictions.append(
                    "Benign"
                )

        return hybrid_predictions

    measure_model(
        "Random Forest",
        run_rf
    )

    measure_model(
        "Isolation Forest",
        run_if
    )

    measure_model(
        "Hybrid RF + Isolation Forest",
        run_hybrid
    )


if __name__ == "__main__":
    main()
