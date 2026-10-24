from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


TRAIN_FILE = str(PROJECT_ROOT / "data/campus_dns_features.csv")

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

    print("Loading training dataset...")

    data = pd.read_csv(TRAIN_FILE)

    print("Dataset shape:", data.shape)

    X = data[FEATURES]
    y = data["label"]

    print("\nSelected features:")
    for feature in FEATURES:
        print("-", feature)

    print("\nTraining class distribution:")
    print(y.value_counts())

    # Convert text labels to numerical class values
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("\nEncoded classes:")
    for index, label in enumerate(encoder.classes_):
        print(f"{index}: {label}")

    # Random Forest supervised classifier
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    model.fit(X, y_encoded)

    print("Random Forest training completed.")

    # Save trained model and label encoder
    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    print("\nSaved model to:")
    print(MODEL_FILE)

    print("\nSaved label encoder to:")
    print(ENCODER_FILE)


if __name__ == "__main__":
    main()
