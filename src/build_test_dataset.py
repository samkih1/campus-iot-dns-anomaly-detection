from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import csv
import os

BASE_DIR = str(PROJECT_ROOT / "raw_data/zeek_test")
OUTPUT_FILE = str(PROJECT_ROOT / "raw_data/campus_dns_test_dataset.csv")

DATASETS = {
    "normal": ("Benign", "Normal"),
    "busy": ("Benign", "Busy"),
    "peak": ("Benign", "Peak"),
    "nxdomain": ("NXDOMAIN_Flood", "NXDOMAIN_Flood"),
    "tunnel": ("DNS_Tunnel", "DNS_Tunnel"),
}

SELECTED_FIELDS = [
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "rtt",
    "query",
    "qtype_name",
    "rcode",
    "rcode_name",
    "answers",
    "TTLs",
]


def read_zeek_log(file_path):

    fields = []

    with open(file_path, "r") as file:

        for line in file:

            if line.startswith("#fields"):
                fields = line.strip().split("\t")[1:]
                continue

            if line.startswith("#"):
                continue

            values = line.rstrip("\n").split("\t")

            if not fields:
                continue

            yield dict(zip(fields, values))


def main():

    total_records = 0

    with open(OUTPUT_FILE, "w", newline="") as output:

        writer = csv.DictWriter(
            output,
            fieldnames=SELECTED_FIELDS + ["scenario", "label"]
        )

        writer.writeheader()

        for folder, (label, scenario) in DATASETS.items():

            dns_log = os.path.join(
                BASE_DIR,
                folder,
                "dns.log"
            )

            scenario_count = 0

            for record in read_zeek_log(dns_log):

                row = {
                    field: record.get(field, "")
                    for field in SELECTED_FIELDS
                }

                row["scenario"] = scenario
                row["label"] = label

                writer.writerow(row)

                scenario_count += 1
                total_records += 1

            print(
                f"{scenario}: {scenario_count} records"
            )

    print(f"\nTotal records: {total_records}")
    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
