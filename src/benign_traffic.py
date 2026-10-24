import argparse
import random
import subprocess
import time
from datetime import datetime


DNS_SERVER = "10.0.0.53"

DEVICE_DOMAINS = {
    "cam": [
        "cloud.campus.test",
        "updates.campus.test",
        "services.campus.test"
    ],

    "sens": [
        "cloud.campus.test",
        "updates.campus.test"
    ],

    "class": [
        "elearning.campus.test",
        "cloud.campus.test",
        "services.campus.test"
    ],

    "acad": [
        "elearning.campus.test",
        "services.campus.test",
        "updates.campus.test"
    ]
}


def generate_traffic(device, queries, min_delay, max_delay):

    if device not in DEVICE_DOMAINS:
        print(f"Unknown device: {device}")
        return

    domains = DEVICE_DOMAINS[device]

    print(f"Starting benign DNS traffic for {device}")
    print(f"Number of queries: {queries}")

    for number in range(1, queries + 1):

        domain = random.choice(domains)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"[{timestamp}] "
            f"{device} query {number}/{queries}: {domain}"
        )

        subprocess.run(
            [
                "dig",
                f"@{DNS_SERVER}",
                domain,
                "A",
                "+short"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        time.sleep(random.uniform(min_delay, max_delay))

    print(f"Completed benign DNS traffic for {device}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        required=True,
choices=["cam", "sens", "class", "acad"]
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=20
    )

    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--max-delay",
        type=float,
        default=2.0
    )

    args = parser.parse_args()

    generate_traffic(
        args.device,
        args.queries,
        args.min_delay,
        args.max_delay
    )
