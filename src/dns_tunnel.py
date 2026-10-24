
import argparse
import random
import string
import subprocess
import time
from datetime import datetime


DNS_SERVER = "10.0.0.53"


def random_payload(length):
    return ''.join(
        random.choices(
            string.ascii_lowercase + string.digits,
            k=length
        )
    )


def generate_tunnel_traffic(queries, min_delay, max_delay, payload_length):

    print("Starting DNS tunnelling simulation")
    print(f"Number of queries: {queries}")
    print(f"Payload length: {payload_length}")

    for number in range(1, queries + 1):

        payload = random_payload(payload_length)

        domain = f"{payload}.tunnel.campus.test"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"[{timestamp}] "
            f"Tunnel query {number}/{queries}: {domain}"
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

        time.sleep(
            random.uniform(
                min_delay,
                max_delay
            )
        )

    print("Completed DNS tunnelling simulation")


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
        default=0.02
    )

    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.08
    )

    parser.add_argument(
        "--payload-length",
        type=int,
        default=40
    )

    args = parser.parse_args()

    generate_tunnel_traffic(
        args.queries,
        args.min_delay,
        args.max_delay,
        args.payload_length
    )
