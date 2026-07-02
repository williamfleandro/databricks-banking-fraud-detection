import argparse
import json
import os
import sys

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="Path to JSON payload file.")
    args = parser.parse_args()

    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    endpoint = os.environ.get("DATABRICKS_ENDPOINT_NAME", "fraud-detection-endpoint-prod")

    if not host or not token:
        print("DATABRICKS_HOST and DATABRICKS_TOKEN must be set.", file=sys.stderr)
        return 1

    with open(args.payload, "r", encoding="utf-8") as f:
        payload = json.load(f)

    url = f"{host.rstrip('/')}/serving-endpoints/{endpoint}/invocations"

    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=(30, 180),
    )

    print(response.status_code)
    print(response.text)
    response.raise_for_status()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
