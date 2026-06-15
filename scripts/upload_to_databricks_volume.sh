#!/usr/bin/env bash
set -euo pipefail
databricks fs cp data/raw/creditcard.csv dbfs:/Volumes/mlops_dev/banking/raw/creditcard.csv --overwrite
