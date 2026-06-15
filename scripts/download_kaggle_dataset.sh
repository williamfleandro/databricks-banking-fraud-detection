#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
ls -lh data/raw
