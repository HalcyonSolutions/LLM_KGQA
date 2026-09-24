#!/usr/bin/env bash
set -euo pipefail

# This preprocessing assumes that the Kinship dataset has been downloaded to:
#   ./raw_data/kinship_hinton
#
# For example:
#   hf download HalcyonSolutions/Kinship \
#       --repo-type dataset \
#       --local-dir ./raw_data/kinship_hinton
#
# This script copies the required KG and QA files into:
#   ./data/kinship/

RAW_DIR="./raw_data/kinship_hinton"
OUTPUT_DIR="./data/kinship"

mkdir -p "${OUTPUT_DIR}"

cp "${RAW_DIR}/kg/orig/triplets.txt" \
   "${OUTPUT_DIR}/triplets.txt"

cp "${RAW_DIR}/qa/kinship_qa_nhop.csv" \
   "${OUTPUT_DIR}/qa_nhop.csv"

echo "Kinship preprocessing complete."
echo "Files written to ${OUTPUT_DIR}/"