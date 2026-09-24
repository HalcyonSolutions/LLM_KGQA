#!/usr/bin/env bash
set -euo pipefail

# Preprocess the MQuAKE-ST dataset.
#
# This script assumes that the dataset has been downloaded to:
#   ./raw_data/mquake_st_dataset
#
# For example:
#   hf download HalcyonSolutions/MQuAKE-ST \
#       --repo-type dataset \
#       --local-dir ./raw_data/mquake_st_dataset
#
# The processed files are written to:
#   ./data/mquake_single/
#   ./data/mquake_multi/

RAW_DIR="./raw_data/mquake_st_dataset"
OUTPUT_DIR_SA="./data/mquake_single"
OUTPUT_DIR_MA="./data/mquake_multi"

mkdir -p "${OUTPUT_DIR_SA}" "${OUTPUT_DIR_MA}"

# Copy files shared by both single-answer and multi-answer variants.
for OUTPUT_DIR in "${OUTPUT_DIR_SA}" "${OUTPUT_DIR_MA}"; do
    # Knowledge graph
    cp "${RAW_DIR}/kg/triplets.txt" \
       "${OUTPUT_DIR}/triplets.txt"

    # Metadata
    cp "${RAW_DIR}/metadata/node_data.csv" \
       "${OUTPUT_DIR}/node_data.csv"

    cp "${RAW_DIR}/metadata/relation_data.csv" \
       "${OUTPUT_DIR}/relation_data.csv"
done

# Single-answer QA data
cp "${RAW_DIR}/qa/single_answers/qa_nhop.csv" \
   "${OUTPUT_DIR_SA}/qa_nhop.csv"

# Multi-answer QA data
cp "${RAW_DIR}/qa/multi_answers/qa_nhop.csv" \
   "${OUTPUT_DIR_MA}/qa_nhop.csv"

echo "MQuAKE-ST preprocessing complete."
echo "Single-answer data written to: ${OUTPUT_DIR_SA}/"
echo "Multi-answer data written to:  ${OUTPUT_DIR_MA}/"