#!/usr/bin/env bash
# Runs both QA conditions (baseline, transfer) for BanglaT5, then evaluates both.
# Safe to re-run after an interruption: train.py skips any condition whose
# outputs/model/<condition>/best already exists, and resumes mid-condition
# from the latest epoch checkpoint if one exists.
set -e
cd "$(dirname "$0")"

echo "=== [1/4] Train BanglaT5 QA -- baseline ==="
.venv/bin/python src/train.py --source base

echo "=== [2/4] Evaluate baseline ==="
.venv/bin/python src/evaluate_model.py --source base

echo "=== [3/4] Train BanglaT5 QA -- transfer (from summarization checkpoint) ==="
.venv/bin/python src/train.py --source summarization

echo "=== [4/4] Evaluate transfer ==="
.venv/bin/python src/evaluate_model.py --source summarization

echo "=== Comparison ==="
.venv/bin/python src/compare_results.py
