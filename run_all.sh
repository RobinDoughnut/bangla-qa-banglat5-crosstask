#!/usr/bin/env bash
# Runs all three conditions (baseline, transfer, joint) for BanglaT5, then evaluates them.
# Safe to re-run after an interruption: train.py skips any condition whose
# outputs/model/<condition>/best already exists, and resumes mid-condition
# from the latest epoch checkpoint if one exists.
set -e
cd "$(dirname "$0")"

echo "=== [1/6] Train BanglaT5 QA -- baseline ==="
.venv/bin/python src/train.py --source base

echo "=== [2/6] Evaluate baseline ==="
.venv/bin/python src/evaluate_model.py --source base

echo "=== [3/6] Train BanglaT5 QA -- transfer (from summarization checkpoint) ==="
.venv/bin/python src/train.py --source summarization

echo "=== [4/6] Evaluate transfer ==="
.venv/bin/python src/evaluate_model.py --source summarization

echo "=== [5/6] Train BanglaT5 -- joint (multi-task, D_S + D_Q simultaneously) ==="
.venv/bin/python src/train.py --source joint

echo "=== [6/6] Evaluate joint ==="
.venv/bin/python src/evaluate_model.py --source joint

echo "=== Comparison ==="
.venv/bin/python src/compare_results.py
