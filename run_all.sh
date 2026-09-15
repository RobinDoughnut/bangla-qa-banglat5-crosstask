#!/usr/bin/env bash
# Runs all five conditions (baseline, transfer, joint, baseline_untied, joint_untied)
# for BanglaT5, then evaluates them.
# Safe to re-run after an interruption: train.py skips any condition whose
# outputs/model/<condition>/best already exists, and resumes mid-condition
# from the latest epoch checkpoint if one exists.
set -e
cd "$(dirname "$0")"

echo "=== [1/10] Train BanglaT5 QA -- baseline ==="
.venv/bin/python src/train.py --source base

echo "=== [2/10] Evaluate baseline ==="
.venv/bin/python src/evaluate_model.py --source base

echo "=== [3/10] Train BanglaT5 QA -- transfer (from summarization checkpoint) ==="
.venv/bin/python src/train.py --source summarization

echo "=== [4/10] Evaluate transfer ==="
.venv/bin/python src/evaluate_model.py --source summarization

echo "=== [5/10] Train BanglaT5 -- joint (multi-task, D_S + D_Q simultaneously) ==="
.venv/bin/python src/train.py --source joint

echo "=== [6/10] Evaluate joint ==="
.venv/bin/python src/evaluate_model.py --source joint

echo "=== [7/10] Train BanglaT5 QA -- baseline_untied (ablation: untied embeddings, no cross-task exposure) ==="
.venv/bin/python src/train.py --source base_untied

echo "=== [8/10] Evaluate baseline_untied ==="
.venv/bin/python src/evaluate_model.py --source base_untied

echo "=== [9/10] Train BanglaT5 -- joint_untied (ablation: untied embeddings, D_S + D_Q simultaneously) ==="
.venv/bin/python src/train.py --source joint_untied

echo "=== [10/10] Evaluate joint_untied ==="
.venv/bin/python src/evaluate_model.py --source joint_untied

echo "=== Comparison ==="
.venv/bin/python src/compare_results.py
