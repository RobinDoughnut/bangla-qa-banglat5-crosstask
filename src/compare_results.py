"""
Prints a baseline-vs-transfer-vs-joint comparison table from
outputs/results/{baseline,transfer,joint}.json. `joint` is optional -- the table
still prints baseline-vs-transfer if it's missing.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("outputs/results")
SOURCE_FLAG = {"baseline": "base", "transfer": "summarization", "joint": "joint"}


def main():
    results = {}
    for condition in ["baseline", "transfer"]:
        path = RESULTS_DIR / f"{condition}.json"
        if not path.exists():
            print(f"Missing {path} -- run evaluate_model.py --source {SOURCE_FLAG[condition]} first")
            return
        with open(path, encoding="utf-8") as f:
            results[condition] = json.load(f)

    joint_path = RESULTS_DIR / "joint.json"
    have_joint = joint_path.exists()
    if have_joint:
        with open(joint_path, encoding="utf-8") as f:
            results["joint"] = json.load(f)

    conditions = ["baseline", "transfer", "joint"] if have_joint else ["baseline", "transfer"]

    for split in ["validation", "test"]:
        print(f"\n=== {split} (QA, D_Q) ===")
        header = f"{'metric':<15}" + "".join(f"{c:>12}" for c in conditions)
        if not have_joint:
            header += f"{'delta':>12}"
        print(header)
        for metric in ["EM", "F1", "BERTScore-F1"]:
            row = f"{metric:<15}" + "".join(f"{results[c][split][metric]:>12.2f}" for c in conditions)
            if not have_joint:
                row += f"{results['transfer'][split][metric] - results['baseline'][split][metric]:>+12.2f}"
            print(row)

    if have_joint and "summary_bn" in results["joint"]:
        for split in ["validation", "test"]:
            print(f"\n=== {split} (Summarization, D_S -- joint condition only) ===")
            metrics = results["joint"]["summary_bn"][split]
            print(f"  ROUGE-1: {metrics['rouge1']:.2f}  ROUGE-2: {metrics['rouge2']:.2f}  "
                  f"ROUGE-L: {metrics['rougeL']:.2f}  (N={metrics['N']:,})")


if __name__ == "__main__":
    main()
