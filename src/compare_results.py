"""
Prints a baseline-vs-transfer comparison table from outputs/results/{baseline,transfer}.json.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("outputs/results")


def main():
    results = {}
    for condition in ["baseline", "transfer"]:
        path = RESULTS_DIR / f"{condition}.json"
        if not path.exists():
            print(f"Missing {path} -- run evaluate_model.py --source "
                  f"{'base' if condition == 'baseline' else 'summarization'} first")
            return
        with open(path, encoding="utf-8") as f:
            results[condition] = json.load(f)

    for split in ["validation", "test"]:
        print(f"\n=== {split} ===")
        print(f"{'metric':<15}{'baseline':>12}{'transfer':>12}{'delta':>12}")
        for metric in ["EM", "F1", "BERTScore-F1"]:
            b = results["baseline"][split][metric]
            t = results["transfer"][split][metric]
            print(f"{metric:<15}{b:>12.2f}{t:>12.2f}{t - b:>+12.2f}")


if __name__ == "__main__":
    main()
