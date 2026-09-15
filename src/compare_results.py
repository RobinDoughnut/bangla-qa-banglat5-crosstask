"""
Prints a baseline-vs-transfer-vs-joint-vs-baseline_untied-vs-joint_untied comparison
table from outputs/results/{baseline,transfer,joint,baseline_untied,joint_untied}.json.
Everything but baseline/transfer is optional -- the table adapts to whichever result
files exist.

`baseline_untied` is the parameter-count ablation of `transfer`: same init/data as
`baseline` (no cross-task exposure) but with tie_word_embeddings=False, matching
`transfer`'s 296.9M-parameter (untied-embedding) size. Reading its delta vs. baseline
separates the `transfer` gain into a parameter-count component and a
cross-task-transfer component:
  baseline_untied ~= baseline   -> transfer's gain is real cross-task transfer
  baseline_untied ~= transfer   -> transfer's gain was mostly extra parameters

`joint_untied` is the same ablation applied to `joint`: same D_S+D_Q multi-task
training as `joint` but with tie_word_embeddings=False, matching `joint`'s 247.6M-param
model to the other conditions' 296.9M. Its delta vs. `joint` isolates whether embedding
capacity (rather than the multi-task objective itself) explains joint's QA regression.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("outputs/results")
SOURCE_FLAG = {"baseline": "base", "transfer": "summarization", "joint": "joint",
               "baseline_untied": "base_untied", "joint_untied": "joint_untied"}
REQUIRED = ["baseline", "transfer"]
OPTIONAL = ["baseline_untied", "joint", "joint_untied"]


def main():
    results = {}
    for condition in REQUIRED:
        path = RESULTS_DIR / f"{condition}.json"
        if not path.exists():
            print(f"Missing {path} -- run evaluate_model.py --source {SOURCE_FLAG[condition]} first")
            return
        with open(path, encoding="utf-8") as f:
            results[condition] = json.load(f)

    present_optional = []
    for condition in OPTIONAL:
        path = RESULTS_DIR / f"{condition}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                results[condition] = json.load(f)
            present_optional.append(condition)

    # Fixed display order regardless of which optional conditions are present.
    conditions = ["baseline", "baseline_untied", "transfer", "joint", "joint_untied"]
    conditions = [c for c in conditions if c in results]

    for split in ["validation", "test"]:
        print(f"\n=== {split} (QA, D_Q) ===")
        header = f"{'metric':<15}" + "".join(f"{c:>16}" for c in conditions)
        header += f"{'d(vs baseline)':>16}" * (len(conditions) - 1)
        print(header)
        for metric in ["EM", "F1", "BERTScore-F1"]:
            row = f"{metric:<15}" + "".join(f"{results[c][split][metric]:>16.2f}" for c in conditions)
            for c in conditions[1:]:
                row += f"{results[c][split][metric] - results['baseline'][split][metric]:>+16.2f}"
            print(row)

    if "baseline_untied" in results:
        for split in ["validation", "test"]:
            print(f"\n=== {split}: parameter-count vs. cross-task-transfer decomposition ===")
            for metric in ["EM", "F1", "BERTScore-F1"]:
                total = results["transfer"][split][metric] - results["baseline"][split][metric]
                param_effect = results["baseline_untied"][split][metric] - results["baseline"][split][metric]
                transfer_effect = results["transfer"][split][metric] - results["baseline_untied"][split][metric]
                print(f"  {metric:<13} total(transfer-baseline)={total:+.2f}  "
                      f"parameter-count={param_effect:+.2f}  cross-task-transfer={transfer_effect:+.2f}")

    if "joint" in results and "joint_untied" in results:
        for split in ["validation", "test"]:
            print(f"\n=== {split}: joint embedding-capacity ablation ===")
            for metric in ["EM", "F1", "BERTScore-F1"]:
                delta = results["joint_untied"][split][metric] - results["joint"][split][metric]
                print(f"  {metric:<13} joint={results['joint'][split][metric]:.2f}  "
                      f"joint_untied={results['joint_untied'][split][metric]:.2f}  delta={delta:+.2f}")

    for condition in ["joint", "joint_untied"]:
        if condition in results and "summary_bn" in results[condition]:
            for split in ["validation", "test"]:
                print(f"\n=== {split} (Summarization, D_S -- {condition} condition) ===")
                metrics = results[condition]["summary_bn"][split]
                print(f"  ROUGE-1: {metrics['rouge1']:.2f}  ROUGE-2: {metrics['rouge2']:.2f}  "
                      f"ROUGE-L: {metrics['rougeL']:.2f}  (N={metrics['N']:,})")


if __name__ == "__main__":
    main()
