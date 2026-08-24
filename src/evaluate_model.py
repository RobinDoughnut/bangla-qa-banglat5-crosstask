"""
Evaluates a fine-tuned QA model on validation and test splits.
Metrics: EM, F1, BERTScore-F1 -- identical to bangla-qa-banglat5.

Usage:
  python src/evaluate_model.py --source base
  python src/evaluate_model.py --source summarization
"""

import argparse
import json
import re
import string
import collections
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from tqdm import tqdm
from bert_score import score as bert_score_fn
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls_id, sep_id = self.cls_token_id, self.sep_token_id
        if token_ids_1 is None:
            return [cls_id] + token_ids_0 + [sep_id]
        return [cls_id] + token_ids_0 + [sep_id] + token_ids_1 + [sep_id]

    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs_with_special_tokens

DATA_DIR = Path("data/squad_bn")
RESULTS_DIR = Path("outputs/results")
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 16
NUM_BEAMS = 4
BERTSCORE_MODEL = "bert-base-multilingual-cased"


def load_squad_json(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    ids, questions, contexts, answers = [], [], [], []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if not qa.get("answers"):
                    continue
                ids.append(qa["id"])
                questions.append(qa["question"])
                contexts.append(context)
                answers.append({
                    "text": [a["text"] for a in qa["answers"]],
                    "answer_start": [a["answer_start"] for a in qa["answers"]],
                })
    return Dataset.from_dict({"id": ids, "question": questions, "context": contexts, "answers": answers})


def generate_predictions(model, tokenizer, dataset, device):
    model.eval()
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(dataset["question"], dataset["context"])
    ]
    predictions = []
    for i in tqdm(range(0, len(inputs), BATCH_SIZE), desc="generating"):
        batch = inputs[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=MAX_TARGET_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(decoded)
    return predictions


def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def exact_match(gold, pred):
    return int(normalize(gold) == normalize(pred))


def token_f1(gold, pred):
    g = normalize(gold).split()
    p = normalize(pred).split()
    common = collections.Counter(g) & collections.Counter(p)
    n = sum(common.values())
    if not g or not p:
        return int(g == p)
    if n == 0:
        return 0.0
    return 2 * (n / len(p)) * (n / len(g)) / ((n / len(p)) + (n / len(g)))


def compute_metrics(predictions, dataset):
    em_total = f1_total = 0.0
    examples = list(dataset)
    for ex, pred in zip(examples, predictions):
        golds = [a for a in ex["answers"]["text"] if normalize(a)]
        if not golds:
            golds = [""]
        em_total += max(exact_match(g, pred) for g in golds)
        f1_total += max(token_f1(g, pred) for g in golds)
    n = len(examples)
    return {"EM": 100 * em_total / n, "F1": 100 * f1_total / n, "N": n}


def compute_bertscore(predictions, dataset, device):
    refs = [ex["answers"]["text"] for ex in dataset]
    _, _, f1 = bert_score_fn(
        predictions,
        refs,
        model_type=BERTSCORE_MODEL,
        rescale_with_baseline=False,
        verbose=False,
        device=device,
    )
    return 100 * float(f1.mean())


def evaluate_split(split, tokenizer, model, device):
    print(f"\n=== {split} ===")
    ds = load_squad_json(DATA_DIR / f"{split}.json")
    preds = generate_predictions(model, tokenizer, ds, device)
    metrics = compute_metrics(preds, ds)
    metrics["BERTScore-F1"] = compute_bertscore(preds, ds, device)
    print(f"  EM: {metrics['EM']:.2f}  F1: {metrics['F1']:.2f}  BERTScore-F1: {metrics['BERTScore-F1']:.2f}  (N={metrics['N']:,})")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["base", "summarization"], required=True)
    args = parser.parse_args()
    condition = "baseline" if args.source == "base" else "transfer"
    model_path = Path(f"outputs/model/{condition}/best")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in ["validation", "test"]:
        metrics = evaluate_split(split, tokenizer, model, device)
        results[split] = metrics

    out_file = RESULTS_DIR / f"{condition}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
