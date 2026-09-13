"""
Evaluates a fine-tuned QA model on validation and test splits.
Metrics: EM, F1, BERTScore-F1 -- identical to bangla-qa-banglat5.

For --source joint, also evaluates the same checkpoint on D_S (summarization,
data/summary_bn) with ROUGE-1/2/L, since that condition was trained on both tasks.

Usage:
  python src/evaluate_model.py --source base
  python src/evaluate_model.py --source summarization
  python src/evaluate_model.py --source joint
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
from rouge_score import rouge_scorer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls_id, sep_id = self.cls_token_id, self.sep_token_id
        if token_ids_1 is None:
            return [cls_id] + token_ids_0 + [sep_id]
        return [cls_id] + token_ids_0 + [sep_id] + token_ids_1 + [sep_id]

    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs_with_special_tokens

DATA_DIR = Path("data/squad_bn")
SUMMARY_DATA_DIR = Path("data/summary_bn")
RESULTS_DIR = Path("outputs/results")
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
SUMMARY_MAX_TARGET_LENGTH = 128
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


def generate_from_inputs(model, tokenizer, inputs, device, max_new_tokens):
    model.eval()
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
                max_new_tokens=max_new_tokens,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(decoded)
    return predictions


def generate_predictions(model, tokenizer, dataset, device):
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(dataset["question"], dataset["context"])
    ]
    return generate_from_inputs(model, tokenizer, inputs, device, MAX_TARGET_LENGTH)


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


def load_summary_json(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return Dataset.from_dict({
        "id": [ex["id"] for ex in raw],
        "article": [ex["article"] for ex in raw],
        "summary": [ex["summary"] for ex in raw],
    })


class _WhitespaceTokenizer:
    """rouge_score's default tokenizer only matches ASCII [a-z0-9]+, silently
    dropping all Bangla text (0 tokens -> every score 0.0). Split on whitespace
    instead, consistent with how the rest of this repo treats Bangla text."""
    def tokenize(self, text):
        return text.split()


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=_WhitespaceTokenizer()
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = len(predictions)
    return {key: 100 * val / n for key, val in totals.items()}


def evaluate_summary_split(split, tokenizer, model, device):
    print(f"\n=== D_S {split} ===")
    ds = load_summary_json(SUMMARY_DATA_DIR / f"{split}.json")
    inputs = [f"summarize: {a}" for a in ds["article"]]
    preds = generate_from_inputs(model, tokenizer, inputs, device, SUMMARY_MAX_TARGET_LENGTH)
    metrics = compute_rouge(preds, ds["summary"])
    metrics["N"] = len(ds)
    print(f"  ROUGE-1: {metrics['rouge1']:.2f}  ROUGE-2: {metrics['rouge2']:.2f}  "
          f"ROUGE-L: {metrics['rougeL']:.2f}  (N={metrics['N']:,})")
    return metrics


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
    parser.add_argument("--source", choices=["base", "summarization", "joint"], required=True)
    args = parser.parse_args()
    condition = {"base": "baseline", "summarization": "transfer", "joint": "joint"}[args.source]
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

    if args.source == "joint":
        results["summary_bn"] = {}
        for split in ["validation", "test"]:
            results["summary_bn"][split] = evaluate_summary_split(split, tokenizer, model, device)

    out_file = RESULTS_DIR / f"{condition}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
