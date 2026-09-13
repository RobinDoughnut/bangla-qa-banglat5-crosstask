"""
Downloads csebuetnlp/squad_bn from HuggingFace and saves it in SQuAD JSON format
under data/squad_bn/{train,validation,test}.json. Same dataset as bangla-qa-banglat5.

Also builds the D_S (summarization) split used by the `joint` multi-task condition,
from a local MultiBanAbs-style JSONL file (one {"Index", "Article", "Summary"} object
per line) expected at data/raw/MultiBanAbs_article_summary.json -- saved under
data/summary_bn/{train,validation,test}.json.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

OUT_DIR = Path("data/squad_bn")

SUMMARY_RAW_PATH = Path("data/raw/MultiBanAbs_article_summary.json")
SUMMARY_OUT_DIR = Path("data/summary_bn")
SUMMARY_VAL_SIZE = 1000
SUMMARY_TEST_SIZE = 1000
SUMMARY_SPLIT_SEED = 42


def to_squad_json(hf_split):
    by_title = defaultdict(lambda: defaultdict(list))
    for ex in hf_split:
        title = ex.get("title", "")
        context = ex["context"]
        by_title[title][context].append({
            "id": ex["id"],
            "question": ex["question"],
            "answers": [
                {"text": t, "answer_start": s}
                for t, s in zip(ex["answers"]["text"], ex["answers"]["answer_start"])
            ],
        })
    data = []
    for title, contexts in by_title.items():
        paragraphs = [{"context": ctx, "qas": qas} for ctx, qas in contexts.items()]
        data.append({"title": title, "paragraphs": paragraphs})
    return {"data": data}


def prepare_summary_data():
    if not SUMMARY_RAW_PATH.exists():
        print(f"\nWARNING: {SUMMARY_RAW_PATH} not found -- skipping D_S (summarization) prep. "
              f"Copy the MultiBanAbs JSONL file there to build data/summary_bn/.")
        return

    print(f"\nLoading D_S source from {SUMMARY_RAW_PATH}...")
    examples = []
    with open(SUMMARY_RAW_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append({
                "id": obj["Index"],
                "article": obj["Article"],
                "summary": obj["Summary"],
            })
    print(f"  Loaded {len(examples):,} article/summary pairs")

    rng = random.Random(SUMMARY_SPLIT_SEED)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    val = shuffled[:SUMMARY_VAL_SIZE]
    test = shuffled[SUMMARY_VAL_SIZE:SUMMARY_VAL_SIZE + SUMMARY_TEST_SIZE]
    train = shuffled[SUMMARY_VAL_SIZE + SUMMARY_TEST_SIZE:]

    SUMMARY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train", train), ("validation", val), ("test", test)]:
        out = SUMMARY_OUT_DIR / f"{split_name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(split_data):,} examples -> {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading csebuetnlp/squad_bn...")
    ds = load_dataset("csebuetnlp/squad_bn")
    print("Available splits:", list(ds.keys()))

    for split in ["train", "validation", "test"]:
        if split not in ds:
            print(f"  WARNING: split '{split}' not found, skipping")
            continue
        squad = to_squad_json(ds[split])
        out = OUT_DIR / f"{split}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(squad, f, ensure_ascii=False, indent=2)
        print(f"  {split}: {len(ds[split]):,} examples -> {out}")

    prepare_summary_data()
    print("\nDone.")


if __name__ == "__main__":
    main()
