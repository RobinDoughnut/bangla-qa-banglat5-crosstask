"""
Downloads csebuetnlp/squad_bn from HuggingFace and saves it in SQuAD JSON format
under data/squad_bn/{train,validation,test}.json. Same dataset as bangla-qa-banglat5.
"""

import json
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

OUT_DIR = Path("data/squad_bn")


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

    print("Done.")


if __name__ == "__main__":
    main()
