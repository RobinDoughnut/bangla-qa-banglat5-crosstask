"""
Fine-tunes BanglaT5 for question answering (question + context -> answer) on squad_bn.

Four conditions, selected with --source:
  base           start from the original csebuetnlp/banglat5 checkpoint, QA only
  summarization  start from ../Bangla-T5-finetuned-summary (this model, already
                 fine-tuned on Bangla summarization) -- the sequential cross-task
                 transfer condition
  joint          start from the original csebuetnlp/banglat5 checkpoint and train on
                 D_S (summarization, MultiBanAbs) and D_Q (QA, squad_bn) simultaneously,
                 a single model distinguishing the two tasks purely via input prefix
                 ("summarize: ..." vs "question: ... context: ...") -- the multi-task
                 joint-learning baseline
  base_untied    ablation: same as base (original pretrained checkpoint, D_Q only,
                 no cross-task exposure), but with tie_word_embeddings=False set
                 before training so the model has the same untied-embedding parameter
                 count as `summarization` (296.9M vs 247.6M). Isolates whether the
                 `summarization` condition's gain over `base` comes from cross-task
                 transfer or merely from having more parameters.
  joint_untied   same as joint (D_S + D_Q simultaneously, original pretrained
                 checkpoint) but with tie_word_embeddings=False set before training,
                 matching `joint`'s 247.6M-param model to `base_untied`/`transfer`'s
                 296.9M. Isolates whether joint multi-task training's QA regression
                 (vs. baseline) is affected by embedding capacity, the same question
                 base_untied asks of the sequential-transfer condition.

Same task/format/hyperparameters as bangla-qa-banglat5 and the base/summarization
conditions above; the joint and joint_untied conditions additionally mix in D_S.
"""

import argparse
import json
import time
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from datasets import Dataset, concatenate_datasets

BASE_MODEL = "csebuetnlp/banglat5"
SUMMARIZATION_CHECKPOINT = Path("../Bangla-T5-finetuned-summary")

DATA_DIR = Path("data/squad_bn")
SUMMARY_DATA_DIR = Path("data/summary_bn")
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
SUMMARY_MAX_TARGET_LENGTH = 128  # summaries (p90 ~47 words) run longer than QA answers
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # effective batch size stays 16
EPOCHS = 3
LEARNING_RATE = 3e-5


class EpochTimer(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self._t0
        epoch = int(state.epoch)
        print(f"\n  Epoch {epoch} time: {elapsed/60:.1f} min ({elapsed:.0f} s)")


def print_model_stats(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    print(f"  Trainable params : {trainable:,}")
    print(f"  Total params     : {total:,}")
    print(f"  Model size (fp32): {size_mb:.1f} MB")


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


def tokenize(examples, tokenizer):
    # QA input format, identical to bangla-qa-banglat5: "question: <Q> context: <C>"
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(examples["question"], examples["context"])
    ]
    targets = [
        a["text"][0] if a["text"] else ""
        for a in examples["answers"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )
    labels["input_ids"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_summary_json(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return Dataset.from_dict({
        "id": [ex["id"] for ex in raw],
        "article": [ex["article"] for ex in raw],
        "summary": [ex["summary"] for ex in raw],
    })


def tokenize_summary(examples, tokenizer):
    # D_S input format: task-specific prefix distinguishing it from the QA task above.
    inputs = [f"summarize: {a}" for a in examples["article"]]
    targets = examples["summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=SUMMARY_MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )
    labels["input_ids"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["base", "summarization", "joint", "base_untied", "joint_untied"], required=True,
                         help="base = fine-tune from csebuetnlp/banglat5 on QA only; "
                              "summarization = continue fine-tuning from the summarization checkpoint (sequential transfer condition); "
                              "joint = fine-tune from csebuetnlp/banglat5 on D_S + D_Q simultaneously (multi-task joint-learning baseline); "
                              "base_untied = same as base but with tie_word_embeddings=False (parameter-count ablation vs. summarization); "
                              "joint_untied = same as joint but with tie_word_embeddings=False (parameter-count ablation of joint)")
    args = parser.parse_args()

    if args.source == "base":
        model_name = BASE_MODEL
        output_dir = Path("outputs/model/baseline")
    elif args.source == "summarization":
        model_name = str(SUMMARIZATION_CHECKPOINT)
        output_dir = Path("outputs/model/transfer")
    elif args.source == "base_untied":
        model_name = BASE_MODEL
        output_dir = Path("outputs/model/baseline_untied")
    elif args.source == "joint_untied":
        model_name = BASE_MODEL
        output_dir = Path("outputs/model/joint_untied")
    else:
        model_name = BASE_MODEL
        output_dir = Path("outputs/model/joint")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "best"
    if (save_path / "config.json").exists():
        print(f"{save_path} already contains a finished model -- skipping (delete it to retrain).")
        return

    checkpoint_dir = output_dir / "checkpoints"
    existing_checkpoints = sorted(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
    resume_from_checkpoint = bool(existing_checkpoints)

    print(f"Condition: {args.source}")
    print(f"Loading base weights from: {model_name}")
    if resume_from_checkpoint:
        print(f"Found {len(existing_checkpoints)} existing checkpoint(s) -- resuming from the latest.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.source in ("base_untied", "joint_untied"):
        # tie_word_embeddings must be passed at load time, not set on model.config
        # after from_pretrained() returns: this checkpoint's on-disk shared.weight and
        # lm_head.weight already differ, so recent transformers versions refuse to
        # (re-)tie them at load time regardless of the config value found afterwards --
        # setting the flag post-hoc is a no-op here and silently reproduces `base`
        # instead of the parameter-matched ablation. Passing it into from_pretrained
        # forces genuinely independent encoder/decoder/lm_head embedding matrices,
        # matching the `summarization` checkpoint's 296.9M-parameter untied structure.
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, tie_word_embeddings=False)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("Model stats:")
    print_model_stats(model)

    print("Loading data...")
    train_ds = load_squad_json(DATA_DIR / "train.json")
    val_ds = load_squad_json(DATA_DIR / "validation.json")
    print(f"  D_Q train: {len(train_ds):,}  D_Q val: {len(val_ds):,}")

    print("Tokenizing...")
    train_features = train_ds.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="train (D_Q)",
    )
    val_features = val_ds.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
        desc="val (D_Q)",
    )

    if args.source in ("joint", "joint_untied"):
        summary_train_ds = load_summary_json(SUMMARY_DATA_DIR / "train.json")
        summary_val_ds = load_summary_json(SUMMARY_DATA_DIR / "validation.json")
        print(f"  D_S train: {len(summary_train_ds):,}  D_S val: {len(summary_val_ds):,}")

        summary_train_features = summary_train_ds.map(
            lambda ex: tokenize_summary(ex, tokenizer),
            batched=True,
            remove_columns=summary_train_ds.column_names,
            desc="train (D_S)",
        )
        summary_val_features = summary_val_ds.map(
            lambda ex: tokenize_summary(ex, tokenizer),
            batched=True,
            remove_columns=summary_val_ds.column_names,
            desc="val (D_S)",
        )

        # Concatenate D_Q + D_S into one mixture; the Trainer's default random sampler
        # shuffles every epoch, so tasks are interleaved throughout training, distinguished
        # only by their input prefix ("question: ... context: ..." vs "summarize: ...").
        train_features = concatenate_datasets([train_features, summary_train_features])
        val_features = concatenate_datasets([val_features, summary_val_features])
        print(f"  {args.source} train (D_Q + D_S): {len(train_features):,}  {args.source} val: {len(val_features):,}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        optim="adafactor",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=SUMMARY_MAX_TARGET_LENGTH if args.source in ("joint", "joint_untied") else MAX_TARGET_LENGTH,
        bf16=use_bf16,
        report_to="none",
        logging_steps=200,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=val_features,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EpochTimer()],
    )

    print("Training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
