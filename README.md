# Bangla Question Answering with BanglaT5 -- Cross-Task Transfer

Fine-tunes [`csebuetnlp/banglat5`](https://huggingface.co/csebuetnlp/banglat5) for extractive
question answering (question + context -> answer) on Bangla SQuAD
([csebuetnlp/squad_bn](https://huggingface.co/datasets/csebuetnlp/squad_bn)), under two
initialization conditions, to test whether prior fine-tuning on a different task (summarization)
transfers to QA.

## Research question

Does fine-tuning on Bangla summarization first help a model learn question answering faster /
better than fine-tuning on QA directly? Two conditions, identical data and hyperparameters,
differing only in the starting checkpoint:

| Condition | Init checkpoint | What it tests |
|---|---|---|
| **baseline** | `csebuetnlp/banglat5` (original pretrained) | QA fine-tuned from scratch |
| **transfer** | `../Bangla-T5-finetuned-summary` (this model, already fine-tuned on Bangla summarization) | QA fine-tuned on top of a *different* downstream task |

## Related Experiments

| Repo | Model | Condition |
|------|-------|-----------|
| **bangla-qa-banglat5-crosstask** (this repo) | BanglaT5 | QA baseline + transfer-from-summarization |
| [bangla-qa-mt5-crosstask](../bangla-qa-mt5-crosstask) | mT5-small | Same design, mT5 backbone |
| [bangla-qa-banglat5](../bangla-qa-banglat5) | BanglaT5 | Original QA repo, pretrained-only, this repo's "baseline" condition reproduces it |
| [bangla-qg-banglat5](../bangla-qg-banglat5) | BanglaT5 | Same baseline/transfer design, applied to question generation instead |

This repo exists because the original request (fine-tune the summarization checkpoints on a
downstream task, and measure cross-task transfer) was initially misread as "question generation."
[bangla-qg-banglat5](../bangla-qg-banglat5)/[bangla-qg-mt5](../bangla-qg-mt5) cover that (valid,
separate) experiment; this repo and its mT5 counterpart are the actually-requested QA version.

## Task format

Identical to [bangla-qa-banglat5](../bangla-qa-banglat5): given a question and a context passage,
generate the answer as text (seq2seq generation, not span-index prediction).

```
input : question: <Q> context: <C>
target: <answer text>
```

## Dataset

Same as the other `bangla-qa-*` repos: `csebuetnlp/squad_bn`, unanswerable questions excluded.

| Split | Examples |
|-------|----------|
| Train | 68,674 |
| Validation | 1,251 |
| Test | 1,252 |

## Data Preprocessing & Text Representation

Documented here for citation purposes.

**Preprocessing pipeline** (`src/prepare_data.py`, `src/train.py:load_squad_json`):

1. **Source**: `csebuetnlp/squad_bn` (Bangla SQuAD), loaded via the HuggingFace `datasets` library.
2. **Re-grouping**: the raw dataset is flat (one row per question-context pair, with the context
   string repeated for every question that shares it). Rows are re-grouped into SQuAD's canonical
   `{title: [{context, qas: [...]}]}` nested structure so each unique context appears once.
3. **Unanswerable-question filtering**: rows with an empty `answers` list are dropped. This is what
   shrinks the raw split sizes -- 118,117 / 2,502 / 2,504 train/validation/test in the current
   dataset snapshot -- down to 68,674 / 1,251 / 1,252, the sizes reported under Dataset above.
4. **Target answer selection**: `squad_bn` provides multiple human-annotated gold answer spans for
   some questions. Training uses only the first (`answers.text[0]`) as the target sequence `y`; at
   evaluation time all gold answers are retained, and EM/F1/BERTScore are computed against every
   gold answer with the maximum taken per example (standard SQuAD protocol).
5. **Prompt construction** (text-to-text framing, no task-specific output head):
   `"question: {question} context: {context}"` -> target: answer text.
6. **No text normalization** (lowercasing, punctuation stripping, diacritic folding) is applied
   before tokenization -- raw Bangla UTF-8 text is passed directly into the model's subword
   tokenizer.
7. **Truncation/padding**: input sequences truncated to 512 subword tokens, targets to 64; batches
   are dynamically padded per-batch (`DataCollatorForSeq2Seq`), and label padding positions are
   masked with `-100` so they don't contribute to the cross-entropy loss.

**Text embedding**:

- Tokenizer: SentencePiece unigram-model subword tokenizer (`T5Tokenizer`), operating directly on
  raw text -- distinct from the whitespace-then-WordPiece pipeline BERT-family models use, which
  matters for Bangla since it doesn't reliably space-delimit at the "word" level the way English
  does.
- Vocabulary: 32,128 subword tokens, trained on a Bangla-only corpus.
- Embedding dimension (`d_model`): 768, across 12 encoder + 12 decoder layers.
- A single learned embedding matrix (`vocab_size x d_model`) maps token ids to dense vectors. In
  the base pretrained checkpoint (`tie_word_embeddings=True`) this matrix is **tied** across the
  encoder input embedding, decoder input embedding, and output (LM head) projection. As with the
  sibling QG repo, the summarization checkpoint used for the transfer condition was saved with
  `tie_word_embeddings=False`, giving it separate matrices for each of the three roles -- this
  accounts for the transfer condition's larger parameter count below, not a difference in
  architecture depth/width.
- Positional information is **not** injected via absolute or sinusoidal position embeddings added
  to the token embedding (unlike BERT-family models). Instead, T5-family models add a learned
  **relative position bias** directly to the attention logits at every layer, bucketed by relative
  token distance (`relative_attention_num_buckets=32`, `relative_attention_max_distance=128`).

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended
- `sentencepiece` for the BanglaT5 tokenizer

## Setup

```bash
cd bangla-qa-banglat5-crosstask
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data is already included under `data/squad_bn/`. To regenerate from scratch:
`python src/prepare_data.py`.

This repo also expects `../Bangla-T5-finetuned-summary` to exist (a standard HF `save_pretrained`
checkpoint directory) for the transfer condition.

## Training

```bash
python src/train.py --source base            # baseline condition
python src/train.py --source summarization   # transfer condition
```

Saves to `outputs/model/{baseline,transfer}/best`. Safe to interrupt and re-run: a condition whose
`best/` checkpoint already exists is skipped, and an in-progress condition resumes from its latest
epoch checkpoint rather than restarting.

**Hyperparameters** (identical between conditions, and identical to `bangla-qa-banglat5` -- the
only variable is the init checkpoint):

| Parameter | Value |
|-----------|-------|
| Max input length | 512 tokens |
| Max target length | 64 tokens |
| Batch size (per device) | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Epochs | 3 |
| Learning rate | 3e-5 |
| Optimizer | Adafactor |
| Mixed precision | bf16 |

## Evaluation

```bash
python src/evaluate_model.py --source base
python src/evaluate_model.py --source summarization
python src/compare_results.py
```

**Metrics:** EM, F1, and BERTScore-F1 (`bert-base-multilingual-cased`) -- identical to
`bangla-qa-banglat5`.

## Results

**Model stats:**

| | baseline | transfer |
|---|---|---|
| Init checkpoint | `csebuetnlp/banglat5` | `Bangla-T5-finetuned-summary` |
| Total params | 247,577,856 | 296,926,464 |
| Model size (fp32) | 944.4 MB | 1132.7 MB |

**Training** (RTX 4070, effective batch size 16):

| Epoch | baseline eval_loss | baseline time | transfer eval_loss | transfer time |
|-------|---------------------|----------------|----------------------|-----------------|
| 1 | 0.9947 | 30.8 min | 0.6900 | 30.8 min |
| 2 | 0.8127 | 30.8 min | 0.6681 | 30.9 min |
| 3 | 0.7953 | 30.8 min | 0.6685 | 30.6 min |

The baseline condition's loss here (epoch 3: 0.7953) is close to, but not identical to, the
original `bangla-qa-banglat5` repo's own run (epoch 3: 0.8003) -- expected run-to-run variance from
a different random seed / data shuffle, not a methodological difference.

**Evaluation:**

| Split | Condition | EM | F1 | BERTScore-F1 | N |
|-------|-----------|-----|-----|--------------|-----|
| Validation | baseline | 54.68 | 68.35 | 91.17 | 1,251 |
| Validation | transfer | 57.07 | 70.24 | 91.76 | 1,251 |
| Test | baseline | 53.27 | 68.05 | 91.16 | 1,252 |
| Test | transfer | 54.55 | 69.34 | 91.60 | 1,252 |

| Test metric | baseline | transfer | delta |
|---|---|---|---|
| EM | 53.27 | 54.55 | **+1.28** |
| F1 | 68.05 | 69.34 | **+1.29** |
| BERTScore-F1 | 91.16 | 91.60 | **+0.44** |

**Finding:** prior summarization fine-tuning transfers positively to question answering for
BanglaT5 -- every metric improves, on both splits. This matches the direction of the finding in
[bangla-qg-banglat5](../bangla-qg-banglat5) (summarization -> QG also transferred positively),
reinforcing that BanglaT5's summarization fine-tuning produced some general-purpose gain (likely in
fluent, contextually-grounded Bangla generation) rather than something narrowly specific to
summarization's own input/output shape.

## Limitations

**Single run, no variance estimate.** One seed, one training run per condition. The +1.28 EM /
+1.29 F1 gap is a point estimate; on a 1,252-example test set this corresponds to roughly 16
questions, noticeably larger than the kind of single-question noise the original
`bangla-qa-banglat5` repo flagged when comparing against mBERT (0.95 EM, ~12 questions), but still
not backed by a multi-seed variance estimate.

**The two init checkpoints are not perfectly matched.** The summarization checkpoint has untied
embeddings (see Results); its own training data, epoch budget, and hyperparameters (from when it
was originally fine-tuned for summarization) are outside this repo's control and not documented
here.

**Fixed 3-epoch budget**, held constant between conditions to isolate the init-checkpoint variable.
The transfer condition's eval loss is nearly flat after epoch 2 (0.6681 -> 0.6685), suggesting it
converges faster than 3 epochs actually requires, while the baseline was still improving at epoch 3.

**Dataset caveat.** Shared with the sibling `bangla-qa-*` repos: a subset of `squad_bn`'s
`answer_start` offsets are off by 1-5 characters in validation/test. Irrelevant here since this
repo (like all the seq2seq QA repos) trains on answer *text*, never on `answer_start` offsets.

## Project Structure

```
bangla-qa-banglat5-crosstask/
├── data/
│   └── squad_bn/                    # train/validation/test.json
├── outputs/
│   ├── model/
│   │   ├── baseline/{checkpoints,best}
│   │   └── transfer/{checkpoints,best}
│   ├── results/
│   │   └── baseline.json / transfer.json
│   └── logs/
├── src/
│   ├── prepare_data.py
│   ├── train.py              # --source {base,summarization}
│   ├── evaluate_model.py     # --source {base,summarization}
│   └── compare_results.py
├── run_all.sh                # trains + evaluates both conditions end to end
├── requirements.txt
└── README.md
```

## Reproducing Results

```bash
cd bangla-qa-banglat5-crosstask
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```
