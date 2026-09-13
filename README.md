# Bangla Question Answering with BanglaT5 -- Cross-Task Transfer

Fine-tunes [`csebuetnlp/banglat5`](https://huggingface.co/csebuetnlp/banglat5) for extractive
question answering (question + context -> answer) on Bangla SQuAD
([csebuetnlp/squad_bn](https://huggingface.co/datasets/csebuetnlp/squad_bn)), under three
conditions, to test whether -- and how -- cross-task exposure to a different task (Bangla
summarization) interacts with QA: not at all (baseline), sequentially (transfer), or
simultaneously (joint).

## Research question

Does exposure to Bangla summarization help a model learn question answering, and does *how* that
exposure happens (before QA training vs. simultaneously with it) matter? Three conditions, identical
QA data/hyperparameters, differing in whether/how summarization data (D_S) enters training:

| Condition | Init checkpoint | Training data | What it tests |
|---|---|---|---|
| **baseline** | `csebuetnlp/banglat5` (original pretrained) | D_Q only | QA fine-tuned from scratch |
| **transfer** | `../Bangla-T5-finetuned-summary` (already fine-tuned on Bangla summarization) | D_Q only | QA fine-tuned on top of a *different*, already-completed downstream task (sequential transfer) |
| **joint** | `csebuetnlp/banglat5` (original pretrained) | D_S + D_Q simultaneously | Multi-task joint learning: a single model trained on both tasks at once, distinguished only by input prefix (`"question: ..."` vs `"summarize: ..."`) |

D_S is [MultiBanAbs](https://arxiv.org/abs/2511.19317) (A Comprehensive Multi-Domain Bangla
Abstractive Text Summarization Dataset), supplied locally (54,620 examples; not from HuggingFace --
see [D_S: summarization data](#d_s-summarization-data) below), D_Q is `csebuetnlp/squad_bn` as in
the other conditions.

## Related Experiments

| Repo | Model | Condition |
|------|-------|-----------|
| **bangla-qa-banglat5-crosstask** (this repo) | BanglaT5 | QA baseline + transfer-from-summarization + multi-task joint |
| [bangla-qa-mt5-crosstask](../bangla-qa-mt5-crosstask) | mT5-small | Same design, mT5 backbone |
| [bangla-qa-banglat5](../bangla-qa-banglat5) | BanglaT5 | Original QA repo, pretrained-only, this repo's "baseline" condition reproduces it |
| [bangla-qg-banglat5](../bangla-qg-banglat5) | BanglaT5 | Same baseline/transfer design, applied to question generation instead |

This repo exists because the original request (fine-tune the summarization checkpoints on a
downstream task, and measure cross-task transfer) was initially misread as "question generation."
[bangla-qg-banglat5](../bangla-qg-banglat5)/[bangla-qg-mt5](../bangla-qg-mt5) cover that (valid,
separate) experiment; this repo and its mT5 counterpart are the actually-requested QA version.

## Task format

QA (D_Q), identical to [bangla-qa-banglat5](../bangla-qa-banglat5): given a question and a context
passage, generate the answer as text (seq2seq generation, not span-index prediction).

```
input : question: <Q> context: <C>
target: <answer text>
```

Summarization (D_S, `joint` condition only): given a news article, generate its summary. The
`"summarize: "` prefix is what lets a single shared model tell the two tasks apart during joint
training -- there is no task-specific output head or architecture change, purely a difference in
input framing.

```
input : summarize: <article>
target: <summary text>
```

## Dataset

D_Q is the same as the other `bangla-qa-*` repos: `csebuetnlp/squad_bn`, unanswerable questions
excluded.

| Split | Examples |
|-------|----------|
| Train | 68,674 |
| Validation | 1,251 |
| Test | 1,252 |

### D_S: summarization data

Used only by the `joint` condition. [MultiBanAbs](https://arxiv.org/abs/2511.19317), a multi-domain
Bangla abstractive summarization corpus (The Business Standard, Samakal, and the Cinegolpo blog),
supplied locally (not via HuggingFace `datasets`) as JSONL, one `{"Index", "Article", "Summary"}`
object per line, at `data/raw/MultiBanAbs_article_summary.json`, and split by
`src/prepare_data.py` into `data/summary_bn/{train,validation,test}.json`:

| Split | Examples |
|-------|----------|
| Train | 52,620 |
| Validation | 1,000 |
| Test | 1,000 |

The split is a random 1,000/1,000 validation/test holdout (fixed seed 42) from the full 54,620
examples, with the remainder as train. D_S's train size (52,620) is deliberately close to D_Q's
(68,674, ratio 0.77) so that concatenating both in full for joint training already gives
roughly-balanced per-task exposure without needing to up/downsample either side.

Article/summary length (words, not subword tokens): articles average 262 words (p90 474, max
1,052); summaries average 30 words (p90 47, max 63). Summaries routinely exceed QA's 64-token target
length, so the `joint` condition uses a separate, longer target cap for D_S (128 tokens) -- see
Training below.

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
`python src/prepare_data.py`. The same command also builds `data/summary_bn/` (D_S) if
`data/raw/MultiBanAbs_article_summary.json` is present -- copy your local MultiBanAbs file there
first.

This repo also expects `../Bangla-T5-finetuned-summary` to exist (a standard HF `save_pretrained`
checkpoint directory) for the transfer condition.

## Training

```bash
python src/train.py --source base            # baseline condition
python src/train.py --source summarization   # transfer condition
python src/train.py --source joint            # joint (multi-task) condition
```

Saves to `outputs/model/{baseline,transfer,joint}/best`. Safe to interrupt and re-run: a condition
whose `best/` checkpoint already exists is skipped, and an in-progress condition resumes from its
latest epoch checkpoint rather than restarting.

**Hyperparameters** (identical across all three conditions, and identical to
`bangla-qa-banglat5` -- baseline vs. transfer varies only the init checkpoint; joint additionally
mixes in D_S):

| Parameter | Value |
|-----------|-------|
| Max input length | 512 tokens |
| Max target length | 64 tokens (QA); 128 tokens (D_S targets, `joint` only) |
| Batch size (per device) | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Epochs | 3 |
| Learning rate | 3e-5 |
| Optimizer | Adafactor |
| Mixed precision | bf16 |

**How `joint` mixes D_S and D_Q:** both datasets are tokenized independently (each with its own
prefix and target-length cap), then concatenated into a single `Dataset` before training --
`121,294` examples total (68,674 D_Q + 52,620 D_S). No custom sampler or interleaving schedule: the
`Trainer`'s default per-epoch random shuffle mixes the two tasks throughout training on its own,
and the model sees each task's full data once per epoch, same as it would see D_Q alone in
`baseline`. The two tasks are distinguished purely by input prefix -- there's no auxiliary loss
weighting, task embedding, or architecture change.

## Evaluation

```bash
python src/evaluate_model.py --source base
python src/evaluate_model.py --source summarization
python src/evaluate_model.py --source joint
python src/compare_results.py
```

**Metrics:** EM, F1, and BERTScore-F1 (`bert-base-multilingual-cased`) on D_Q -- identical to
`bangla-qa-banglat5`, computed the same way for all three conditions. For `joint`, also ROUGE-1/2/L
on D_S (`rouge-score`, no stemming) since that's the only condition that learns summarization.
Note: `rouge-score`'s default tokenizer only matches ASCII `[a-z0-9]+` and silently produces 0 for
non-Latin scripts; `evaluate_model.py` uses a plain whitespace tokenizer instead so Bangla text
scores correctly.

## Results

**Model stats:**

| | baseline | transfer | joint |
|---|---|---|---|
| Init checkpoint | `csebuetnlp/banglat5` | `Bangla-T5-finetuned-summary` | `csebuetnlp/banglat5` |
| Training data | D_Q (68,674) | D_Q (68,674) | D_S + D_Q (121,294) |
| Total params | 247,577,856 | 296,926,464 | 247,577,856 |
| Model size (fp32) | 944.4 MB | 1132.7 MB | 944.4 MB |

**Training** (RTX 4070, effective batch size 16):

| Epoch | baseline eval_loss | baseline time | transfer eval_loss | transfer time | joint eval_loss | joint time |
|-------|---------------------|----------------|----------------------|-----------------|-------------------|-------------|
| 1 | 0.9947 | 30.8 min | 0.6900 | 30.8 min | 2.422 | 67.5 min |
| 2 | 0.8127 | 30.8 min | 0.6681 | 30.9 min | 1.694 | 69.2 min |
| 3 | 0.7953 | 30.8 min | 0.6685 | 30.6 min | 1.647 | 67.4 min |

The baseline condition's loss here (epoch 3: 0.7953) is close to, but not identical to, the
original `bangla-qa-banglat5` repo's own run (epoch 3: 0.8003) -- expected run-to-run variance from
a different random seed / data shuffle, not a methodological difference.

**`joint`'s eval_loss is not comparable to baseline/transfer's.** It's averaged over the
concatenated D_Q + D_S validation set (2,251 examples: 1,251 QA + 1,000 summarization), a
fundamentally harder/longer generation task mixed in -- the higher absolute numbers say nothing
about QA quality on their own; see the EM/F1/BERTScore table below for that. `joint` also takes
~2.2x longer per epoch than baseline/transfer, tracking its ~1.77x larger combined training set
(121,294 vs 68,674 examples) plus the longer 128-token generation cap during eval.

**Evaluation (QA, D_Q):**

| Split | Condition | EM | F1 | BERTScore-F1 | N |
|-------|-----------|-----|-----|--------------|-----|
| Validation | baseline | 54.68 | 68.35 | 91.17 | 1,251 |
| Validation | transfer | 57.07 | 70.24 | 91.76 | 1,251 |
| Validation | joint | 52.76 | 66.93 | 90.90 | 1,251 |
| Test | baseline | 53.27 | 68.05 | 91.16 | 1,252 |
| Test | transfer | 54.55 | 69.34 | 91.60 | 1,252 |
| Test | joint | 52.64 | 67.29 | 90.95 | 1,252 |

| Test metric | baseline | transfer | joint | joint vs. baseline | joint vs. transfer |
|---|---|---|---|---|---|
| EM | 53.27 | 54.55 | 52.64 | -0.63 | -1.91 |
| F1 | 68.05 | 69.34 | 67.29 | -0.76 | -2.05 |
| BERTScore-F1 | 91.16 | 91.60 | 90.95 | -0.21 | -0.65 |

**Evaluation (Summarization, D_S -- `joint` only):**

| Split | ROUGE-1 | ROUGE-2 | ROUGE-L | N |
|-------|---------|---------|---------|-----|
| Validation | 21.41 | 9.26 | 18.74 | 1,000 |
| Test | 21.83 | 9.64 | 18.95 | 1,000 |

Manual inspection of `joint`'s D_S generations (see `outputs/logs/evaluate_joint.log` / re-run
`evaluate_model.py --source joint` to reproduce) shows fluent, on-topic Bangla summaries that
capture the article's main point, even where they don't lexically match the reference closely
enough to score high on ROUGE -- e.g. paraphrasing "গুড়ে ভেজাল" as "পচা চিনির রস দিয়ে গুড় তৈরির
অভিযোগ", correct in meaning but different wording. ROUGE-1/2/L, being surface n-gram overlap
metrics, systematically undercount abstractive paraphrasing like this.

**Finding:** the three conditions trace out the expected multi-task trade-off curve.
**Sequential transfer** (summarization-then-QA) is the best QA performer of the three -- prior
summarization fine-tuning transfers positively to question answering (+1.28 EM / +1.29 F1 over
baseline), matching the direction of the finding in
[bangla-qg-banglat5](../bangla-qg-banglat5) (summarization -> QG also transferred positively).
**Joint multi-task training**, in contrast, is the *worst* QA performer of the three (-0.63 EM vs.
baseline, -1.91 EM vs. transfer) -- but it's the only condition that also produces a model
genuinely capable of summarization (ROUGE-L ~19, fluent generations) from a single training run.
This is the standard single-model-capacity cost of joint multi-task learning: splitting gradient
updates and shared parameters across two objectives at once trades a bit of peak single-task
performance for one model that does both, whereas sequential transfer gets to fully specialize on
QA in a dedicated final phase after benefiting from whatever general-purpose fluency the
summarization phase installed.

## Limitations

**Single run, no variance estimate.** One seed, one training run per condition. The +1.28 EM /
+1.29 F1 (transfer) and -0.63 EM / -0.76 F1 (joint) gaps vs. baseline are point estimates; on a
1,252-example test set +1.28 EM corresponds to roughly 16 questions, noticeably larger than the
kind of single-question noise the original `bangla-qa-banglat5` repo flagged when comparing against
mBERT (0.95 EM, ~12 questions), but none of the three conditions is backed by a multi-seed variance
estimate.

**The two init checkpoints are not perfectly matched.** The summarization checkpoint has untied
embeddings (see Results); its own training data, epoch budget, and hyperparameters (from when it
was originally fine-tuned for summarization) are outside this repo's control and not documented
here.

**Fixed 3-epoch budget**, held constant across all three conditions to isolate the training-regime
variable. The transfer condition's eval loss is nearly flat after epoch 2 (0.6681 -> 0.6685),
suggesting it converges faster than 3 epochs actually requires, while the baseline was still
improving at epoch 3. `joint`'s eval_loss was also still falling at epoch 3 (2.422 -> 1.694 ->
1.647) with the steepest drop between epochs 1-2 -- plausible given it has ~1.77x more data to get
through per epoch than baseline/transfer, so at a fixed 3-epoch budget it simply sees each example
fewer effective "model-updates from convergence" than the single-task conditions do.

**`joint`'s D_Q and D_S are not a matched pair.** Unlike baseline vs. transfer (which hold D_Q and
hyperparameters fixed and vary only the init checkpoint), `joint` introduces an entirely different,
non-HuggingFace summarization corpus (MultiBanAbs) whose own collection methodology, domain
(Bangla news), and quality are outside this repo's control. The QA regression under `joint` could
be attributable to genuine multi-task capacity competition, to D_S's characteristics specifically
(e.g. news-domain text differing from squad_bn's Wikipedia-style contexts), or some mix of both --
this repo can't distinguish those without a second D_S source to compare against.

**No mixing-ratio sweep.** `joint` uses simple full-concatenation (D_S and D_Q each seen once per
epoch, no up/downsampling or temperature-based mixing) since the two datasets were already close in
size (52,620 vs. 68,674 train examples). A true multi-task learning study would typically sweep the
D_S:D_Q ratio to characterize the capacity-tradeoff curve rather than reporting one point on it.

**ROUGE undercounts abstractive correctness.** As noted in Results, `joint`'s D_S ROUGE scores
(surface n-gram overlap) likely understate its actual summarization quality where generations
paraphrase the reference rather than matching it lexically -- no human evaluation or a semantic
metric (e.g. BERTScore, as used for QA above) was run on the D_S outputs.

**Dataset caveat.** Shared with the sibling `bangla-qa-*` repos: a subset of `squad_bn`'s
`answer_start` offsets are off by 1-5 characters in validation/test. Irrelevant here since this
repo (like all the seq2seq QA repos) trains on answer *text*, never on `answer_start` offsets.

## Project Structure

```
bangla-qa-banglat5-crosstask/
├── data/
│   ├── raw/
│   │   └── MultiBanAbs_article_summary.json  # D_S source (local, not from HuggingFace)
│   ├── squad_bn/                    # D_Q: train/validation/test.json
│   └── summary_bn/                  # D_S: train/validation/test.json (joint condition only)
├── outputs/
│   ├── model/
│   │   ├── baseline/{checkpoints,best}
│   │   ├── transfer/{checkpoints,best}
│   │   └── joint/{checkpoints,best}
│   ├── results/
│   │   └── baseline.json / transfer.json / joint.json
│   └── logs/
├── src/
│   ├── prepare_data.py
│   ├── train.py              # --source {base,summarization,joint}
│   ├── evaluate_model.py     # --source {base,summarization,joint}
│   └── compare_results.py
├── run_all.sh                # trains + evaluates all three conditions end to end
├── requirements.txt
└── README.md
```

## Reproducing Results

```bash
cd bangla-qa-banglat5-crosstask
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For the joint condition: place your MultiBanAbs-style JSONL file at
# data/raw/MultiBanAbs_article_summary.json first (see D_S: summarization data above).
./run_all.sh
```
