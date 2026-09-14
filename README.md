# Bangla Question Answering with BanglaT5 -- Cross-Task Transfer

Fine-tunes [`csebuetnlp/banglat5`](https://huggingface.co/csebuetnlp/banglat5) for extractive
question answering (question + context -> answer) on Bangla SQuAD
([csebuetnlp/squad_bn](https://huggingface.co/datasets/csebuetnlp/squad_bn)), under four
conditions, to test whether -- and how -- cross-task exposure to a different task (Bangla
summarization) interacts with QA: not at all (baseline), sequentially (transfer), simultaneously
(joint), and a parameter-count-matched ablation of transfer (baseline_untied) that isolates
whether transfer's gain over baseline is cross-task learning or just extra parameters.

## Research question

Does exposure to Bangla summarization help a model learn question answering, and does *how* that
exposure happens (before QA training vs. simultaneously with it) matter? Four conditions, identical
QA data/hyperparameters, differing in whether/how summarization data (D_S) enters training and in
embedding tying:

| Condition | Init checkpoint | Training data | Embeddings | What it tests |
|---|---|---|---|---|
| **baseline** | `csebuetnlp/banglat5` (original pretrained) | D_Q only | tied (247.6M params) | QA fine-tuned from scratch |
| **baseline_untied** | `csebuetnlp/banglat5` (original pretrained) | D_Q only | untied (296.9M params) | Ablation: ties `transfer`'s parameter count to `baseline`'s init/data, with no cross-task exposure -- isolates whether `transfer`'s gain is cross-task learning or just more parameters |
| **transfer** | `../Bangla-T5-finetuned-summary` (already fine-tuned on Bangla summarization) | D_Q only | untied (296.9M params, inherited from the summarization checkpoint) | QA fine-tuned on top of a *different*, already-completed downstream task (sequential transfer) |
| **joint** | `csebuetnlp/banglat5` (original pretrained) | D_S + D_Q simultaneously | tied (247.6M params) | Multi-task joint learning: a single model trained on both tasks at once, distinguished only by input prefix (`"question: ..."` vs `"summarize: ..."`) |

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
python src/train.py --source base_untied      # baseline_untied ablation condition
```

Saves to `outputs/model/{baseline,transfer,joint,baseline_untied}/best`. Safe to interrupt and
re-run: a condition whose `best/` checkpoint already exists is skipped, and an in-progress
condition resumes from its latest epoch checkpoint rather than restarting.

**`base_untied` implementation note:** `tie_word_embeddings=False` must be passed into
`AutoModelForSeq2SeqLM.from_pretrained(...)` itself, not set on `model.config` after the call
returns. On the `transformers` version this repo was verified against (5.12.1), the pretrained
`csebuetnlp/banglat5` checkpoint already stores `shared.weight` and `lm_head.weight` as two
distinct on-disk tensors with different values; the loader detects that mismatch and refuses to
(re-)tie them at load time regardless of what the config says, before or after loading. Setting the
flag post-hoc is therefore a no-op that silently reproduces `baseline` instead of a
parameter-matched ablation -- confirmed empirically: both approaches gave `shared.weight ==
lm_head.weight` (False, distinct objects) and the same 247,577,856-param total either way, until
the flag was passed at construction time, which produces 296,926,464 params (matching `transfer`
exactly) via genuinely independent encoder-embedding/decoder-embedding/lm_head matrices.

**Hyperparameters** (identical across all four conditions, and identical to
`bangla-qa-banglat5` -- baseline/baseline_untied/transfer vary only the init checkpoint and
embedding tying; joint additionally mixes in D_S):

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
python src/evaluate_model.py --source base_untied
python src/compare_results.py
```

**Metrics:** EM, F1, and BERTScore-F1 (`bert-base-multilingual-cased`) on D_Q -- identical to
`bangla-qa-banglat5`, computed the same way for all four conditions. For `joint`, also ROUGE-1/2/L
on D_S (`rouge-score`, no stemming) since that's the only condition that learns summarization.
Note: `rouge-score`'s default tokenizer only matches ASCII `[a-z0-9]+` and silently produces 0 for
non-Latin scripts; `evaluate_model.py` uses a plain whitespace tokenizer instead so Bangla text
scores correctly.

## Results

**Model stats:**

| | baseline | baseline_untied | transfer | joint |
|---|---|---|---|---|
| Init checkpoint | `csebuetnlp/banglat5` | `csebuetnlp/banglat5` | `Bangla-T5-finetuned-summary` | `csebuetnlp/banglat5` |
| Training data | D_Q (68,674) | D_Q (68,674) | D_Q (68,674) | D_S + D_Q (121,294) |
| Total params | 247,577,856 | 296,926,464 | 296,926,464 | 247,577,856 |
| Model size (fp32) | 944.4 MB | 1132.7 MB | 1132.7 MB | 944.4 MB |

**Training** (RTX 4070, effective batch size 16):

| Epoch | baseline eval_loss | baseline time | baseline_untied eval_loss | baseline_untied time | transfer eval_loss | transfer time | joint eval_loss | joint time |
|-------|---------------------|----------------|----------------------------|------------------------|----------------------|-----------------|-------------------|-------------|
| 1 | 0.9947 | 30.8 min | 0.999 | 31.2 min | 0.6900 | 30.8 min | 2.422 | 67.5 min |
| 2 | 0.8127 | 30.8 min | 0.8267 | 32.2 min | 0.6681 | 30.9 min | 1.694 | 69.2 min |
| 3 | 0.7953 | 30.8 min | 0.8063 | 31.2 min | 0.6685 | 30.6 min | 1.647 | 67.4 min |

The baseline condition's loss here (epoch 3: 0.7953) is close to, but not identical to, the
original `bangla-qa-banglat5` repo's own run (epoch 3: 0.8003) -- expected run-to-run variance from
a different random seed / data shuffle, not a methodological difference. `baseline_untied` tracks
`baseline` closely throughout training (epoch 3: 0.8063 vs. 0.7953) and per-epoch time is
indistinguishable from `baseline`/`transfer` -- untying the embeddings adds ~49M parameters but
they're a small fraction of the ~198M non-embedding backbone, so it doesn't materially change
training cost. It does **not** track `transfer`'s much lower loss (epoch 3: 0.6685), which is the
first sign that `transfer`'s advantage isn't coming from having more parameters.

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
| Validation | baseline_untied | 55.16 | 68.66 | 91.25 | 1,251 |
| Validation | transfer | 57.07 | 70.24 | 91.76 | 1,251 |
| Validation | joint | 52.76 | 66.93 | 90.90 | 1,251 |
| Test | baseline | 53.27 | 68.05 | 91.16 | 1,252 |
| Test | baseline_untied | 53.35 | 68.27 | 91.21 | 1,252 |
| Test | transfer | 54.55 | 69.34 | 91.60 | 1,252 |
| Test | joint | 52.64 | 67.29 | 90.95 | 1,252 |

| Test metric | baseline | transfer | joint | joint vs. baseline | joint vs. transfer |
|---|---|---|---|---|---|
| EM | 53.27 | 54.55 | 52.64 | -0.63 | -1.91 |
| F1 | 68.05 | 69.34 | 67.29 | -0.76 | -2.05 |
| BERTScore-F1 | 91.16 | 91.60 | 90.95 | -0.21 | -0.65 |

### Ablation: is transfer's gain from cross-task learning or from parameter count?

`transfer` beats `baseline` on every metric, but `transfer` also carries ~49M more parameters
(untied embeddings, inherited from the summarization checkpoint) than `baseline`. `baseline_untied`
holds `baseline`'s init checkpoint and data fixed (no cross-task exposure at all) while matching
`transfer`'s parameter count exactly (296,926,464), decomposing `transfer`'s total gain into a
parameter-count term and a cross-task-transfer term:

| Test metric | baseline | baseline_untied | transfer | total gain (transfer − baseline) | parameter-count (baseline_untied − baseline) | cross-task-transfer (transfer − baseline_untied) |
|---|---|---|---|---|---|---|
| EM | 53.27 | 53.35 | 54.55 | +1.28 | +0.08 (6%) | +1.20 (94%) |
| F1 | 68.05 | 68.27 | 69.34 | +1.29 | +0.22 (17%) | +1.07 (83%) |
| BERTScore-F1 | 91.16 | 91.21 | 91.60 | +0.43 | +0.04 (9%) | +0.39 (91%) |

Validation shows the same pattern, slightly more pronounced (EM: +2.40 total, +0.48 parameter-count
[20%], +1.92 cross-task-transfer [80%]).

**Finding: the `transfer` gain is overwhelmingly a cross-task-transfer effect, not a parameter-count
effect.** `baseline_untied` sits barely above `baseline` (+0.08 EM / +0.22 F1 on test) despite having
the exact same 296.9M-parameter budget as `transfer` -- untying the embeddings and giving the model
more capacity, on its own, buys almost nothing without the prior summarization exposure. Nearly all
of `transfer`'s advantage over `baseline` (94% of the EM gap, 83% of F1, 91% of BERTScore-F1) only
shows up once that capacity is actually *used* by prior fine-tuning on a related generation task.
This also explains `baseline_untied`'s training-loss curve (Training, above) tracking `baseline`
rather than `transfer`: more parameters alone don't get you to `transfer`'s lower loss.

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
mBERT (0.95 EM, ~12 questions), but none of the four conditions is backed by a multi-seed variance
estimate. This applies to the `baseline_untied` ablation too: its +0.08 EM / +0.22 F1 edge over
`baseline` is a single point estimate (roughly 1 test question for EM) -- consistent with "no real
parameter-count effect," but not distinguishable from a genuinely tiny effect without repeated
seeds.

**The two init checkpoints are not perfectly matched.** The summarization checkpoint has untied
embeddings (see Results); its own training data, epoch budget, and hyperparameters (from when it
was originally fine-tuned for summarization) are outside this repo's control and not documented
here. `baseline_untied` addresses the *parameter-count* half of this mismatch specifically (see
Ablation, above) but not the rest -- e.g. the summarization checkpoint's own weights, beyond being
untied, still encode 54,620 examples of prior summarization gradient updates that `baseline_untied`
never sees.

**Fixed 3-epoch budget**, held constant across all four conditions to isolate the training-regime
variable. The transfer condition's eval loss is nearly flat after epoch 2 (0.6681 -> 0.6685),
suggesting it converges faster than 3 epochs actually requires, while the baseline was still
improving at epoch 3. `baseline_untied`'s eval loss was also still falling at epoch 3 (0.8267 ->
0.8063), tracking `baseline`'s own still-improving trajectory rather than `transfer`'s early
plateau. `joint`'s eval_loss was also still falling at epoch 3 (2.422 -> 1.694 ->
1.647) with the steepest drop between epochs 1-2 -- plausible given it has ~1.77x more data to get
through per epoch than baseline/transfer, so at a fixed 3-epoch budget it simply sees each example
fewer effective "model-updates from convergence" than the single-task conditions do.

**`baseline_untied`'s extra parameters start as copies of the pretrained checkpoint's embedding
values, not random initialization.** Untying via `tie_word_embeddings=False` at load time gives
`shared` (encoder+decoder input embedding), and `lm_head` four independent copies of the
pretrained checkpoint's embedding values at step 0 (not randomly-initialized matrices) -- they only
diverge from each other as QA fine-tuning proceeds. Verified directly against the trained
checkpoint's raw `model.safetensors` (not just the loader's tie-reconciliation logic, which can be
misleading -- see the `base_untied` implementation note under Training): `shared.weight`,
`encoder.embed_tokens.weight`, `decoder.embed_tokens.weight`, and `lm_head.weight` are four
pairwise-distinct tensors on disk after training, confirming they genuinely trained as independent
parameters rather than silently aliasing the same memory. This is the correct control for isolating
"does more capacity alone help": `baseline_untied` starts from *pretrained* (not random) embedding
values in every copy, same as `transfer` does; the difference under test is purely whether those
extra copies arrive already shaped by summarization fine-tuning (`transfer`) or not
(`baseline_untied`).

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
│   │   ├── joint/{checkpoints,best}
│   │   └── baseline_untied/{checkpoints,best}
│   ├── results/
│   │   └── baseline.json / transfer.json / joint.json / baseline_untied.json
│   └── logs/
├── src/
│   ├── prepare_data.py
│   ├── train.py              # --source {base,summarization,joint,base_untied}
│   ├── evaluate_model.py     # --source {base,summarization,joint,base_untied}
│   └── compare_results.py
├── run_all.sh                # trains + evaluates all four conditions end to end
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
