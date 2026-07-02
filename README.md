# my-LLM

A small language model built **from scratch** as a learning project: a ~30M
parameter decoder-only transformer trained on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset,
running on Kaggle's free T4 GPUs.

The goal is to understand transformers deeply by implementing the core pieces by
hand — model, training loop, data pipeline, and tokenizer — and then running a
set of controlled experiments, ending in a technical write-up.

## Highlights

- **From scratch, no shortcuts.** The transformer is built directly on PyTorch
  tensor ops — embeddings, multi-head causal self-attention, and the training
  loop are hand-written, with no `nn.Transformer` or HuggingFace `model` classes
  doing the heavy lifting.
- **Multi-head attention done the clean way.** Heads are vectorized into a
  single batched matmul (reshape Q/K/V to `(batch, num_heads, seq, head_dim)`)
  rather than looped — the same approach production implementations use.
- **Built bottom-up and tested as it grows.** Each component is its own
  `nn.Module` snapping together against a known `(batch, seq_len, hidden_size)`
  contract, covered by a runnable end-to-end shape test plus a gradient-flow
  test that proves the whole model is trainable (see [Tests](#tests)).
- **Config-driven and reproducible.** Hyperparameters live in versioned YAML
  configs; a fixed seed keeps weights and inputs deterministic during
  development.
- **Designed for a real compute budget.** ~30M params targeting a single free
  Kaggle T4 — small enough to iterate fast, large enough to produce coherent
  text.

## Goals

- Implement a decoder-only transformer end to end, no high-level model libraries.
- Train it to generate coherent TinyStories-style text on a single T4.
- Run **ablations** to build intuition for which design choices matter:
  - RMSNorm vs LayerNorm
  - Warmup length
  - (add others as you go)
- Run a small **scaling experiment** (model size / data / compute vs loss).
- Write it all up as a technical blog post.

## Build progress

Implemented and smoke-tested so far:

- [x] BPE tokenizer training + encode/decode round-trip checks
- [x] Dataset inspection (story counts, char stats, samples)
- [x] YAML config loader (attribute + item access)
- [x] Token + learned-position embeddings
- [x] Multi-head causal self-attention (vectorized across heads)
- [x] Position-wise feed-forward network (GeLU, `d_ff` = 4 × hidden_size)
- [x] Full transformer block (pre-norm attention + FFN, residual connections)
- [x] GPT wrapper (stacked blocks + final norm + LM head → logits)
- [x] Training loop (AdamW, warmup + cosine schedule, AMP, grad clipping, checkpointing, wandb)
- [x] Kaggle T4 runner (notebook + tokenized Kaggle Dataset)
- [x] Sampling / generation (`scripts/generate.py`, stops at end-of-text)
- [x] Baseline trained: 30M model, 80k steps, **val_loss 1.43** (see [Results](#results))
- [x] Config-selectable norm type + hand-written RMSNorm (RMSNorm-vs-LayerNorm ablation implemented & pipeline-verified; see [Ablations](#ablations))
- [ ] Ablation training runs, scaling study, and write-up

## Tests

The model has no external test runner yet — tests live in each module's
`__main__` block and run on execution. For the model:

```bash
python src/model.py
```

This runs three checks against the `configs/tiny.yaml` model:

| Check | What it verifies |
| --- | --- |
| **End-to-end shape** | A `(batch, seq_len)` batch of token IDs produces logits of shape `(batch, seq_len, vocab_size)`. One forward pass exercises embeddings, every block's attention + FFN + norms, the final norm, and the LM head. |
| **Parameter count** | Reports total params (≈ budget). Doubles as a wiring check: a plain `list` instead of `nn.ModuleList`, or a submodule not assigned to `self.`, would silently drop blocks from the count. |
| **Gradient flow** | Builds a tiny throwaway GPT, backprops `logits.sum()`, and asserts every parameter receives a *non-zero* gradient — i.e. the whole model is connected to the autograd graph and trainable. Catches disconnected/untrained submodules that a shape check can't. |

Latest run (`tiny.yaml`: 4 layers, hidden 128, 4 heads, vocab 16k):

```
gpt out shape: (2, 128, 16000)
OK: GPT returns (batch, seq_len, vocab_size)
params: 4.9M
OK: gradients flow to all parameters
```

> The 4.9M here is the fast **debug** config; the ~30M target model uses a
> larger `hidden_size`. At this size the token embedding + LM head (vocab ×
> hidden) account for ~84% of params.

## Project structure

```
my-LLM/
├── configs/              # YAML experiment configs (see configs/tiny.yaml)
├── notebooks/
│   ├── exploration.ipynb  # scratch
│   └── kaggle_train.ipynb # Kaggle T4 run: clone → install → link data → train
├── scripts/
│   ├── dataset_stats.py  # inspect raw TinyStories (counts, char stats, samples)
│   ├── prepare_data.py   # download + tokenize TinyStories
│   ├── generate.py       # sample text from a trained checkpoint
│   └── kaggle_dataset/   # dataset-metadata.json for the tokenized Kaggle Dataset
├── src/
│   ├── model.py          # transformer: embeddings, attention, FFN, blocks, GPT wrapper
│   ├── train.py          # training loop
│   ├── data.py           # data loading / batching
│   ├── tokenizer.py      # tokenizer training
│   └── paths.py          # env-overridable paths (DATA_DIR, TOKENIZER_PATH, CKPT_DIR, CONFIG_PATH)
├── requirements.txt
└── README.md
```

> Data, checkpoints, tokenizers, and logs are git-ignored (regenerable / large).

## Setup

Local (CPU or GPU):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Install a torch build matching your CUDA — see https://pytorch.org/get-started/
```

On **Kaggle**, `torch` and `numpy` are preinstalled with CUDA for the T4. Don't
reinstall torch there; just add the lighter deps:

```bash
pip install tokenizers datasets pyyaml tqdm matplotlib
```

## Usage

<!-- Fill these in as the scripts take shape. -->

```bash
# 0. (optional) Inspect the raw dataset: story count, char stats, samples
python scripts/dataset_stats.py

# 1. Prepare data (download + train tokenizer + tokenize corpus)
python scripts/prepare_data.py --config configs/tiny.yaml

# 1b. (optional) Sanity-check the trained tokenizer: vocab size + encode/decode round-trip
python -m src.tokenizer

# 2. Train (defaults to configs/tiny.yaml; pick another with CONFIG_PATH=...)
python src/train.py

# 3. Generate
python scripts/generate.py --checkpoint checkpoints/... --prompt "Once upon a time"
```

## Run on Kaggle

Training targets a free Kaggle T4. The tokenized corpus and tokenizer ship as a
private **Kaggle Dataset** (the `.bin` files are too large for git), and
`notebooks/kaggle_train.ipynb` wires it all together: clone the repo, install
deps, point the code at the dataset, train.

**One-time: publish the tokenized data as a private Dataset** (from the repo root,
needs the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) authenticated via
`kaggle auth login`):

```bash
# stage the payload next to its metadata (hardlinks -> no ~900 MB copy)
ln -f data/train.bin data/val.bin tokenizer/tokenizer.json scripts/kaggle_dataset/
kaggle datasets create  -p scripts/kaggle_dataset                  # first upload
kaggle datasets version -p scripts/kaggle_dataset -m "refresh"     # later updates
```

**Then on Kaggle:** open `notebooks/kaggle_train.ipynb`, set Accelerator → GPU T4,
add the dataset as input, optionally add a `WANDB_API_KEY` secret, and Run All.

### Path overrides

`src/paths.py` resolves every path the run touches from environment variables,
falling back to the local layout — so the same code runs unchanged on Kaggle
(read-only data under `/kaggle/input`) with no edits or symlinks:

| Env var | Default | What |
| --- | --- | --- |
| `DATA_DIR` | `data` | dir holding `train.bin` / `val.bin` |
| `TOKENIZER_PATH` | `tokenizer/tokenizer.json` | tokenizer JSON |
| `CKPT_DIR` | `checkpoints` | checkpoint output dir |
| `CONFIG_PATH` | `configs/tiny.yaml` | YAML config to load |

So a one-off ablation needs no code change:

```bash
CONFIG_PATH=configs/no_warmup.yaml python src/train.py
```

## Experiments

| Experiment            | Variable            | Configs                     | Status |
| --------------------- | ------------------- | --------------------------- | ------ |
| Norm ablation         | RMSNorm vs LayerNorm| `ablation_norm_layernorm.yaml`, `ablation_norm_rmsnorm.yaml` | Implemented + smoke-tested; 22k runs pending |
| Warmup ablation       | warmup_steps        | `configs/...`               | TODO   |
| Scaling               | model size          | `configs/...`               | TODO   |

## Results

### Baseline — 30M model, 80k steps

The ~30M model (`configs/small.yaml`: hidden 512, 8 layers, 8 heads, vocab 8k,
seq 256 → **33.6M params**) trained on a single Kaggle T4 for the full 80k-step
warmup + cosine schedule (~4.5 h at ~42k tokens/sec):

![Training run: val_loss, train_loss, tokens_per_sec, step, lr, grad_norm over 80k steps](assets/baseline_80k_charts.png)

| Metric | Value |
| --- | --- |
| **val_loss** | **1.43** (train 1.46) |
| Throughput | ~42k tokens/sec (fp16 AMP, batch 32) |
| Params | 33.6M |
| Schedule | linear warmup (2k) → cosine decay to 10% of peak |

Sample (prompt `"Once upon a time"`, temperature 0.8):

> Once upon a time, there was a little girl named Lily. She loved to play in the
> garden with her toys. One day, she found a shiny rock and she wanted to keep it
> safe. She put it in her pocket and went to play with her toys. … She remembered
> the shiny rock in her pocket and thought it might make a magic wand.

The model produces fluent, coherent TinyStories-style narratives with consistent
characters and a clear story arc. Residual small-model artifacts (occasional
repetition or mild logic drift) are expected at this scale and are exactly what
the planned scaling study is meant to probe.

## Ablations

### 1. RMSNorm vs LayerNorm

**Status:** implemented and pipeline-verified; the two training runs are queued.

RMSNorm is hand-written (`src/model.py`, ~6 lines: divide each token vector by
its root-mean-square, then apply one learned scale — no mean-subtraction and no
shift, unlike LayerNorm). The norm type is selectable per run via `norm_type` in
the config, so the two arms share identical code and differ in exactly one field:

- `configs/ablation_norm_layernorm.yaml` — `norm_type: layernorm`
- `configs/ablation_norm_rmsnorm.yaml` — `norm_type: rmsnorm`

Both run at a reduced **22k-step** budget (vs the 80k baseline) to keep ablations
cheap, and share `wandb_group: norm_ablation` so their curves auto-overlay in
Weights & Biases.

**Implementation sanity checks** (done before spending any GPU time — a random
`(2, 4, 16)` input through each norm):

| Property | LayerNorm | RMSNorm |
| --- | --- | --- |
| Per-token mean of output | ≈ 0 (it re-centers) | non-zero (never centers) |
| Per-token RMS of output | ≈ 1 | ≈ 1 |

Both control magnitude identically (RMS ≈ 1); only LayerNorm also forces the mean
to 0. That single difference — LayerNorm's extra centering + shift parameter — is
the whole substance of the ablation.

**Parameter cost.** On the `tiny.yaml` debug build (hidden 128, 4 layers → 9
norms), LayerNorm gives 2.866M params vs RMSNorm's 2.865M. The ~1.15k gap is
exactly RMSNorm dropping LayerNorm's per-feature shift vector (9 norms × 128).
RMSNorm is strictly lighter.

**Pipeline smoke test.** A 200-step CPU run of the LayerNorm arm confirmed the
full path end to end: config → model builds with `norm_type` → training loop
(`train_loss` 9.13 → 3.94) → eval (`val_loss` 8.98 → 4.13) → sample + checkpoint
→ correct W&B grouping. So the real 22k runs should launch clean on the T4.

## Acknowledgements

Dataset: TinyStories (Eldan & Li, 2023).
