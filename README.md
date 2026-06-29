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
- [ ] Sampling / generation
- [ ] Ablations, scaling study, and write-up

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
| Norm ablation         | RMSNorm vs LayerNorm| `configs/...`               | TODO   |
| Warmup ablation       | warmup_steps        | `configs/...`               | TODO   |
| Scaling               | model size          | `configs/...`               | TODO   |

## Results

<!-- Plots and findings go here once you have training runs. -->

## Acknowledgements

Dataset: TinyStories (Eldan & Li, 2023).
