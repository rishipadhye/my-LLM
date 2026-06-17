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
- **Built bottom-up and tested in isolation.** Each component is its own
  `nn.Module` with a runnable smoke test asserting tensor shapes, so building
  blocks snap together against a known `(batch, seq_len, hidden_size)` contract.
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
- [ ] Full transformer block (attention + FFN + residuals + norm)
- [ ] GPT wrapper (stacked blocks + LM head)
- [ ] Training loop (AdamW, warmup, gradient clipping)
- [ ] Sampling / generation
- [ ] Ablations, scaling study, and write-up

## Project structure

```
my-LLM/
├── configs/              # YAML experiment configs (see configs/tiny.yaml)
├── notebooks/            # exploration / scratch
├── scripts/
│   ├── dataset_stats.py  # inspect raw TinyStories (counts, char stats, samples)
│   ├── prepare_data.py   # download + tokenize TinyStories
│   └── generate.py       # sample text from a trained checkpoint
├── src/
│   ├── model.py          # transformer: attention, blocks, embeddings
│   ├── train.py          # training loop
│   ├── data.py           # data loading / batching
│   └── tokenizer.py      # tokenizer training
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

# 2. Train
python -m src.train --config configs/tiny.yaml

# 3. Generate
python scripts/generate.py --checkpoint checkpoints/... --prompt "Once upon a time"
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
