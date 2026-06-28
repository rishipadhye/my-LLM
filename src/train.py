"""Training loop for the TinyStories LM.

Wires the from-scratch pieces together into one run: load config + tokenizer,
build the GPT model, then step through AdamW updates on random batches drawn
from the pre-tokenized data/*.bin files. Around the core update it layers the
things a real run needs:

    - a manual LR schedule (linear warmup -> cosine decay to 10% of peak)
    - mixed precision (fp16 autocast + GradScaler) for speed on the GPU
    - gradient clipping to survive the occasional bad batch
    - periodic wandb logging (loss / lr / grad_norm), validation, and samples
    - checkpoints that save RNG state so a run resumes bit-for-bit

Run `python src/train.py` (from the repo root, so configs/ resolves).
"""

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from config import load_config
from model import GPT
import math
import os
import re
import glob
import random
import numpy as np
from data import get_batch
from tokenizer import load_tokenizer
import wandb

CKPT_DIR = "checkpoints"


def evaluate(model, get_val_batch, eval_iters, device):
    """Average cross-entropy over `eval_iters` validation batches.

    eval() disables dropout and no_grad() skips the backward graph so this is
    cheap and deterministic-ish; both are restored (model.train()) before
    returning so the caller can keep training.
    """
    model.eval()
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for i in range(eval_iters):
            x, y = get_val_batch()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses[i] = loss.item()
    model.train()
    return losses.mean().item()


@torch.no_grad()
def generate(model, tokenizer, device, prompt="Once upon a time", max_new_tokens=200,
             temperature=0.8, top_k=200):
    """Autoregressively sample a continuation from `prompt` for logging.

    Greedy-ish sampling: at each step take the model's logits for the last
    position, optionally restrict to the top_k most likely tokens, divide by
    temperature (lower = more deterministic), softmax, and draw one token.
    The context is cropped to the model's trained seq_len because the position
    embeddings only exist for positions 0..seq_len-1.
    """
    was_training = model.training
    model.eval()
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.seq_len:]            # crop to the trained window
        logits = model(idx_cond)[:, -1, :]            # logits for the next token only
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    if was_training:
        model.train()
    return tokenizer.decode(idx[0].tolist())


def save_checkpoint(path, model, optimizer, scaler, step, config):
    """Save everything needed to resume bit-for-bit: weights, optimizer +
    scaler state, the step counter, the config, and all RNG states so the same
    data batches / dropout masks continue after a resume."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": config.as_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, path)


def find_latest_checkpoint(ckpt_dir=CKPT_DIR):
    """Return the path of the highest-step ckpt_<N>.pt in ckpt_dir, or None."""
    paths = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    if not paths:
        return None
    return max(paths, key=lambda p: int(re.search(r"ckpt_(\d+)\.pt", p).group(1)))


def load_checkpoint(path, model, optimizer, scaler, device):
    """Restore model/optimizer/scaler/RNG in place and return the saved step."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    rng = ckpt.get("rng")
    if rng is not None:
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return ckpt["step"]


def main():
    # ---- setup ----
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config("configs/tiny.yaml")

    wandb.init(project="tinystories-lm", name=config.run_name, config=config.as_dict())

    # Sanity check: the tokenizer that produced data/*.bin must match the vocab
    # the model's embedding/LM-head are sized for, or token IDs will index past
    # the embedding table (or silently waste rows).
    tokenizer = load_tokenizer()
    assert tokenizer.get_vocab_size() == config.vocab_size, (
        f"tokenizer vocab {tokenizer.get_vocab_size()} != config.vocab_size {config.vocab_size}"
    )

    model = GPT(config.vocab_size, config.hidden_size, config.seq_len, config.num_attention_heads, config.d_ff, config.num_layers, config.dropout_rate)
    model = model.to(device)

    def get_train_batch():
        return get_batch("train", config.seq_len, config.batch_size, device)

    def get_val_batch():
        return get_batch("val", config.seq_len, config.batch_size, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    def get_lr(step):
        # Linear warmup for the first warmup_steps (ramp from ~0 up to max_lr),
        # then a half-cosine decay from max_lr down to min_lr (= 10% of max_lr)
        # over the remaining steps. Warmup keeps the early, high-variance updates
        # from blowing up; cosine decay anneals the LR for a stable final fit.
        if step < config.warmup_steps:
            return config.max_lr * (step + 1) / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        if step > config.max_steps:
            progress = 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr + coeff * (config.max_lr - config.min_lr)

    # ---- resume from latest checkpoint if one exists ----
    start_step = 0
    latest = find_latest_checkpoint(CKPT_DIR)
    if latest is not None:
        start_step = load_checkpoint(latest, model, optimizer, scaler, device) + 1
        print(f"resuming from {latest} at step {start_step}")

    sample_interval = config.get("sample_interval", config.eval_interval)

    # ---- training loop ----
    model.train()
    for step in range(start_step, config.max_steps):
        # set LR manually (simpler than a scheduler object while learning)
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_train_batch()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Mixed-precision backward. Order matters: scale the loss up so small
        # fp16 grads don't underflow to 0, then unscale_ back to true values
        # *before* clipping (clip must see real gradient norms), clip, then let
        # the scaler step the optimizer (it skips the step if grads overflowed).
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # ---- periodic ----
        if step % config.log_interval == 0:
            wandb.log({"train_loss": loss.item(), "lr": lr,
                       "grad_norm": grad_norm.item(), "step": step})
        if step % config.eval_interval == 0:
            val_loss = evaluate(model, get_val_batch, config.eval_iters, device)
            wandb.log({"val_loss": val_loss, "step": step})
        if step % sample_interval == 0:
            sample = generate(model, tokenizer, device)
            wandb.log({"sample": wandb.Html(f"<pre>{sample}</pre>"), "step": step})
        if step > 0 and step % config.ckpt_interval == 0:
            save_checkpoint(os.path.join(CKPT_DIR, f"ckpt_{step}.pt"),
                            model, optimizer, scaler, step, config)

    # ---- final checkpoint (loop may not land on a ckpt_interval boundary) ----
    save_checkpoint(os.path.join(CKPT_DIR, f"ckpt_{config.max_steps}.pt"),
                    model, optimizer, scaler, config.max_steps, config)


if __name__ == "__main__":
    main()
