"""Generate stories from our TinyStories models + GPT-2 on a shared set of
prompts, for the LLM-as-judge evaluation.

Writes outputs/eval_generations.json — a list of {model, prompt, story} records
consumed by scripts/eval_judge.py. Same prompts + sampling settings for every
model, so the downstream judging is apples-to-apples.

Run from the repo root:  python scripts/eval_generate.py
"""
import os
import sys
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Keep the HuggingFace model cache INSIDE the repo (gitignored .hf_cache/) rather
# than writing ~500MB to ~/.cache — stays within the project directory.
os.environ.setdefault("HF_HOME", str(REPO / ".hf_cache"))

import torch  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from config import Config           # noqa: E402
from model import GPT               # noqa: E402
from tokenizer import load_tokenizer  # noqa: E402
from train import generate          # noqa: E402  (reuse the sampling loop)
from paths import TOKENIZER_PATH    # noqa: E402

# ---- shared generation settings (identical across all models) ----
DEVICE = "cpu"          # these models are tiny; CPU is instant
MAX_NEW_TOKENS = 160
TEMPERATURE = 0.8
TOP_K = 200
SEED = 1337

# Our trained checkpoints (label -> path). Add scale_l_80k here if downloaded.
OUR_MODELS = {
    "scale_xs (1.8M)": "checkpoints/ckpt_scale_xs.pt",
    "scale_l (56.6M)": "checkpoints/ckpt_scale_l.pt",
}

# Story openings, TinyStories-style. Same prompts for every model.
PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "One day, a boy named Tom found a shiny key in the garden.",
    "There was a cat who loved to sleep all day.",
    "In a big forest, a little rabbit was looking for food.",
    "Sara had a red balloon. She took it to the park.",
    "The sun was shining and Ben wanted to play outside.",
    "A tiny bird could not fly yet. Its mom said,",
    "Once there was a dog named Max who lost his ball.",
    "Lucy opened the old box and saw something magic inside.",
    "It was a cold winter day, so Mia put on her warm hat and",
]


def load_our_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = Config(ckpt["config"])
    model = GPT(cfg.vocab_size, cfg.hidden_size, cfg.seq_len,
                cfg.num_attention_heads, cfg.d_ff, cfg.num_layers, cfg.dropout_rate)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    records = []

    # ---- our TinyStories models (reuse the project's generate()) ----
    tok = load_tokenizer(TOKENIZER_PATH)
    for name, path in OUR_MODELS.items():
        print(f"generating: {name}")
        torch.manual_seed(SEED)  # reset per model so each sees the same noise
        model = load_our_model(path)
        for prompt in PROMPTS:
            story = generate(model, tok, DEVICE, prompt=prompt,
                             max_new_tokens=MAX_NEW_TOKENS,
                             temperature=TEMPERATURE, top_k=TOP_K)
            records.append({"model": name, "prompt": prompt, "story": story})

    # ---- GPT-2 (124M) baseline via HuggingFace ----
    print("generating: gpt2 (124M) — downloads ~500MB to .hf_cache/ on first run")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    torch.manual_seed(SEED)
    for prompt in PROMPTS:
        inp = gpt2_tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = gpt2.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=True, temperature=TEMPERATURE, top_k=TOP_K,
                                pad_token_id=gpt2_tok.eos_token_id)
        story = gpt2_tok.decode(out[0], skip_special_tokens=True)
        records.append({"model": "gpt2 (124M)", "prompt": prompt, "story": story})

    out_path = REPO / "outputs" / "eval_generations.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nwrote {len(records)} generations ({len(OUR_MODELS) + 1} models x {len(PROMPTS)} prompts) to {out_path}")


if __name__ == "__main__":
    main()
