"""Generate text samples from a trained TinyStories checkpoint.

This is just a command-line wrapper: it loads a saved checkpoint, rebuilds the
model, and calls the generate() sampling loop that already lives in train.py.

Usage (run from the repo root):
    python scripts/generate.py
    python scripts/generate.py --prompt "Once upon a time" --num-samples 3
    python scripts/generate.py --checkpoint checkpoints/ckpt_20000.pt --temperature 0.7 --top-k 100
"""

import argparse
import sys
from pathlib import Path

import torch

# Put src/ on the import path so the bare imports below resolve the same way
# they do when you run `python src/train.py`. __file__ is this script;
# parents[1] is the repo root, and we append "src".
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import Config                            # noqa: E402
from model import GPT                                # noqa: E402
from tokenizer import load_tokenizer                 # noqa: E402
from train import generate, find_latest_checkpoint  # noqa: E402  (reuse the sampling loop)
from paths import CKPT_DIR, TOKENIZER_PATH           # noqa: E402


def main():
    # ---- command-line arguments ----
    # Each add_argument defines one flag. `type=` converts the text you type on
    # the command line into the right Python type; `default=` is used when you
    # don't pass the flag. The strings are shown by `--help`.
    parser = argparse.ArgumentParser(description="Sample text from a trained checkpoint.")
    parser.add_argument("--prompt", default="Once upon a time",
                        help="text to continue from")
    parser.add_argument("--checkpoint", default=None,
                        help="path to a ckpt_*.pt file (default: latest in CKPT_DIR)")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="how many tokens to generate after the prompt")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="lower = more deterministic, higher = more random")
    parser.add_argument("--top-k", type=int, default=200,
                        help="only sample from the k most likely next tokens")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="how many independent continuations to print")
    parser.add_argument("--seed", type=int, default=None,
                        help="set for reproducible samples")
    args = parser.parse_args()  # reads sys.argv; argparse names become args.<dest>
    #   note: argparse turns "--max-new-tokens" into the attribute args.max_new_tokens

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- pick the checkpoint ----
    # `or` falls through to find_latest_checkpoint only when --checkpoint wasn't given.
    ckpt_path = args.checkpoint or find_latest_checkpoint(CKPT_DIR)
    if ckpt_path is None:
        raise SystemExit(f"no --checkpoint given and none found in {CKPT_DIR}")
    print(f"loading {ckpt_path}")

    # ---- load the checkpoint file ----
    # map_location=device moves the saved tensors onto whatever hardware we have
    # right now, so a GPU-trained checkpoint still loads on a CPU-only machine.
    ckpt = torch.load(ckpt_path, map_location=device)

    # Rebuild the architecture from the config SAVED INSIDE the checkpoint, not
    # from configs/small.yaml on disk. This guarantees the model's shape matches
    # these exact weights even if you edited the YAML since training.
    cfg = Config(ckpt["config"])
    model = GPT(cfg.vocab_size, cfg.hidden_size, cfg.seq_len,
                cfg.num_attention_heads, cfg.d_ff, cfg.num_layers, cfg.dropout_rate)
    model.load_state_dict(ckpt["model"])  # copy the trained weights into the model
    model.to(device)
    model.eval()                          # disable dropout for clean inference

    # ---- tokenizer: converts your prompt text <-> token ids ----
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # ---- generate and print ----
    for i in range(args.num_samples):
        text = generate(model, tokenizer, device,
                        prompt=args.prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k)
        print(f"\n===== sample {i + 1} =====")
        print(text)


if __name__ == "__main__":
    main()
