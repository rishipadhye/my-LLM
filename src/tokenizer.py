"""BPE tokenizer for the TinyStories LM.

The plan: train a byte-pair-encoding tokenizer with the `tokenizers` library,
save it as a single self-contained JSON, and reload it for data prep / training.

`train_tokenizer` is intentionally left for you to implement — see the module
chat / README for the API walkthrough. The save/load helpers and the round-trip
test below are plumbing that works against whatever `Tokenizer` you build.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE = 8000
DEFAULT_TOKENIZER_PATH = "tokenizer/tokenizer.json"

# A handful of strings that stress whitespace, punctuation, and casing so the
# round-trip test catches a mismatched pre-tokenizer/decoder pairing.
_DEFAULT_TEST_SAMPLES = [
    "Once upon a time, there was a little girl named Lily.",
    'She said, "Hello!" and ran   home.\nThe end.',
    "Numbers like 3 and 42, and a tab\tthen done.",
]


def train_tokenizer(corpus, vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Build and train a BPE tokenizer, returning the trained Tokenizer.

    YOU implement this. Sketch (do not just paste — understand each piece):
      1. Tokenizer(models.BPE(...))            # byte-level => no unk_token
      2. tokenizer.pre_tokenizer = ...         # e.g. ByteLevel(add_prefix_space=False)
      3. tokenizer.decoder = ...               # must invert the pre-tokenizer
      4. trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[...])
      5. tokenizer.train_from_iterator(corpus, trainer=trainer)
      6. return tokenizer

    `corpus` is an iterable of strings (e.g. the dataset's `text` column).
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=0, show_progress=True)
    tokenizer.train_from_iterator(corpus, trainer=trainer, length=len(corpus))
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: str | Path = DEFAULT_TOKENIZER_PATH) -> Path:
    """Save the tokenizer to a single self-contained JSON file.

    The `tokenizers` native format bundles the vocab, merges, pre-tokenizer, and
    decoder config into one JSON, so reloading needs nothing but this file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return path


def load_tokenizer(path: str | Path = DEFAULT_TOKENIZER_PATH) -> Tokenizer:
    """Load a tokenizer previously written by `save_tokenizer`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No tokenizer at {path}. Train one first (train_tokenizer + save_tokenizer)."
        )
    return Tokenizer.from_file(str(path))


def _compression_verdict(ratio: float) -> str:
    """Label a chars/token ratio against the ~3-4 healthy band for English BPE."""
    return "healthy" if 3.0 <= ratio <= 4.0 else "outside the ~3-4 healthy range"


def measure_compression(tokenizer: Tokenizer, texts: list[str]) -> float:
    """Return overall chars/token across `texts` (a real corpus sample).

    Uses `encode_batch` so this stays fast over thousands of stories. A single
    sentence is noisy; this aggregate is the number worth quoting.
    """
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(enc.ids) for enc in tokenizer.encode_batch(texts))
    return total_chars / total_tokens if total_tokens else 0.0


def test_roundtrip(tokenizer: Tokenizer, samples: list[str] | None = None) -> bool:
    """Encode then decode each sample and report whether text survives round-trip.

    Returns True if every sample decodes back to the original string. With a
    byte-level pre-tokenizer + byte-level decoder this should be lossless; a
    whitespace pre-tokenizer will typically *not* round-trip exactly (expected).
    """
    samples = samples if samples is not None else _DEFAULT_TEST_SAMPLES
    print(f"vocab size: {tokenizer.get_vocab_size():,}")

    all_ok = True
    total_chars = 0
    total_tokens = 0
    for i, text in enumerate(samples, start=1):
        enc = tokenizer.encode(text)
        decoded = tokenizer.decode(enc.ids)
        ok = decoded == text
        all_ok &= ok
        n_tokens = len(enc.ids)
        total_chars += len(text)
        total_tokens += n_tokens
        ratio = len(text) / n_tokens if n_tokens else 0.0
        status = "ok " if ok else "MISMATCH"
        print(f"\n[{status}] sample {i}: {n_tokens} tokens, {ratio:.2f} chars/token")
        print(f"  text:    {text!r}")
        print(f"  tokens:  {enc.tokens}")
        print(f"  ids:     {enc.ids}")
        if not ok:
            print(f"  decoded: {decoded!r}   <-- differs from input")

    # Compression ratio: ~3-4 chars/token is healthy for English BPE. Much lower
    # means the vocab is too small (over-splitting); much higher is unusual here.
    overall = total_chars / total_tokens if total_tokens else 0.0
    print(f"\ncompression: {overall:.2f} chars/token overall ({_compression_verdict(overall)})")
    print(f"round-trip: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check a trained tokenizer.")
    parser.add_argument(
        "--tokenizer-path",
        default=DEFAULT_TOKENIZER_PATH,
        help=f"Path to the tokenizer JSON. Default: {DEFAULT_TOKENIZER_PATH}.",
    )
    parser.add_argument(
        "--corpus-sample",
        type=int,
        default=None,
        metavar="N",
        help="Also measure chars/token over the first N TinyStories (needs `datasets`).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to sample for --corpus-sample. Default: train.",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    test_roundtrip(tokenizer)

    if args.corpus_sample:
        from datasets import load_dataset

        texts = load_dataset("roneneldan/TinyStories", split=args.split).select(
            range(args.corpus_sample)
        )["text"]
        ratio = measure_compression(tokenizer, texts)
        print(
            f"\ncorpus sample: {ratio:.2f} chars/token over {len(texts):,} stories "
            f"[{args.split}] ({_compression_verdict(ratio)})"
        )
