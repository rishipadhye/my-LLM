"""Download TinyStories from the Hugging Face Hub and print dataset stats.

This is an exploratory / EDA utility — it does NOT tokenize, batch, or otherwise
feed the model. It just pulls the raw dataset and reports basic numbers so you
can sanity-check the corpus and pick sensible vocab/seq_len values.

Usage:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --split validation --num-samples 5
    python scripts/dataset_stats.py --max-stories 50000   # quick subsample run

The dataset is cached by `datasets` under ~/.cache/huggingface, so repeated
runs don't re-download.
"""

from __future__ import annotations

import argparse
import textwrap

from datasets import load_dataset

DATASET_ID = "roneneldan/TinyStories"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="train",
        help="Which split to inspect (e.g. train, validation). Default: train.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="How many sample stories to print. Default: 3.",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Only scan the first N stories (faster spot-check). Default: all.",
    )
    parser.add_argument(
        "--sample-chars",
        type=int,
        default=500,
        help="Truncate each printed sample to this many characters. Default: 500.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for choosing which sample stories to print. Default: 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {DATASET_ID} [{args.split}] ...")
    dataset = load_dataset(DATASET_ID, split=args.split)

    if args.max_stories is not None:
        dataset = dataset.select(range(min(args.max_stories, len(dataset))))

    num_stories = len(dataset)

    # Single pass: total characters + per-story min/max for a quick length sense.
    total_chars = 0
    min_chars = None
    max_chars = 0
    for text in dataset["text"]:
        n = len(text)
        total_chars += n
        max_chars = max(max_chars, n)
        min_chars = n if min_chars is None else min(min_chars, n)

    avg_chars = total_chars / num_stories if num_stories else 0

    print("\n" + "=" * 60)
    print(f"Dataset:        {DATASET_ID}")
    print(f"Split:          {args.split}")
    print(f"Stories:        {num_stories:,}")
    print(f"Total chars:    {total_chars:,}")
    print(f"Avg chars/story:{avg_chars:>12,.1f}")
    print(f"Min chars/story:{min_chars if min_chars is not None else 0:>12,}")
    print(f"Max chars/story:{max_chars:>12,}")
    print("=" * 60)

    # Print a few random sample stories.
    n_show = min(args.num_samples, num_stories)
    if n_show:
        sample = dataset.shuffle(seed=args.seed).select(range(n_show))
        print(f"\n{n_show} sample stories (seed={args.seed}):")
        for i, text in enumerate(sample["text"], start=1):
            snippet = text[: args.sample_chars]
            if len(text) > args.sample_chars:
                snippet += " ..."
            print(f"\n--- sample {i} ({len(text):,} chars) ---")
            print(textwrap.fill(snippet, width=80))


if __name__ == "__main__":
    main()
