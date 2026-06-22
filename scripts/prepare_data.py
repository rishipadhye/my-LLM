import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

# Make `src/` importable when this is run as `python scripts/prepare_data.py`
# (the script's own dir, not the repo root, is what lands on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tokenizer import VOCAB_SIZE, load_tokenizer  # noqa: E402

DATA_DIR = "data"
BATCH_SIZE = 1000
EOT_TOKEN = "<|endoftext|>"
assert VOCAB_SIZE <= 2**16, "uint16 only holds vocab up to 65536; use uint32"


def main():
    # 1. LOAD ------------------------------------------------------------
    # HF ships 'train' and 'validation' splits, so no manual splitting.
    ds = load_dataset("roneneldan/TinyStories")

    # Load the tokenizer once and resolve the end-of-doc id up front: both
    # passes below must use the exact same encode + EOT, so they share these.
    tok = load_tokenizer()
    eot_id = tok.token_to_id(EOT_TOKEN)

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    # 2. PROCESS EACH SPLIT ---------------------------------------------
    for split_name, split in [("train", ds["train"]), ("val", ds["validation"])]:
        write_bin(split, f"{DATA_DIR}/{split_name}.bin", tok, eot_id)


def iterate_in_batches(split, batch_size=BATCH_SIZE):
    """Yield the 'text' column in lists of `batch_size` stories."""
    for start in range(0, len(split), batch_size):
        yield split[start : start + batch_size]["text"]


def write_bin(split, out_path, tok, eot_id):
    # --- PASS 1: count total tokens so we can preallocate the memmap ---
    # Tokenize in batches and sum the per-story lengths. The +1 accounts for
    # the EOT token PASS 2 appends to every story — keep these two in lockstep
    # or the final `assert idx == total_tokens` will trip.
    def count_lens(batch):
        encs = tok.encode_batch(batch["text"])
        return {"len": [len(e.ids) + 1 for e in encs]}

    lens = split.map(
        count_lens,
        batched=True,
        remove_columns=split.column_names,
        desc=f"counting tokens ({out_path})",
    )["len"]
    total_tokens = sum(lens)

    # --- allocate the file on disk (this part is the numpy mechanic) ---
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))

    # --- PASS 2: tokenize again, write each chunk at a moving cursor ---
    idx = 0
    for texts in iterate_in_batches(split):
        encs = tok.encode_batch(texts)
        # Flatten every story's ids + its EOT into one 1-D array for this batch.
        ids = []
        for e in encs:
            ids.extend(e.ids)
            ids.append(eot_id)
        ids = np.array(ids, dtype=np.uint16)

        arr[idx : idx + len(ids)] = ids         # <-- in-place slice write
        idx += len(ids)

    arr.flush()                                 # force everything to disk
    assert idx == total_tokens                  # sanity: you filled it exactly
    print(f"wrote {out_path}: {total_tokens:,} tokens")


if __name__ == "__main__":
    main()
