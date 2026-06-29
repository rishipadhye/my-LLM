"""Filesystem paths for a run, overridable via environment variables.

Every path the training code touches has a default that assumes you launched
from the repo root (the local dev layout). Each can be overridden with an env
var so the *same* code runs unchanged on Kaggle, where the data and tokenizer
are mounted read-only under /kaggle/input/<slug>/ — just export the vars before
calling train.py instead of symlinking files into place.

    DATA_DIR        dir holding train.bin / val.bin     (default: "data")
    TOKENIZER_PATH  path to the tokenizer JSON          (default: "tokenizer/tokenizer.json")
    CKPT_DIR        where checkpoints are written        (default: "checkpoints")
    CONFIG_PATH     YAML config to load                  (default: "configs/tiny.yaml")
"""

import os

DATA_DIR = os.environ.get("DATA_DIR", "data")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "tokenizer/tokenizer.json")
CKPT_DIR = os.environ.get("CKPT_DIR", "checkpoints")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/tiny.yaml")


def data_path(split: str) -> str:
    """Path to the {split}.bin file (split is "train" or "val")."""
    return os.path.join(DATA_DIR, f"{split}.bin")
