#!/usr/bin/env bash
# =============================================================================
# setup.sh — one-shot, idempotent project setup for the TinyStories LM.
#
#   1. Ensures the git repo is initialized.
#   2. Ensures the expected directory structure exists.
#   3. Logs in to Weights & Biases (optional; skippable).
#
# Safe to re-run: it never overwrites an existing repo or existing files.
#
# Usage:
#   bash scripts/setup.sh           # full setup
#   SKIP_WANDB=1 bash scripts/setup.sh   # skip the wandb login step
# =============================================================================

set -euo pipefail

# Resolve repo root (parent of this script's dir) and work from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

info()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[setup]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }

# -----------------------------------------------------------------------------
# 1. Git
# -----------------------------------------------------------------------------
info "Checking git repository..."
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  ok "git already initialized ($(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'no branch yet'))."
else
  git init
  git symbolic-ref HEAD refs/heads/main 2>/dev/null || true   # prefer 'main'
  ok "git initialized on 'main'."
fi

# -----------------------------------------------------------------------------
# 2. Directory structure
# -----------------------------------------------------------------------------
# Source dirs are tracked; runtime dirs are gitignored (data/checkpoints/etc.)
# and only need to exist locally so paths in configs/tiny.yaml resolve.
info "Ensuring directory structure..."
SOURCE_DIRS=(src configs scripts notebooks)
RUNTIME_DIRS=(data tokenizer checkpoints outputs logs)

for d in "${SOURCE_DIRS[@]}" "${RUNTIME_DIRS[@]}"; do
  if [[ -d "$d" ]]; then
    printf '  exists   %s/\n' "$d"
  else
    mkdir -p "$d"
    printf '  created  %s/\n' "$d"
  fi
done
ok "Directory structure in place."

# -----------------------------------------------------------------------------
# 3. Weights & Biases login
# -----------------------------------------------------------------------------
if [[ "${SKIP_WANDB:-0}" == "1" ]]; then
  warn "SKIP_WANDB=1 set — skipping wandb login."
elif ! command -v wandb >/dev/null 2>&1; then
  warn "wandb not installed. Install it with:  pip install wandb"
  warn "then re-run this script (or just: wandb login)."
else
  info "Setting up Weights & Biases..."
  # `wandb login` is idempotent: it no-ops if a valid key is already cached.
  # It prompts for an API key (https://wandb.ai/authorize) if not logged in.
  if wandb login; then
    ok "wandb login complete."
  else
    warn "wandb login did not complete — you can re-run 'wandb login' anytime."
  fi
fi

ok "Setup done."
