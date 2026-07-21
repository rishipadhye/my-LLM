import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import Config          
from model import GPT              
from tokenizer import load_tokenizer 
from paths import TOKENIZER_PATH   

CHECKPOINT = "checkpoints/ckpt_scale_xs.pt"
PROMPT = "Once upon a time. Once upon a time"

ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
cfg = Config(ckpt["config"])
model = GPT(cfg.vocab_size, cfg.hidden_size, cfg.seq_len, cfg.num_attention_heads, cfg.d_ff, cfg.num_layers, cfg.dropout_rate)
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = load_tokenizer(TOKENIZER_PATH)
enc = tokenizer.encode(PROMPT)
ids = enc.ids           # list of token ids, e.g. [402, 15, 88, ...]
tokens = enc.tokens     # matching strings, e.g. ['Once', 'Ġupon', 'Ġa', ...]
print(f"{len(ids)} tokens: {tokens}")

idx = torch.tensor([ids])    # shape (1, T): a batch of one sequence
with torch.no_grad():        # we don't need the logits — just the side effect
    model(idx)  

n_layers = cfg.num_layers
n_heads = cfg.num_attention_heads

induction_cells = []
seen = {}
for pos, tid in enumerate(ids):
    if tid in seen:         # this token appeared before
        target = seen[tid] + 1      # the position right after its first occurrence
        if target < pos:        # guard: target must be a real earlier position
            induction_cells.append((pos, target))
    seen[tid] = pos     # remember where we last saw this token id
print("induction cells (query_pos -> key_pos):", induction_cells)

scores = []
for layer in range(n_layers):
    for head in range(n_heads):
        A = model.blocks[layer].attn.last_attn[0, head]      # (T, T)
        score = sum(A[q, k].item() for (q, k) in induction_cells) / len(induction_cells)
        scores.append((score, layer, head))

scores.sort(reverse=True)
print("\ntop induction heads:")
for score, layer, head in scores[:5]:
    print(f"  L{layer} H{head}: {score:.3f}")