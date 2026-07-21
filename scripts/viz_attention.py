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
PROMPT = "Once upon a time there was a little girl"
LAYER = 0     # which transformer block (0 to num_layers-1)
HEAD = 0      # which attention head

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

attn = model.blocks[LAYER].attn.last_attn      # (1, num_heads, T, T)
matrix = attn[0, HEAD].numpy()      # (T, T) for one head

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix, cmap="viridis")
fig.colorbar(im, ax=ax, label="attention weight")

ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=90)
ax.set_yticklabels(tokens)
ax.set_xlabel("key (attended-to token)")
ax.set_ylabel("query (current token)")
ax.set_title(f"scale_xs — layer {LAYER}, head {HEAD}")

fig.savefig("assets/attn_xs_l0_h0.png", dpi=150, bbox_inches="tight")
print("saved to assets/attn_xs_l0_h0.png")
