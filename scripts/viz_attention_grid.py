import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import Config          
from model import GPT              
from tokenizer import load_tokenizer 
from paths import TOKENIZER_PATH   

CHECKPOINT = "checkpoints/ckpt_scale_l.pt"
PROMPT = "Once upon a time there was a little girl"

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

fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.2 * n_heads, 2.2 * n_layers))

for layer in range(n_layers):
    for head in range(n_heads):
        ax = axes[layer][head]
        matrix = model.blocks[layer].attn.last_attn[0, head].numpy()
        ax.imshow(matrix, cmap="viridis")
        ax.set_title(f"L{layer} H{head}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle(f"scale_l attention — '{PROMPT}'", fontsize=11)
fig.savefig("assets/attn_grid_l.png", dpi=150, bbox_inches="tight")
print("saved to assets/attn_grid_l.png")