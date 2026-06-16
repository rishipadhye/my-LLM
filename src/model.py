"""Model components for the TinyStories LM.

Built bottom-up, one nn.Module per building block so each can be tested in
isolation: token+position embeddings first, then single-head causal
self-attention. Later layers (FFN, full transformer block, the GPT wrapper)
slot in by reusing the same (batch, seq_len, hidden_size) tensor shape, so the
blocks snap together in sequence.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import load_config
import math

torch.manual_seed(1337)  # reproducible weights/inputs while developing


class Embeddings(nn.Module):
    """Turn token IDs into vectors that also encode position.

    Two lookup tables added together: one maps each vocabulary token to a
    learned vector, the other maps each position (0..seq_len-1) to a learned
    vector. The sum lets a token's representation depend on both *what* it is
    and *where* it sits in the sequence.
    """

    def __init__(self, vocab_size, hidden_size, seq_len):
        super().__init__()
        # (vocab_size, hidden_size): one row per possible token.
        self.token_embedding_table = nn.Embedding(vocab_size, hidden_size)
        # (seq_len, hidden_size): one row per position slot.
        self.position_embedding_table = nn.Embedding(seq_len, hidden_size)

    def forward(self, x):
        # x: (batch, seq_len) of token IDs.
        token_embedding = self.token_embedding_table(x)  # (batch, seq_len, hidden_size)
        # Position IDs are just 0,1,2,... built on the fly; kept on x's device
        # so this works on both CPU and GPU.
        position_embedding = self.position_embedding_table(
            torch.arange(x.shape[1], device=x.device)
        )  # (seq_len, hidden_size)
        # Broadcasting adds the position vectors to every sequence in the batch.
        return token_embedding + position_embedding  # (batch, seq_len, hidden_size)


class SelfAttention(nn.Module):
    """Single-head causal self-attention.

    Each token forms a query, key, and value (three learned projections of the
    input). A token's query is compared against every token's key to decide how
    much to pull from each token's value. A causal mask prevents any token from
    attending to tokens that come after it, which is what makes this usable for
    next-token prediction.
    """

    def __init__(self, hidden_size):
        super().__init__()
        # Single head: the head spans the full hidden width.
        self.head_dim = hidden_size
        self.wq = nn.Linear(hidden_size, self.head_dim)  # query projection
        self.wk = nn.Linear(hidden_size, self.head_dim)  # key projection
        self.wv = nn.Linear(hidden_size, self.head_dim)  # value projection

    def forward(self, x):
        # x: (batch, seq_len, hidden_size). T is the current sequence length.
        T = x.shape[1]
        q = self.wq(x)  # (batch, T, head_dim)
        k = self.wk(x)  # (batch, T, head_dim)
        v = self.wv(x)  # (batch, T, head_dim)

        # Score every query against every key. transpose(-2,-1) swaps only the
        # last two dims -> (batch, head_dim, T), so the matmul contracts head_dim
        # and yields (batch, T, T): entry [i, j] = how much token i attends to j.
        scores = q @ k.transpose(-2, -1)  # (batch, T, T)
        # Scale by sqrt(head_dim) to keep score variance in check; without this,
        # large dot products push softmax into vanishing-gradient regions.
        scores = scores / math.sqrt(self.head_dim)

        # Causal mask: 1s strictly above the diagonal (diagonal=1 keeps the
        # diagonal itself, so a token can always see itself). Those future
        # positions are set to -inf so softmax drives them to exactly 0.
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float("-inf"))

        # Softmax over the last dim normalizes each query's row to sum to 1.
        attn = F.softmax(scores, dim=-1)  # (batch, T, T)
        # Weighted sum of values -> each token's context-mixed representation.
        out = attn @ v  # (batch, T, head_dim)
        return out


cfg = load_config("configs/tiny.yaml")
model = Embeddings(cfg.vocab_size, cfg.hidden_size, cfg.seq_len)
attention = SelfAttention(cfg.hidden_size)


if __name__ == "__main__":
    # --- Embeddings: check output shape on a fake batch of token IDs ---------
    batch, seq_len = 2, cfg.seq_len
    x = torch.randint(0, cfg.vocab_size, (batch, seq_len))  # fake token IDs
    out = model(x)
    print("input  shape:", tuple(x.shape))
    print("output shape:", tuple(out.shape))
    assert out.shape == (batch, seq_len, cfg.hidden_size), out.shape
    print("OK: embeddings return (batch, seq_len, hidden_size)")

    # --- Attention: tiny B=1, T=4 case to eyeball the causal mask ------------
    # On real embeddings this would be the Embeddings output; randn stands in.
    B = 1
    T = 4
    x_attn = torch.randn(B, T, cfg.hidden_size)
    out = attention(x_attn)
    print("attn out shape:", tuple(out.shape))
    assert out.shape == (B, T, cfg.hidden_size), out.shape
    print("OK: attention returns (batch, seq_len, hidden_size)")
