"""Model for the TinyStories LM: a from-scratch decoder-only transformer.

Built bottom-up, one nn.Module per building block, each reusing the same
(batch, seq_len, hidden_size) tensor shape so they snap together in sequence:

    Embeddings        token + learned-position lookup
    SelfAttention     multi-head causal self-attention (vectorized over heads)
    FeedForward       position-wise MLP (GeLU, d_ff = 4 x hidden_size)
    TransformerBlock  pre-norm attention + FFN sublayers, each with a residual
    GPT               embeddings -> N blocks -> final norm -> LM head -> logits

Run `python src/model.py` to execute the smoke + gradient-flow tests at the
bottom of this file.
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
    """Multi-head causal self-attention.

    Each token forms a query, key, and value (three learned projections of the
    input). A token's query is compared against every token's key to decide how
    much to pull from each token's value. A causal mask prevents any token from
    attending to tokens that come after it, which is what makes this usable for
    next-token prediction.

    "Multi-head" means hidden_size is split into num_heads independent heads,
    each of width head_dim = hidden_size / num_heads. Every head runs the same
    attention computation in parallel on its own slice of channels, letting the
    model attend to different relationships in different subspaces. The heads'
    outputs are concatenated and mixed by a final output projection. Total width
    (and so cost) stays at hidden_size regardless of num_heads.
    """

    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        # head_dim must divide evenly so the heads tile hidden_size exactly.
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(hidden_size, hidden_size)  # query projection
        self.wk = nn.Linear(hidden_size, hidden_size)  # key projection
        self.wv = nn.Linear(hidden_size, hidden_size)  # value projection
        self.proj = nn.Linear(hidden_size, hidden_size)  # output mix across heads
        # Two dropouts, GPT-2 placement: one on the attention weights (drops
        # whole token-to-token links), one on the block's output before it
        # re-enters the residual stream.
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size). T is the current sequence length.
        B = x.shape[0]
        T = x.shape[1]
        # Project, then split the hidden_size axis into (num_heads, head_dim) and
        # transpose so heads sit next to batch -> (batch, num_heads, T, head_dim).
        # With heads as a leading/batch dim, the matmuls below run over all heads
        # in parallel (no Python loop).
        q = self.wq(x).view(B, T, self.num_heads, -1)  # (batch, T, num_heads, head_dim)
        q = q.transpose(-3, -2)                         # (batch, num_heads, T, head_dim)
        k = self.wk(x).view(B, T, self.num_heads, -1)
        k = k.transpose(-3, -2)
        v = self.wv(x).view(B, T, self.num_heads, -1)
        v = v.transpose(-3, -2)

        # Score every query against every key. transpose(-2,-1) swaps the last
        # two dims of k -> (batch, num_heads, head_dim, T), so the matmul contracts
        # head_dim and yields (batch, num_heads, T, T): entry [..., i, j] = how
        # much token i attends to token j, computed independently per head.
        scores = q @ k.transpose(-2, -1)  # (batch, num_heads, T, T)
        # Scale by sqrt(head_dim) (the contracted dim) to keep score variance in
        # check; without this, large dot products push softmax into
        # vanishing-gradient regions.
        scores = scores / math.sqrt(self.head_dim)

        # Causal mask: 1s strictly above the diagonal (diagonal=1 keeps the
        # diagonal itself, so a token can always see itself). Those future
        # positions are set to -inf so softmax drives them to exactly 0. The
        # (T, T) mask broadcasts over the batch and num_heads dims.
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float("-inf"))

        # Softmax over the last dim normalizes each query's row to sum to 1.
        attn = F.softmax(scores, dim=-1)  # (batch, num_heads, T, T)
        attn = self.attn_dropout(attn)    # randomly zero some attention links
        # Weighted sum of values -> each head's context-mixed representation.
        out = attn @ v  # (batch, num_heads, T, head_dim)
        # Recombine heads: transpose back, then merge (num_heads, head_dim) into
        # one hidden_size axis. transpose makes the tensor non-contiguous, so
        # contiguous() is needed before view can re-flatten it.
        out = out.transpose(-3, -2)  # (batch, T, num_heads, head_dim)
        out = out.contiguous().view(B, T, self.num_heads * self.head_dim)  # (batch, T, hidden_size)
        # Output projection mixes information across heads.
        out = self.proj(out)  # (batch, T, hidden_size)
        out = self.resid_dropout(out)  # regularize the residual contribution
        return out

class FeedForward(nn.Module):
    """Position-wise feed-forward network: the per-token half of a block.

    Attention mixes information *across* tokens (each token looks back at
    earlier ones). This module does the opposite: it processes every token's
    vector independently and identically, with no looking sideways. Intuitively,
    once attention has gathered context into a token, the FFN is where the model
    "thinks" about what it gathered.

    It widens the channel dim to d_ff, applies a non-linearity in that larger
    space, then projects back down to hidden_size:
        (batch, seq_len, hidden_size) -> (..., d_ff) -> (batch, seq_len, hidden_size)
    The wider intermediate (conventionally d_ff = 4 * hidden_size) gives the
    non-linearity room to work; returning to hidden_size keeps the output shape
    matched to the residual stream so blocks stack cleanly.
    """

    def __init__(self, hidden_size, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, d_ff)  # up-projection:   hidden_size -> d_ff
        self.act = nn.GELU()                     # smooth non-linearity (GPT-style)
        self.fc2 = nn.Linear(d_ff, hidden_size)  # down-projection: d_ff -> hidden_size
        self.dropout = nn.Dropout(dropout)       # on the block's residual output

    def forward(self, x):
        # x: (batch, seq_len, hidden_size). Linear acts on the last dim only, so
        # batch and seq_len pass through untouched -- this is the per-token part.
        x = self.fc1(x)      # (batch, seq_len, d_ff)
        x = self.act(x)      # (batch, seq_len, d_ff)
        x = self.fc2(x)      # (batch, seq_len, hidden_size)
        x = self.dropout(x)  # (batch, seq_len, hidden_size)
        return x
    
class RMSNorm(nn.Module):
    """Root-mean-square normalization: a lighter alternative to LayerNorm.

    LayerNorm does two things to each token's vector: re-centers it to mean 0,
    then rescales it to unit variance -- with a learned scale *and* a learned
    shift. RMSNorm keeps only the rescaling: it divides each vector by its
    root-mean-square magnitude and applies a single learned per-feature scale.
    Dropping the mean-subtraction (and the shift parameter) makes it cheaper,
    and in practice just as effective -- it's what Llama and most modern LMs use.

    Selectable against nn.LayerNorm via the model's `norm_type` config, which is
    exactly the RMSNorm-vs-LayerNorm ablation this project runs.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps  # guards the division against a zero-magnitude vector
        # One learned scale per feature, initialized to 1 so the layer starts as
        # a pure normalizer (and note: no shift term, unlike LayerNorm).
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq_len, dim). RMS is computed per token over the last
        # (feature) dim; keepdim=True keeps the (..., 1) shape so it broadcasts
        # cleanly back over x. eps sits *inside* the sqrt so the denominator can
        # never be exactly 0.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = (x / rms) * self.scale  # normalize, then apply the learned scale
        return x



class TransformerBlock(nn.Module):
    """One pre-norm transformer block: attention sublayer + FFN sublayer.

    Pre-norm means each sublayer normalizes its input *before* the attention/FFN
    and adds the result back onto the residual stream (`x + sublayer(norm(x))`).
    Keeping the residual path un-normalized is what lets many blocks stack while
    gradients still flow cleanly to the bottom.
    """

    def __init__(self, hidden_size, num_heads, d_ff, dropout, norm_type="layernorm"):
        super().__init__()
        self.attn = SelfAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, d_ff, dropout)
        # Norm type is config-selectable so the RMSNorm-vs-LayerNorm ablation
        # needs no model-code change -- just a different `norm_type` in the YAML.
        if norm_type == "layernorm":
            self.norm_attn = nn.LayerNorm(hidden_size)
            self.norm_ffn = nn.LayerNorm(hidden_size)
        else:  # "rmsnorm"
            self.norm_attn = RMSNorm(hidden_size)
            self.norm_ffn = RMSNorm(hidden_size)

    def forward(self, x):
        # Pre-norm + residual: normalize, run the sublayer, add back onto x.
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_len, num_heads, d_ff, num_layers, dropout, norm_type="layernorm"):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, hidden_size, seq_len)
        self.drop = nn.Dropout(dropout)  # embedding dropout, applied once at the bottom
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads, d_ff, dropout, norm_type) for _ in range(num_layers)])
        # Final norm before the LM head; matches the blocks' norm_type so the
        # whole model uses one norm family per run (see TransformerBlock).
        if norm_type == "layernorm":
            self.norm_final = nn.LayerNorm(hidden_size)
        else:  # "rmsnorm"
            self.norm_final = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers

        # GPT-2 style init: weights ~ N(0, 0.02), biases 0, embeddings ~ N(0, 0.02).
        self.apply(self._init_weights)
        # Residual scaling: the projections that *write back* into the residual
        # stream (attn.proj, ffn.fc2) get an extra 1/sqrt(2*num_layers) shrink so
        # the residual variance doesn't grow with depth (GPT-2 / nanoGPT trick).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.drop(self.embeddings(idx))
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        logits = self.head(x)
        return logits

def test_gradient_flow():
    """Confirm the whole model is trainable end to end.

    Stronger than a shape check: builds a tiny throwaway GPT, runs a forward
    pass, collapses the logits to a scalar with .sum() (backward needs a
    scalar; summing routes gradient to every logit), and backprops. Then it
    asserts every parameter received a *non-zero* gradient. A None grad means a
    param is disconnected from the graph -- exactly the symptom of a plain list
    instead of nn.ModuleList, or a sublayer built in __init__ but never used in
    forward. named_parameters() surfaces which param if one fails.
    """
    m = GPT(vocab_size=50, hidden_size=16, seq_len=8, num_heads=2, d_ff=32, num_layers=2, dropout=0.1, norm_type="layernorm")

    B, T = 4, 8
    idx = torch.randint(0, 50, (B, T))  # fake token IDs

    logits = m(idx)
    assert logits.shape == (B, T, 50), logits.shape

    loss = logits.sum()  # scalar so backward() has something to differentiate
    loss.backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert p.grad.abs().sum() > 0, f"zero grad for {name}"

    print("OK: gradients flow to all parameters")

if __name__ == "__main__":
    # Building the model is a side effect, so it lives here (not at module
    # scope): importing GPT from train.py must not require a config on disk or
    # silently construct a throwaway model.
    cfg = load_config("configs/tiny.yaml")
    gpt = GPT(cfg.vocab_size, cfg.hidden_size, cfg.seq_len,
              cfg.num_attention_heads, cfg.d_ff, cfg.num_layers, cfg.dropout_rate, cfg.norm_type)

    # --- GPT end-to-end: token IDs in, per-position vocab logits out ----------
    # One pass exercises every component transitively (embeddings, all blocks'
    # attention + FFN + norms, final norm, LM head), so this single shape check
    # subsumes the per-component shape tests.
    batch, seq_len = 2, cfg.seq_len
    idx = torch.randint(0, cfg.vocab_size, (batch, seq_len))  # fake token IDs
    logits = gpt(idx)
    print("gpt out shape:", tuple(logits.shape))
    assert logits.shape == (batch, seq_len, cfg.vocab_size), logits.shape
    print("OK: GPT returns (batch, seq_len, vocab_size)")

    # --- Parameter count: sanity-check the ~budget. Also a wiring check: a
    # missing nn.ModuleList / self. would silently drop blocks from this sum.
    n_params = sum(p.numel() for p in gpt.parameters())
    print(f"params: {n_params / 1e6:.1f}M")

    # --- Gradient flow: every param must receive a non-zero grad after backward.
    test_gradient_flow()

    idx = torch.randn(2,4,16)
    rms = RMSNorm(16)
    logits = rms(idx)
    print("rms out shape:", tuple(logits.shape))
    assert logits.shape == (2, 4, 16), logits.shape
    print(logits.pow(2).mean(dim=-1))
    print("rms mean:", logits.mean(dim=-1))
    print("OK: RMS returns (batch, seq_len, vocab_size)")
    layer_norm = nn.LayerNorm(16)
    logits = layer_norm(idx)
    print("layer_norm out shape:", tuple(logits.shape))
    assert logits.shape == (2, 4, 16), logits.shape
    print(logits.pow(2).mean(dim=-1))
    print("layernorm mean:", logits.mean(dim=-1))
    print("OK: Layer Norm returns (batch, seq_len, vocab_size)")

