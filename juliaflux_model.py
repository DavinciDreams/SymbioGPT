"""JuliaFluxGPT — PyTorch reimplementation of JuliaFluxGPT (Flux.jl).

LLaMA-style decoder with Grouped Query Attention (8Q/2KV), RMSNorm,
SwiGLU, RoPE, and weight-tied output. Matches model.jl exactly.

Config: d_model=512, n_layers=8, n_heads=8, n_kv_heads=2, head_dim=64,
        ctx=256, vocab=2000, ~23M params.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class JuliaFluxConfig:
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    context_length: int = 256
    vocab_size: int = 2000
    dropout: float = 0.0
    weight_tying: bool = True
    rope_base: float = 10000.0


# ═══════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos_cache", angles.cos())
        self.register_buffer("sin_cache", angles.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, T, head_dim)
        seq_len = x.size(2)
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos = self.cos_cache[:seq_len, :half].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len, :half].unsqueeze(0).unsqueeze(0)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        raw_inner = int(4 * d_model * 2 / 3)
        inner_dim = max(64, 64 * ((raw_inner + 32) // 64))  # round-to-nearest-64 (matches Julia)
        self.w_gate = nn.Linear(d_model, inner_dim, bias=False)
        self.w_up = nn.Linear(d_model, inner_dim, bias=False)
        self.w_down = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GQACausalAttention(nn.Module):
    """Grouped Query Attention with fused K+V projection.

    Matches JuliaFluxGPT's CausalSelfAttention:
    - wq: (d_model → n_heads * head_dim) for query
    - wkv: (d_model → 2 * n_kv_heads * head_dim) for fused key+value
    - proj: (d_model → d_model) output projection
    - KV heads repeated `groups` times to match query head count
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.groups = n_heads // n_kv_heads
        kv_dim = n_kv_heads * head_dim

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wkv = nn.Linear(d_model, 2 * kv_dim, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding,
                mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, KVH, HD = self.n_heads, self.n_kv_heads, self.head_dim

        # Query: (B, T, H*HD) → (B, H, T, HD)
        q = self.wq(x).view(B, T, H, HD).transpose(1, 2)

        # Fused K+V: (B, T, 2*KVH*HD) → split → each (B, KVH, T, HD)
        kv = self.wkv(x)
        kv_dim = KVH * HD
        k = kv[..., :kv_dim].view(B, T, KVH, HD).transpose(1, 2)
        v = kv[..., kv_dim:].view(B, T, KVH, HD).transpose(1, 2)

        # Apply RoPE
        q = rope(q)
        k = rope(k)

        # Repeat KV heads to match query heads
        if self.groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.groups, -1, -1)
            k = k.reshape(B, H, T, HD)
            v = v.unsqueeze(2).expand(-1, -1, self.groups, -1, -1)
            v = v.reshape(B, H, T, HD)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(HD)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back: (B, H, T, HD) → (B, T, H*HD)
        out = out.transpose(1, 2).contiguous().view(B, T, H * HD)
        return self.proj(out)


# ═══════════════════════════════════════════════════════════════════
# Transformer block and model
# ═══════════════════════════════════════════════════════════════════


class TransformerBlock(nn.Module):
    def __init__(self, config: JuliaFluxConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = GQACausalAttention(
            config.d_model, config.n_heads, config.n_kv_heads, config.head_dim
        )
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding,
                mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope, mask)
        x = x + self.ffn(self.ln2(x))
        return x


class JuliaFluxGPT(nn.Module):
    def __init__(self, config: JuliaFluxConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.head_dim, config.context_length, config.rope_base)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = RMSNorm(config.d_model)
        if config.weight_tying:
            self.head = None
        else:
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, self.rope, mask)
        x = self.ln_f(x)
        if self.head is not None:
            return self.head(x)
        return F.linear(x, self.tok_emb.weight)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def weight_entropy(self) -> float:
        """Shannon entropy of weight distribution (bits), 100 bins."""
        all_w = torch.cat([p.detach().flatten() for p in self.parameters()])
        if all_w.numel() == 0:
            return 0.0
        hist = torch.histc(all_w.float(), bins=100)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        return -(probs * torch.log2(probs)).sum().item()

    @property
    def effective_rank(self) -> float:
        """Average effective rank across all Linear layers (SVD, >1% threshold)."""
        ranks = []
        for module in self.modules():
            if isinstance(module, nn.Linear):
                w = module.weight.detach()
                try:
                    s = torch.linalg.svdvals(w)
                    threshold = 0.01 * s[0] if s.numel() > 0 and s[0] > 0 else 0.0
                    ranks.append((s > threshold).sum().item())
                except Exception:
                    ranks.append(float(min(w.shape)))
        return sum(ranks) / len(ranks) if ranks else 0.0


def load_from_npz(npz_path: str, config: JuliaFluxConfig = None) -> JuliaFluxGPT:
    """Load JuliaFluxGPT from NPZ file exported by convert_juliaflux.jl."""
    import numpy as np

    data = np.load(npz_path)

    # Read hyperparams if config not provided
    if config is None:
        config = JuliaFluxConfig(
            vocab_size=int(data["_hp_vocab_size"][0]),
            d_model=int(data["_hp_n_embd"][0]),
            context_length=int(data["_hp_block_size"][0]),
            n_layers=int(data["_hp_n_layer"][0]),
            n_heads=int(data["_hp_n_head"][0]),
            n_kv_heads=int(data["_hp_n_kv_head"][0]),
        )

    model = JuliaFluxGPT(config)

    # Build state_dict from NPZ arrays
    state_dict = {}
    for key in data.files:
        if key.startswith("_hp_"):
            continue
        arr = data[key]
        state_dict[key] = torch.from_numpy(arr.copy())

    model.load_state_dict(state_dict, strict=False)
    return model
