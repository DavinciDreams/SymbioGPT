"""SymbioGPT — Multi-organelle GPT with learned per-channel gating.

Ports the Julia SymbioSLM architecture (DavinciDreams/julia-slm) to PyTorch
and adds CausalSelfAttention as a 4th organelle. Each SymbioBlock contains:

  1. CausalDepthwiseConv1d — local n-gram detection (O(n))
  2. MonarchMatrix — sub-quadratic global mixing via factored butterfly matrices (O(n√n))
  3. LongConv — dense causal convolution with exponential decay (O(n))
  4. CausalSelfAttention — standard multi-head causal attention with RoPE (O(n²))

The OrganelleGate learns a per-channel softmax blend over all organelles with
learnable temperature, allowing each embedding channel to independently specialize.

References:
  - Julia SymbioSLM: DavinciDreams/julia-slm (symbiogenesis.jl, monarch.jl)
  - Monarch Mixer: Dao et al., 2023
  - Hyena: Poli et al., 2023
  - Symbiogenesis: DavinciDreams/symbiogenesis
  - Margulis (1967): Endosymbiotic theory of organelle evolution
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════
# Building blocks (inlined from symbiogenesis for portability)
# ═══════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos_cache", angles.cos())
        self.register_buffer("sin_cache", angles.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to x: (batch, n_heads, seq_len, head_dim)."""
        seq_len = x.size(2)
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos = self.cos_cache[:seq_len, :half].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len, :half].unsqueeze(0).unsqueeze(0)
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        return torch.cat([o1, o2], dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: out = W2(swish(W1·x) * V·x)."""

    def __init__(self, d_model: int, ffn_mult: int = 4):
        super().__init__()
        raw_hidden = 2 * d_model * ffn_mult // 3
        hidden_dim = max(64, (raw_hidden // 64) * 64)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.v = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)) * self.v(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        total_dim = n_heads * head_dim
        self.wq = nn.Linear(d_model, total_dim, bias=False)
        self.wk = nn.Linear(d_model, total_dim, bias=False)
        self.wv = nn.Linear(d_model, total_dim, bias=False)
        self.wo = nn.Linear(total_dim, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, HD = self.n_heads, self.head_dim
        q = self.wq(x).view(B, T, H, HD).transpose(1, 2)
        k = self.wk(x).view(B, T, H, HD).transpose(1, 2)
        v = self.wv(x).view(B, T, H, HD).transpose(1, 2)
        q = rope(q)
        k = rope(k)
        scale = 1.0 / math.sqrt(HD)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * HD)
        return self.wo(out)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SymbioConfig:
    """Configuration for a SymbioGPT model."""

    d_model: int = 320
    n_layers: int = 8
    n_heads: int = 5  # for CausalSelfAttention organelle
    head_dim: int = 64
    ffn_mult: int = 4
    dropout: float = 0.0
    context_length: int = 256  # must be a perfect square for Monarch
    vocab_size: int = 2000
    weight_tying: bool = True

    # Organelle configuration
    organelles: Tuple[str, ...] = ("causal_conv", "monarch", "long_conv", "attention")
    conv_kernel_size: int = 4
    n_monarch_heads: int = 1

    # OrganelleGate
    gate_temperature_init: float = 1.0

    # Free energy regularization
    free_energy_beta: float = 0.001  # 0 = disabled

    # Per-layer organelle override (None = use global organelles for all layers)
    per_layer_organelles: Optional[List[Tuple[str, ...]]] = None

    def __post_init__(self):
        p = int(math.isqrt(self.context_length))
        if p * p != self.context_length:
            raise ValueError(
                f"context_length must be a perfect square for Monarch, "
                f"got {self.context_length}"
            )
        if self.d_model % self.n_monarch_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_monarch_heads ({self.n_monarch_heads})"
            )
        valid = {"causal_conv", "monarch", "long_conv", "attention"}
        for org in self.organelles:
            if org not in valid:
                raise ValueError(f"Unknown organelle: {org!r}, must be one of {valid}")

    @property
    def p(self) -> int:
        """Block size for Monarch factorization (sqrt of context_length)."""
        return int(math.isqrt(self.context_length))

    @property
    def n_organelles(self) -> int:
        return len(self.organelles)


# ═══════════════════════════════════════════════════════════════════
# Organelle 1: CausalDepthwiseConv1d (local n-gram patterns)
# ═══════════════════════════════════════════════════════════════════


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal convolution for local n-gram pattern detection.

    Each channel has its own 1D convolution kernel.
    Causality enforced via left-padding of (kernel_size - 1).

    Ports Julia CausalDepthwiseConv1d (monarch.jl).
    Parameters: kernel_size × channels
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        # Shape: (out_channels, in_channels/groups, kernel_size) for groups=channels
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        self._init_weights()

    def _init_weights(self):
        scale = math.sqrt(1.0 / self.kernel_size)
        nn.init.normal_(self.weight, mean=0.0, std=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape
        x_t = x.transpose(1, 2)  # (B, D, T)
        x_padded = F.pad(x_t, (self.kernel_size - 1, 0))  # (B, D, T+K-1)
        out = F.conv1d(x_padded, self.weight, groups=D)  # (B, D, T)
        return out.transpose(1, 2)  # (B, T, D)


# ═══════════════════════════════════════════════════════════════════
# Organelle 2: MonarchMatrix (sub-quadratic global mixing)
# ═══════════════════════════════════════════════════════════════════


class MonarchMatrix(nn.Module):
    """Monarch factored T×T mixing matrix (sub-quadratic).

    M = P^T · BlockDiag(L1) · P · BlockDiag(L2)
    where L1, L2 are p blocks of (p×p), T = p².

    Ports Julia MonarchMatrix (monarch.jl).
    Parameters: 2 × p³ = 2 × T^(3/2)
    """

    def __init__(self, seq_len: int):
        super().__init__()
        p = int(math.isqrt(seq_len))
        assert p * p == seq_len, f"Monarch requires perfect-square seq_len, got {seq_len}"
        self.seq_len = seq_len
        self.p = p

        scale = math.sqrt(2.0 / (p + p))
        self.L1 = nn.Parameter(torch.randn(p, p, p) * scale)
        self.L2 = nn.Parameter(torch.randn(p, p, p) * scale)

    @staticmethod
    def _julia_batched_mul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Julia NNlib.batched_mul: A(M,N,batch) @ B(N,K,batch) → (M,K,batch).

        PyTorch bmm uses batch-first, Julia uses batch-last.
        """
        return torch.bmm(
            A.permute(2, 0, 1),
            B.permute(2, 0, 1),
        ).permute(1, 2, 0)

    def realize(self) -> torch.Tensor:
        """Materialize full T×T Monarch matrix (differentiable).

        Pushes identity through: L2 → permute → L1 → permute.
        Follows Julia monarch_realize() exactly.
        Returns: (T, T) matrix.
        """
        p = self.p
        T = self.seq_len

        I_T = torch.eye(T, device=self.L1.device, dtype=self.L1.dtype)
        x = I_T.reshape(p, p, T)

        # Apply L2 block-diagonal (batch dim = last)
        x = x.permute(0, 2, 1)                        # (p, T, p)
        x = self._julia_batched_mul(self.L2, x)        # (p, T, p)
        x = x.permute(0, 2, 1)                        # (p, p, T)

        # Permutation P: transpose the p×p grid
        x = x.permute(1, 0, 2)

        # Apply L1 block-diagonal
        x = x.permute(0, 2, 1)                        # (p, T, p)
        x = self._julia_batched_mul(self.L1, x)        # (p, T, p)
        x = x.permute(0, 2, 1)                        # (p, p, T)

        # Undo permutation
        x = x.permute(1, 0, 2)

        return x.reshape(T, T)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply Monarch mixing.

        x: (B, T, D_head)
        causal_mask: (T_max, T_max) multiplicative 0/1 mask
        Returns: (B, T, D_head)
        """
        B, T, D_head = x.shape

        M = self.realize()  # (T_max, T_max)
        if causal_mask is not None:
            M = M * causal_mask[:T, :T]
        else:
            M = M[:T, :T]

        # (T, T) @ (T, B*D_head) → (T, B*D_head)
        x_flat = x.permute(1, 0, 2).reshape(T, B * D_head)
        y_flat = M @ x_flat
        return y_flat.reshape(T, B, D_head).permute(1, 0, 2)


# ═══════════════════════════════════════════════════════════════════
# Organelle 3: LongConv (global dense causal filter)
# ═══════════════════════════════════════════════════════════════════


class LongConv(nn.Module):
    """Full-length per-channel causal convolution with exponential decay init.

    Each channel has a kernel of length seq_len. Exponential decay
    initialization so recent positions are weighted more heavily.

    Ports Julia LongConv (symbiogenesis.jl).
    Parameters: seq_len × channels
    """

    def __init__(self, channels: int, seq_len: int):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        # Shape: (out_channels, in_channels/groups, kernel_size)
        self.kernel = nn.Parameter(torch.empty(channels, 1, seq_len))
        self._init_weights()

    def _init_weights(self):
        scale = math.sqrt(1.0 / self.seq_len)
        nn.init.normal_(self.kernel, mean=0.0, std=scale)
        with torch.no_grad():
            decay = torch.exp(-0.1 * torch.arange(self.seq_len, dtype=torch.float32))
            self.kernel.mul_(decay.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape
        K = self.seq_len
        x_t = x.transpose(1, 2)  # (B, D, T)
        x_padded = F.pad(x_t, (K - 1, 0))  # (B, D, T+K-1)
        out = F.conv1d(x_padded, self.kernel, groups=D)  # (B, D, T)
        return out.transpose(1, 2)  # (B, T, D)


# ═══════════════════════════════════════════════════════════════════
# OrganelleGate (per-channel softmax fusion)
# ═══════════════════════════════════════════════════════════════════


class OrganelleGate(nn.Module):
    """Per-channel softmax gating over N organelle outputs.

    Each channel independently learns which organelle to rely on via
    softmax over N logits, with a shared learnable temperature.
    Supports organelle masking for ablation studies.

    Ports Julia OrganelleGate (symbiogenesis.jl).
    Parameters: n_organelles × dim + 1 (temperature)
    """

    def __init__(self, dim: int, n_organelles: int, temperature_init: float = 1.0):
        super().__init__()
        self.dim = dim
        self.n_organelles = n_organelles
        self.logits = nn.Parameter(torch.zeros(n_organelles, dim))
        self.temperature = nn.Parameter(torch.tensor([temperature_init]))

    def forward(
        self,
        organelle_outputs: Tuple[torch.Tensor, ...],
        organelle_mask: Optional[Tuple[bool, ...]] = None,
    ) -> torch.Tensor:
        """Blend organelle outputs via per-channel gated softmax.

        organelle_outputs: tuple of N tensors, each (B, T, D)
        organelle_mask: optional tuple of N bools (True=enabled)
        Returns: (B, T, D)
        """
        logits = self.logits  # (N, D)

        if organelle_mask is not None:
            mask_additive = torch.zeros_like(logits)
            for i in range(self.n_organelles):
                if not organelle_mask[i]:
                    mask_additive[i, :] = -1e10
            logits = logits + mask_additive

        tau = self.temperature.clamp(min=0.01)
        weights = F.softmax(logits / tau, dim=0)  # (N, D)

        out = torch.zeros_like(organelle_outputs[0])
        for i in range(self.n_organelles):
            w = weights[i].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            out = out + w * organelle_outputs[i]

        return out


# ═══════════════════════════════════════════════════════════════════
# SkipGate (learnable residual scaling)
# ═══════════════════════════════════════════════════════════════════


class SkipGate(nn.Module):
    """Learnable scalar gate for residual connections.

    Scales the residual branch by a single learned parameter init=1.0.

    Ports Julia SkipGate (symbiogenesis.jl).
    Parameters: 1
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


# ═══════════════════════════════════════════════════════════════════
# SymbioSequenceMixer (all organelles + gate)
# ═══════════════════════════════════════════════════════════════════


class SymbioSequenceMixer(nn.Module):
    """Multi-organelle sequence mixer with learned gating.

    Runs all configured organelles in parallel on the input,
    then blends outputs via OrganelleGate.

    Ports and extends Julia SymbioSequenceMixer (symbiogenesis.jl).
    """

    def __init__(self, config: SymbioConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        T = config.context_length

        self.organelle_names = list(config.organelles)
        self.organelle_modules = nn.ModuleDict()

        for name in self.organelle_names:
            if name == "causal_conv":
                self.organelle_modules[name] = CausalDepthwiseConv1d(
                    d, config.conv_kernel_size
                )
            elif name == "monarch":
                self.organelle_modules[name] = nn.ModuleList(
                    [MonarchMatrix(T) for _ in range(config.n_monarch_heads)]
                )
            elif name == "long_conv":
                self.organelle_modules[name] = LongConv(d, T)
            elif name == "attention":
                self.organelle_modules[name] = CausalSelfAttention(
                    d, config.n_heads, config.head_dim, config.dropout
                )

        self.gate = OrganelleGate(
            d, len(self.organelle_names), config.gate_temperature_init
        )

        if "monarch" in self.organelle_names:
            self.register_buffer(
                "monarch_causal_mask", torch.tril(torch.ones(T, T))
            )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        attn_mask: Optional[torch.Tensor] = None,
        organelle_mask: Optional[Tuple[bool, ...]] = None,
    ) -> torch.Tensor:
        """Run all organelles in parallel and gate-blend.

        x: (B, T, D)
        rope: RotaryEmbedding for attention organelle
        attn_mask: (T, T) additive mask for attention (-inf/0)
        organelle_mask: optional per-organelle enable/disable
        Returns: (B, T, D)
        """
        B, T, D = x.shape
        outputs = []

        for name in self.organelle_names:
            if name == "causal_conv":
                out = self.organelle_modules[name](x)
            elif name == "monarch":
                heads = self.organelle_modules[name]
                n_mh = len(heads)
                hd = D // n_mh
                slices = []
                for i, monarch in enumerate(heads):
                    x_slice = x[:, :, i * hd : (i + 1) * hd]
                    y_slice = monarch(x_slice, self.monarch_causal_mask)
                    slices.append(y_slice)
                out = torch.cat(slices, dim=-1)
            elif name == "long_conv":
                out = self.organelle_modules[name](x)
            elif name == "attention":
                out = self.organelle_modules[name](x, rope, attn_mask)
            outputs.append(out)

        return self.gate(tuple(outputs), organelle_mask)


# ═══════════════════════════════════════════════════════════════════
# SymbioBlock (pre-norm residual block)
# ═══════════════════════════════════════════════════════════════════


class SymbioBlock(nn.Module):
    """Pre-norm residual block with organelle sequence mixing and skip gates.

    Architecture:
      x → RMSNorm → SymbioSequenceMixer → SkipGate → +residual
        → RMSNorm → SwiGLU → SkipGate → +residual → out

    Ports Julia SymbioBlock (symbiogenesis.jl).
    """

    def __init__(self, config: SymbioConfig, layer_organelles: Optional[Tuple[str, ...]] = None):
        super().__init__()
        d = config.d_model

        if layer_organelles is not None:
            from dataclasses import replace
            layer_config = replace(config, organelles=layer_organelles)
        else:
            layer_config = config

        self.ln1 = RMSNorm(d)
        self.seq_mixer = SymbioSequenceMixer(layer_config)
        self.skip1 = SkipGate()

        self.ln2 = RMSNorm(d)
        self.ffn = SwiGLU(d, config.ffn_mult)
        self.skip2 = SkipGate()

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        attn_mask: Optional[torch.Tensor] = None,
        organelle_mask: Optional[Tuple[bool, ...]] = None,
    ) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        normed = self.ln1(x)
        mixed = self.seq_mixer(normed, rope, attn_mask, organelle_mask)
        x = x + self.skip1(mixed)

        normed2 = self.ln2(x)
        ffn_out = self.ffn(normed2)
        x = x + self.skip2(ffn_out)

        return x


# ═══════════════════════════════════════════════════════════════════
# SymbioGPT (full model)
# ═══════════════════════════════════════════════════════════════════


class SymbioGPT(nn.Module):
    """SymbioGPT — Multi-organelle decoder-only causal language model.

    tok_emb → [SymbioBlock × n_layers] → ln_f → head (weight-tied)

    Supports configurable organelle composition per-layer.
    """

    def __init__(self, config: SymbioConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.head_dim, config.context_length)

        blocks = []
        for i in range(config.n_layers):
            layer_org = None
            if config.per_layer_organelles is not None:
                layer_org = config.per_layer_organelles[i]
            blocks.append(SymbioBlock(config, layer_org))
        self.blocks = nn.ModuleList(blocks)

        self.ln_f = RMSNorm(config.d_model)

        if config.weight_tying:
            self.head = None
        else:
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.in_features
                fan_out = module.out_features
                std = math.sqrt(2.0 / (fan_in + fan_out))
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        organelle_mask: Optional[Tuple[bool, ...]] = None,
    ) -> torch.Tensor:
        """input_ids (B, T) -> logits (B, T, V)"""
        B, T = input_ids.shape

        x = self.tok_emb(input_ids)

        attn_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, self.rope, attn_mask, organelle_mask)

        x = self.ln_f(x)

        if self.head is not None:
            logits = self.head(x)
        else:
            logits = F.linear(x, self.tok_emb.weight)

        return logits

    def get_gate_logits(self) -> List[torch.Tensor]:
        """Extract gate logits from all blocks for monitoring."""
        return [block.seq_mixer.gate.logits.detach() for block in self.blocks]

    def get_gate_weights(self) -> List[torch.Tensor]:
        """Extract gate softmax weights for visualization."""
        weights = []
        for block in self.blocks:
            gate = block.seq_mixer.gate
            tau = gate.temperature.clamp(min=0.01)
            w = F.softmax(gate.logits / tau, dim=0)
            weights.append(w.detach())
        return weights


# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════


def compute_symbio_params(config: SymbioConfig) -> int:
    """Compute exact parameter count for a SymbioGPT model."""
    d = config.d_model
    V = config.vocab_size
    L = config.n_layers
    T = config.context_length
    p = config.p

    emb = V * d

    per_layer = 0
    for org in config.organelles:
        if org == "causal_conv":
            per_layer += config.conv_kernel_size * d
        elif org == "monarch":
            per_layer += config.n_monarch_heads * 2 * p ** 3
        elif org == "long_conv":
            per_layer += T * d
        elif org == "attention":
            total_attn_dim = config.n_heads * config.head_dim
            per_layer += 4 * d * total_attn_dim  # wq, wk, wv, wo

    # OrganelleGate: logits + temperature
    per_layer += config.n_organelles * d + 1

    # SkipGate × 2
    per_layer += 2

    # SwiGLU FFN
    raw_hidden = 2 * d * config.ffn_mult // 3
    ffn_hidden = max(64, (raw_hidden // 64) * 64)
    per_layer += 3 * d * ffn_hidden

    # RMSNorm × 2
    per_layer += 2 * d

    # Final norm
    final_norm = d

    total = emb + L * per_layer + final_norm
    if not config.weight_tying:
        total += V * d

    return total


def complexity_penalty(model: nn.Module) -> torch.Tensor:
    """Free energy regularization: mean of squared log-weight magnitudes.

    Ports Julia complexity_penalty (free_energy.jl).
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    n_arrays = 0
    for param in model.parameters():
        if param.numel() > 0:
            total = total + (torch.log(param.abs() + 1e-6) ** 2).sum() / param.numel()
            n_arrays += 1
    return total / max(n_arrays, 1)


def compute_gate_entropy(model: SymbioGPT) -> float:
    """Average per-channel entropy of organelle gates across all blocks.

    Low entropy = strong specialization; high = uniform mixing.
    """
    gate_weights = model.get_gate_weights()
    if not gate_weights:
        return 0.0
    total_entropy = 0.0
    for w in gate_weights:
        H = -(w * torch.log(w + 1e-10)).sum() / w.shape[1]
        total_entropy += H.item()
    return total_entropy / len(gate_weights)
