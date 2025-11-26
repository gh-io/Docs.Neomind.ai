import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel

# ----------------------------
# RoPE Utilities
# ----------------------------
def apply_rope(q, k, seq_len, rope_theta=10000.0, rope_scaling=None):
    B, H, T, D = q.shape
    dim = D * 2
    theta = torch.arange(0, dim, 2, device=q.device, dtype=q.dtype) / dim
    theta = 1.0 / (rope_theta ** theta)
    pos = torch.arange(T, device=q.device, dtype=q.dtype)
    freqs = torch.einsum('i,j->ij', pos, theta)

    if rope_scaling:
        short_factor = torch.tensor(rope_scaling['short_factor'], device=q.device, dtype=q.dtype)
        long_factor = torch.tensor(rope_scaling['long_factor'], device=q.device, dtype=q.dtype)
        freqs = freqs * short_factor + freqs * long_factor

    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    return q, k

# ----------------------------
# Sliding Window Attention
# ----------------------------
def sliding_window_attention(q, k, v, window_size):
    B, H, T, D = q.shape
    context = torch.zeros_like(q)
    for i in range(T):
        start = max(0, i - window_size // 2)
        end = min(T, i + window_size // 2 + 1)
        attn_scores = torch.matmul(q[:, :, i:i+1, :], k[:, :, start:end, :].transpose(-2, -1)) / math.sqrt(D)
        attn_probs = F.softmax(attn_scores, dim=-1)
        context[:, :, i:i+1, :] = torch.matmul(attn_probs, v[:, :, start:end, :])
    return context

# ----------------------------
# Attention Layer
# ----------------------------
class NeoMindAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = config.sliding_window

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling

    def forward(self, x, mask=None, past_kv=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)

        q, k = apply_rope(q, k, seq_len=T, rope_theta=self.rope_theta, rope_scaling=self.rope_scaling)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        if self.window_size is not None:
            context = sliding_window_attention(q, k, v, self.window_size)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            context = torch.matmul(attn_probs, v)

        out = self.resid_dropout(self.out_proj(context.transpose(1,2).contiguous().view(B, T, C)))
        return out, (k, v)

# ----------------------------
# Feedforward / MLP
# ----------------------------
class NeoMindMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU() if config.hidden_act == "gelu" else nn.SiLU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# ----------------------------
# Transformer Block
# ----------------------------
class NeoMindBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = NeoMindAttention(config)
        self.mlp = NeoMindMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, mask=None, past_kv=None):
        attn_out, kv = self.attn(self.norm1(x), mask=mask, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, kv

# ----------------------------
# Full NeoMind Model
# ----------------------------
class NeoMindModel(PreTrainedModel):
    config_class = NeoMindConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NeoMindBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        x = self.embed_tokens(input_ids)
        all_kv = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv = layer(x, mask=attention_mask, past_kv=past_kv)
            all_kv.append(kv)
        x = self.norm(x)
        return x, all_kv
