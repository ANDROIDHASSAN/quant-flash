"""A Llama-family decoder (Llama, Qwen2) written from scratch, loading Hugging Face checkpoints.

    tokens -> embedding -> N x [RMSNorm -> grouped-query attention with RoPE -> +, RMSNorm -> SwiGLU -> +]
           -> RMSNorm -> lm_head (tied to the embedding for small Qwen models)

The KV cache is a plain list of (keys, values) per layer, so it can be cut back to any length: speculative
decoding needs exactly that when the target model rejects some of the draft's tokens.
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int
    d_model: int
    d_ff: int
    n_layers: int
    n_heads: int
    n_kv_heads: int  # grouped-query attention: several query heads share one key/value head
    rope_theta: float = 10000.0
    eps: float = 1e-6
    qkv_bias: bool = False  # Qwen2 has biases on q, k, v; Llama does not
    tie: bool = True


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary positions, Hugging Face layout: dimension i is paired with i + d/2 (not with i + 1)."""
    d = x.shape[-1]
    freq = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angle = pos[:, None].float() * freq  # (T, d/2)
    cos, sin = torch.cat([angle.cos()] * 2, -1).to(x.dtype), torch.cat([angle.sin()] * 2, -1).to(x.dtype)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg, self.hd = cfg, cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(cfg.n_heads * self.hd, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, cache: list | None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.cfg.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.cfg.n_kv_heads, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.cfg.n_kv_heads, self.hd).transpose(1, 2)
        q, k = rope(q, pos, self.cfg.rope_theta), rope(k, pos, self.cfg.rope_theta)
        if cache is not None:
            if cache:
                k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
            cache[:] = [k, v]
        groups = self.cfg.n_heads // self.cfg.n_kv_heads
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        L = k.shape[2]  # cached + new
        mask = torch.ones(T, L, dtype=torch.bool, device=x.device).tril(L - T)  # new token i sees up to L - T + i
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.self_attn, self.mlp = Attention(cfg), MLP(cfg)
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.eps), RMSNorm(cfg.d_model, cfg.eps)

    def forward(self, x, pos, cache):
        x = x + self.self_attn(self.input_layernorm(x), pos, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Body(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.d_model)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.eps)


class LM(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Body(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        if cfg.tie:
            self.lm_head.weight = self.model.embed_tokens.weight

    def new_cache(self) -> list:
        return [[] for _ in range(self.cfg.n_layers)]

    def forward(self, tokens: torch.Tensor, cache: list | None = None) -> torch.Tensor:
        """tokens: (B, T) new tokens. Returns logits (B, T, vocab) for the next token after each of them."""
        start = cache[0][0].shape[2] if cache and cache[0] else 0
        pos = torch.arange(start, start + tokens.shape[1], device=tokens.device)
        x = self.model.embed_tokens(tokens)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, pos, cache[i] if cache is not None else None)
        return self.lm_head(self.model.norm(x)).float()


def cache_len(cache: list) -> int:
    return cache[0][0].shape[2] if cache[0] else 0


def truncate(cache: list, length: int) -> None:
    """Forget everything after the first `length` tokens (rejected draft tokens)."""
    for layer in cache:
        if layer:
            layer[:] = [layer[0][:, :, :length], layer[1][:, :, :length]]


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """8-byte header length, JSON header (name -> dtype, shape, byte offsets), raw bytes."""
    raw = bytearray(Path(path).read_bytes())
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n])
    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    out = {}
    for name, info in header.items():
        if name != "__metadata__":
            dtype = dtypes[info["dtype"]]
            start, end = info["data_offsets"]
            count = (end - start) // torch.tensor([], dtype=dtype).element_size()
            out[name] = torch.frombuffer(raw, dtype=dtype, count=count, offset=8 + n + start).reshape(info["shape"]).clone()
    return out


def load_pretrained(name: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu") -> LM:
    from huggingface_hub import hf_hub_download

    c = json.loads(Path(hf_hub_download(name, "config.json")).read_text())
    cfg = Config(vocab=c["vocab_size"], d_model=c["hidden_size"], d_ff=c["intermediate_size"],
                 n_layers=c["num_hidden_layers"], n_heads=c["num_attention_heads"], n_kv_heads=c["num_key_value_heads"],
                 rope_theta=c.get("rope_theta", 10000.0), eps=c["rms_norm_eps"],
                 qkv_bias=c.get("model_type") == "qwen2", tie=c.get("tie_word_embeddings", False))
    state = read_safetensors(hf_hub_download(name, "model.safetensors"))
    with torch.device("meta"):
        model = LM(cfg)
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=False, assign=True)
    if cfg.tie:
        model.lm_head.weight = model.model.embed_tokens.weight
    missing = [n for n, p in model.named_parameters() if p.is_meta]
    assert not missing, f"weights not in the checkpoint: {missing[:5]}"
    return model.to(device).eval()


def tokenizer(name: str):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
