"""SVTRv2-style recognizer: mixing-attention backbone + FRM + (optional) SGM head
and per-domain adapters. Forward returns CTC log-probs of shape (B, T, num_classes).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import MODELS, NUM_CLASSES


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device) + keep).floor_()
        return x * mask / keep


class MixingAttention(nn.Module):
    """SVTR mixing layer: global self-attention + local depthwise-conv mixing."""

    def __init__(self, dim: int, num_heads: int = 8, local_k: int = 7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.local_conv = nn.Conv1d(dim, dim, local_k, padding=local_k // 2, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
        g = (attn @ v).transpose(1, 2).reshape(B, N, C)
        local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(g + local)


class SVTRBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MixingAttention(dim, num_heads)
        self.dp1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, dim), nn.Dropout(drop))
        self.dp2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.dp1(self.attn(self.norm1(x)))
        return x + self.dp2(self.mlp(self.norm2(x)))


class FRM(nn.Module):
    """Feature Rearrangement Module (lite): local temporal re-mixing of the
    collapsed sequence to fix CTC misalignment on tilted / partially-rolled wheels.
    """

    def __init__(self, dim: int, k: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim)
        self.pw = nn.Linear(dim, dim)

    def forward(self, x):                      # (B, T, D)
        h = self.norm(x)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        return x + self.pw(h)


class ContextualHead(nn.Module):
    """SGM (semantic guidance) auxiliary head — train-only, discarded at inference."""

    def __init__(self, dim, num_classes, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, nhead, dim * 4, dropout,
                                           activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.proj = nn.Linear(dim, num_classes)

    def forward(self, feats):
        return self.proj(self.encoder(feats))


class DomainAdapter(nn.Module):
    """Per-domain residual bottleneck adapter, FiLM-conditioned on a domain embedding.

    Identity-initialized (up-projection zeroed) so it is a no-op at the start of
    training and only specializes as domains diverge. One shared backbone, tiny
    per-domain params — the efficient multi-domain pattern.
    """

    def __init__(self, num_domains: int, dim: int, bottleneck: Optional[int] = None):
        super().__init__()
        b = bottleneck or max(8, dim // 4)
        self.down = nn.Linear(dim, b)
        self.up = nn.Linear(b, dim)
        self.act = nn.GELU()
        self.film = nn.Embedding(num_domains, b * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, domain_ids):          # x (B, T, D), domain_ids (B,)
        h = self.act(self.down(x))
        gamma, beta = self.film(domain_ids).chunk(2, dim=-1)   # (B, b) each
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return x + self.up(h)


class TambourNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES,
                 dims: Tuple[int, int, int] = (64, 128, 256),
                 depths: Tuple[int, int, int] = (3, 6, 9),
                 heads: Tuple[int, int, int] = (2, 4, 8),
                 drop: float = 0.08, drop_path: float = 0.1,
                 num_domains: int = 1, sgm: bool = False, **_: Any):
        super().__init__()
        self.num_domains = num_domains
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, 2, 1), nn.GELU())
        i = 0
        self.stage1 = nn.Sequential(*[SVTRBlock(dims[0], heads[0], drop=drop, drop_path=dpr[i + j]) for j in range(depths[0])]); i += depths[0]
        self.merge1 = nn.Sequential(nn.Conv2d(dims[0], dims[1], 3, (2, 1), 1), nn.GELU())
        self.stage2 = nn.Sequential(*[SVTRBlock(dims[1], heads[1], drop=drop, drop_path=dpr[i + j]) for j in range(depths[1])]); i += depths[1]
        self.merge2 = nn.Sequential(nn.Conv2d(dims[1], dims[2], 3, (2, 1), 1), nn.GELU())
        self.stage3 = nn.Sequential(*[SVTRBlock(dims[2], heads[2], drop=drop, drop_path=dpr[i + j]) for j in range(depths[2])])
        self.norm = nn.LayerNorm(dims[2])
        self.frm = FRM(dims[2])
        self.adapter = DomainAdapter(num_domains, dims[2]) if num_domains > 1 else None
        self.head = nn.Sequential(nn.Linear(dims[2], dims[2]), nn.ReLU(inplace=True),
                                  nn.Linear(dims[2], num_classes))
        self.sgm_head = ContextualHead(dims[2], num_classes) if sgm else None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Re-zero adapter up-projection after the generic init above.
        if self.adapter is not None:
            nn.init.zeros_(self.adapter.up.weight)
            nn.init.zeros_(self.adapter.up.bias)
            nn.init.zeros_(self.adapter.film.weight)

    @staticmethod
    def _to_2d(x, H, W):
        B, _, C = x.shape
        return x.permute(0, 2, 1).contiguous().view(B, C, H, W)

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        H, W = x.shape[2], x.shape[3]
        x = self.stage1(x.flatten(2).transpose(1, 2))
        x = self.merge1(self._to_2d(x, H, W))
        H, W = x.shape[2], x.shape[3]
        x = self.stage2(x.flatten(2).transpose(1, 2))
        x = self.merge2(self._to_2d(x, H, W))
        H, W = x.shape[2], x.shape[3]
        x = self.norm(self.stage3(x.flatten(2).transpose(1, 2)))
        x = self._to_2d(x, H, W).mean(2)         # collapse height -> (B, C, T)
        return self.frm(x.permute(0, 2, 1))      # (B, T, D)

    def forward(self, x, domain_ids: Optional[torch.Tensor] = None):
        feats = self.forward_backbone(x)
        if self.adapter is not None and domain_ids is not None:
            feats = self.adapter(feats, domain_ids)
        # Force fp32 for the log-softmax: under fp16/bf16 AMP an fp16 log-softmax
        # destroys CTC precision and collapses training (loss stalls, exact stays 0).
        # No-op cost when already fp32 (CPU / non-AMP).
        return self.head(feats).float().log_softmax(2)

    def load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            self.load_state_dict(state, strict=True)
        except RuntimeError:
            self.load_state_dict({k: v for k, v in state.items()
                                  if not k.startswith(("head.", "adapter.", "sgm_head."))}, strict=False)


def build_model(variant: str = "tambour-b", num_classes: int = NUM_CLASSES,
                num_domains: int = 1, sgm: bool = False, **overrides: Any) -> TambourNet:
    if variant not in MODELS:
        raise ValueError(f"unknown variant '{variant}'. choices: {list(MODELS)}")
    kwargs = dict(MODELS[variant]); kwargs.update(overrides)
    return TambourNet(num_classes=num_classes, num_domains=num_domains, sgm=sgm, **kwargs)


def model_config(variant: str, num_domains: int = 1, sgm: bool = False) -> Dict[str, Any]:
    cfg = dict(MODELS[variant])
    cfg.update(variant=variant, num_classes=NUM_CLASSES, num_domains=num_domains, sgm=sgm)
    return cfg
