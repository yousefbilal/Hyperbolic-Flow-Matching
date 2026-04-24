"""Classifier-Free Guidance (CFG) conditioning for the RFM vector field.

Mirrors `manifm/model/arch.py::tMLP` but the time-concatenation layers also
consume a class embedding. The rest of the architecture is untouched: activation
layers still take only scalar time, and the outer `ProjectToTangent` / `Unbatch`
wrappers are reused verbatim.

Conditioning is threaded in via a per-forward attribute on the top-level
`SequentialDiffEqCond`: `ConditionalVecfield.forward(t, x, labels)` sets
`inner._cond_emb = embed(labels)` before calling `inner(t, x)`, and each
`ConcatLinearCondV2` sub-layer reads that tensor to build its time-plus-class
bias input. No vmap call path needs labels (loglikelihood evaluation is
unconditional), so a stored-attribute approach is safe for training + sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import manifm.model.diffeq_layers as diffeq_layers
from manifm.model.arch import ACTFNS, PositionalEncoding


class ConcatLinearCondV2(nn.Module):
    """Linear layer with time + class-embedding additive bias.

    Matches `ConcatLinear_v2` when the class embedding is zero / absent.
    The conditioning tensor is read from `self._cond_emb` (set by the parent
    container each forward call).
    """

    def __init__(self, dim_in: int, dim_out: int, cond_dim: int):
        super().__init__()
        self.cond_dim = cond_dim
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + cond_dim, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)
        self._cond_emb: torch.Tensor | None = None

    def set_cond(self, cond_emb: torch.Tensor | None):
        self._cond_emb = cond_emb

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tt = t.reshape(-1, 1).to(x)
        if tt.shape[0] != B:
            tt = tt.expand(B, 1)
        if self._cond_emb is None:
            zeros = torch.zeros(B, self.cond_dim, device=x.device, dtype=x.dtype)
            tcond = torch.cat([tt, zeros], dim=-1)
        else:
            cond = self._cond_emb.to(x)
            if cond.ndim == 1:
                cond = cond.unsqueeze(0)
            if cond.shape[0] != B:
                cond = cond.expand(B, -1)
            tcond = torch.cat([tt, cond], dim=-1)
        return self._layer(x) + self._hyper_bias(tcond)


class SequentialDiffEqCond(nn.Module):
    """Container that propagates the class embedding to conditional sub-layers.

    Non-conditional layers (e.g. `TimeDependentSwish`) see only (t, x) as before.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(list(layers))
        self._cond_emb: torch.Tensor | None = None

    def set_cond(self, cond_emb: torch.Tensor | None):
        self._cond_emb = cond_emb
        for m in self.layers:
            if isinstance(m, ConcatLinearCondV2):
                m.set_cond(cond_emb)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(t, x)
        return x


def tMLP_cond(d_in, d_out=None, d_model=256, num_layers=6, actfn="swish",
              fourier=None, cond_dim: int = 64):
    assert num_layers > 1
    d_out = d_in if d_out is None else d_out
    act = ACTFNS[actfn]

    layers: list[nn.Module] = []
    if fourier:
        layers.append(diffeq_layers.diffeq_wrapper(
            PositionalEncoding(n_fourier_features=fourier)))
        layers.append(ConcatLinearCondV2(d_in * fourier * 2, d_model, cond_dim))
    else:
        layers.append(ConcatLinearCondV2(d_in, d_model, cond_dim))

    for _ in range(num_layers - 2):
        layers.append(act(d_model))
        layers.append(ConcatLinearCondV2(d_model, d_model, cond_dim))
    layers.append(act(d_model))
    layers.append(ConcatLinearCondV2(d_model, d_out, cond_dim))
    return SequentialDiffEqCond(*layers)


class ConditionalVecfield(nn.Module):
    """Wraps an `Unbatch(ProjectToTangent(tMLP_cond))`-style vector field.

    Forward accepts an optional `labels` tensor; `labels is None` uses the
    null-class embedding (last row of the embedding table). The class embedding
    is set on the innermost `SequentialDiffEqCond` before delegating to the
    wrapped vecfield.
    """

    def __init__(self, wrapped: nn.Module, inner_cond: SequentialDiffEqCond,
                 num_classes: int, cond_dim: int):
        super().__init__()
        self.wrapped = wrapped
        self.inner_cond = inner_cond
        self.num_classes = int(num_classes)
        self.embed = nn.Embedding(num_classes + 1, cond_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def _null_labels(self, B: int, device):
        return torch.full((B,), self.num_classes, dtype=torch.long, device=device)

    def forward(self, t: torch.Tensor, x: torch.Tensor,
                labels: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0] if x.ndim > 1 else 1
        if labels is None:
            labels = self._null_labels(B, x.device)
        else:
            labels = labels.to(device=x.device, dtype=torch.long)
            # Treat -1 as null token.
            labels = torch.where(
                labels < 0,
                torch.full_like(labels, self.num_classes),
                labels,
            )
        cond_emb = self.embed(labels)
        self.inner_cond.set_cond(cond_emb)
        try:
            return self.wrapped(t, x)
        finally:
            self.inner_cond.set_cond(None)
