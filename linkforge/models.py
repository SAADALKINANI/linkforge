"""Knowledge-graph embedding models.

Every model maps entities and relations to dense vectors and defines a
scoring function ``f(h, r, t)`` where a *higher* score means the triple is
*more plausible*.

All scoring functions operate on the last tensor dimension and support
broadcasting, so the same code scores a batch of triples or a batch of
queries against every entity in the graph at once.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class EmbeddingModel(nn.Module):
    """Base class for all embedding models.

    Subclasses implement :meth:`_score` which receives already-looked-up
    embedding tensors and must support broadcasting on leading dimensions.
    """

    #: Multiplier applied to ``k`` to get the stored embedding width
    #: (ComplEx stores real and imaginary parts side by side).
    width_multiplier: int = 1

    def __init__(self, n_entities: int, n_relations: int, k: int = 100, seed: int | None = None):
        super().__init__()
        if k <= 0:
            raise ValueError("Embedding size k must be positive")
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.k = k
        width = k * self.width_multiplier
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        self.ent_emb = nn.Parameter(self._init(n_entities, width, gen))
        self.rel_emb = nn.Parameter(self._init(n_relations, width, gen))

    @staticmethod
    def _init(rows: int, cols: int, gen: torch.Generator | None) -> torch.Tensor:
        bound = math.sqrt(6.0 / (rows + cols))  # Xavier/Glorot uniform
        return (torch.rand(rows, cols, generator=gen) * 2 - 1) * bound

    # ------------------------------------------------------------------ API
    def _score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """Score an ``(N, 3)`` LongTensor of ``(head, relation, tail)`` ids."""
        h = self.ent_emb[triples[:, 0]]
        r = self.rel_emb[triples[:, 1]]
        t = self.ent_emb[triples[:, 2]]
        return self._score(h, r, t)

    def score_all_tails(self, heads: torch.Tensor, rels: torch.Tensor,
                        candidates: torch.Tensor | None = None) -> torch.Tensor:
        """Score ``(head, rel, ?)`` against every entity → ``(B, n_candidates)``."""
        h = self.ent_emb[heads].unsqueeze(1)
        r = self.rel_emb[rels].unsqueeze(1)
        t = (self.ent_emb if candidates is None else self.ent_emb[candidates]).unsqueeze(0)
        return self._score(h, r, t)

    def score_all_heads(self, rels: torch.Tensor, tails: torch.Tensor,
                        candidates: torch.Tensor | None = None) -> torch.Tensor:
        """Score ``(?, rel, tail)`` against every entity → ``(B, n_candidates)``."""
        h = (self.ent_emb if candidates is None else self.ent_emb[candidates]).unsqueeze(0)
        r = self.rel_emb[rels].unsqueeze(1)
        t = self.ent_emb[tails].unsqueeze(1)
        return self._score(h, r, t)

    def regularization(self, triples: torch.Tensor, p: int = 2) -> torch.Tensor:
        """Lp penalty on the embeddings touched by ``triples``."""
        ents = torch.cat([triples[:, 0], triples[:, 2]])
        e = self.ent_emb[ents]
        r = self.rel_emb[triples[:, 1]]
        return (e.abs() ** p).sum(-1).mean() + (r.abs() ** p).sum(-1).mean()

    def post_step(self) -> None:
        """Hook run after every optimiser step (e.g. re-normalisation)."""


class TransE(EmbeddingModel):
    """Translating embeddings (Bordes et al., 2013): ``-||h + r - t||_p``."""

    def __init__(self, *args, norm: int = 1, normalize_entities: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.normalize_entities = normalize_entities
        self.post_step()

    def _score(self, h, r, t):
        return -torch.linalg.vector_norm(h + r - t, ord=self.norm, dim=-1)

    @torch.no_grad()
    def post_step(self) -> None:
        if self.normalize_entities:
            self.ent_emb.data = nn.functional.normalize(self.ent_emb.data, p=2, dim=-1)


class DistMult(EmbeddingModel):
    """Bilinear-diagonal model (Yang et al., 2015): ``<h, r, t>``.

    Symmetric by construction, so it cannot tell ``(a, r, b)`` from ``(b, r, a)``.
    """

    def _score(self, h, r, t):
        return (h * r * t).sum(-1)


class ComplEx(EmbeddingModel):
    """Complex embeddings (Trouillon et al., 2016): ``Re(<h, r, conj(t)>)``.

    Handles symmetric *and* antisymmetric relations. Stores ``2k`` real
    numbers per vector (real part followed by imaginary part).
    """

    width_multiplier = 2

    def _score(self, h, r, t):
        hr, hi = h.chunk(2, dim=-1)
        rr, ri = r.chunk(2, dim=-1)
        tr, ti = t.chunk(2, dim=-1)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)


class HolE(EmbeddingModel):
    """Holographic embeddings (Nickel et al., 2016): ``r · (h ⋆ t)``.

    ``⋆`` is circular correlation, computed in O(k log k) with the FFT.
    """

    def _score(self, h, r, t):
        h, t = torch.broadcast_tensors(h, t)
        corr = torch.fft.irfft(torch.conj(torch.fft.rfft(h, dim=-1)) * torch.fft.rfft(t, dim=-1),
                               n=h.shape[-1], dim=-1)
        return (r * corr).sum(-1)


MODELS: dict[str, type[EmbeddingModel]] = {
    "TransE": TransE,
    "DistMult": DistMult,
    "ComplEx": ComplEx,
    "HolE": HolE,
}


def get_model_class(name: str) -> type[EmbeddingModel]:
    for key, cls in MODELS.items():
        if key.lower() == name.lower():
            return cls
    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODELS)}")
