"""Link-prediction evaluation: filtered ranks, MRR, MR and Hits@N.

The standard protocol (Bordes et al., 2013): for every test triple
``(h, r, t)`` the model scores every corruption ``(h, r, e)`` and
``(e, r, t)``. The rank of the true triple among them is recorded. In the
*filtered* setting, corruptions that are themselves known true facts are
removed first, so the model isn't punished for ranking another real fact
higher.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .models import EmbeddingModel


def build_filter(*id_triple_sets: np.ndarray) -> tuple[dict, dict]:
    """Index known true triples for filtered evaluation.

    Returns ``(tails, heads)`` where ``tails[(h, r)]`` and ``heads[(r, t)]``
    are LongTensors of known entity ids.
    """
    tails: dict[tuple[int, int], set[int]] = defaultdict(set)
    heads: dict[tuple[int, int], set[int]] = defaultdict(set)
    for triples in id_triple_sets:
        for h, r, t in np.asarray(triples).reshape(-1, 3):
            tails[(int(h), int(r))].add(int(t))
            heads[(int(r), int(t))].add(int(h))
    to_t = lambda d: {k: torch.tensor(sorted(v), dtype=torch.long) for k, v in d.items()}  # noqa: E731
    return to_t(tails), to_t(heads)


def _ranks_from_scores(scores: torch.Tensor, true_ids: torch.Tensor, filters: list, strategy: str):
    true_scores = scores.gather(1, true_ids.unsqueeze(1))
    if filters is not None:
        for b, known in enumerate(filters):
            if known is not None and len(known):
                scores[b, known] = float("-inf")
        scores.scatter_(1, true_ids.unsqueeze(1), true_scores)
    greater = (scores > true_scores).sum(1).float()
    equal = (scores == true_scores).sum(1).float() - 1  # exclude the true triple itself
    if strategy == "optimistic":
        return 1 + greater
    if strategy == "pessimistic":
        return 1 + greater + equal
    if strategy == "realistic":
        return 1 + greater + equal / 2
    raise ValueError("strategy must be 'optimistic', 'pessimistic' or 'realistic'")


@torch.no_grad()
def compute_ranks(model: EmbeddingModel, test_ids: np.ndarray, filter_ids: list[np.ndarray] | None = None,
                  corrupt: str = "both", batch_size: int = 256, strategy: str = "realistic") -> np.ndarray:
    """Rank each test triple against its corruptions.

    Args:
        corrupt: ``"tail"``, ``"head"`` or ``"both"``.
        filter_ids: id triple arrays of all known true facts (usually train,
            valid and test). ``None`` gives the raw (unfiltered) protocol.

    Returns:
        ``(N,)`` array for a single side, ``(N, 2)`` ``[head_rank, tail_rank]``
        for ``"both"``.
    """
    if corrupt not in {"head", "tail", "both"}:
        raise ValueError("corrupt must be 'head', 'tail' or 'both'")
    model.eval()
    test = torch.as_tensor(np.asarray(test_ids).reshape(-1, 3), dtype=torch.long)
    tails_f, heads_f = build_filter(*filter_ids) if filter_ids else (None, None)
    head_ranks, tail_ranks = [], []
    for start in range(0, len(test), batch_size):
        batch = test[start:start + batch_size]
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        if corrupt in {"tail", "both"}:
            scores = model.score_all_tails(h, r)
            filt = None if tails_f is None else [tails_f.get((int(a), int(b))) for a, b in zip(h, r)]
            tail_ranks.append(_ranks_from_scores(scores, t, filt, strategy))
        if corrupt in {"head", "both"}:
            scores = model.score_all_heads(r, t)
            filt = None if heads_f is None else [heads_f.get((int(a), int(b))) for a, b in zip(r, t)]
            head_ranks.append(_ranks_from_scores(scores, h, filt, strategy))
    if corrupt == "tail":
        return torch.cat(tail_ranks).numpy()
    if corrupt == "head":
        return torch.cat(head_ranks).numpy()
    return torch.stack([torch.cat(head_ranks), torch.cat(tail_ranks)], dim=1).numpy()


def mrr_score(ranks: np.ndarray) -> float:
    """Mean reciprocal rank (higher is better, max 1)."""
    return float(np.mean(1.0 / np.asarray(ranks, dtype=float)))


def mr_score(ranks: np.ndarray) -> float:
    """Mean rank (lower is better, min 1)."""
    return float(np.mean(np.asarray(ranks, dtype=float)))


def hits_at_n_score(ranks: np.ndarray, n: int) -> float:
    """Share of ranks that are ``<= n``."""
    return float(np.mean(np.asarray(ranks) <= n))


def summarize(ranks: np.ndarray) -> dict[str, float]:
    return {
        "mrr": mrr_score(ranks),
        "mr": mr_score(ranks),
        "hits@1": hits_at_n_score(ranks, 1),
        "hits@3": hits_at_n_score(ranks, 3),
        "hits@10": hits_at_n_score(ranks, 10),
    }
