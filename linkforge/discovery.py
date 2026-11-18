"""Knowledge discovery on top of a trained ``LinkPredictor``.

* :func:`discover_facts` — rank unseen triples to surface likely missing links
  (e.g. cross-sell opportunities, undisclosed supplier relationships).
* :func:`find_nearest_neighbours` — entities most similar in embedding space.
* :func:`find_duplicates` — candidate duplicate records (entity resolution).
* :func:`find_clusters` — group entities (e.g. customer segmentation).
* :func:`project_2d` — PCA coordinates for plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .predictor import LinkPredictor


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


@torch.no_grad()
def discover_facts(lp: LinkPredictor, relation: str, heads: list[str], tails: list[str],
                   top_n: int = 20, max_candidates: int = 2_000_000) -> pd.DataFrame:
    """Score every unseen ``(h, relation, t)`` for ``h in heads, t in tails``.

    Returns the ``top_n`` most plausible new facts, with their score and the
    rank of ``t`` among all candidate tails for that head (rank 1 = the
    model's top suggestion for that head).
    """
    lp._check_fitted()
    if len(heads) * len(tails) > max_candidates:
        raise ValueError("Too many candidates; narrow down heads/tails or raise max_candidates")
    h_ids = torch.tensor([lp.index.entity_id(h) for h in heads])
    t_ids = torch.tensor([lp.index.entity_id(t) for t in tails])
    r_id = lp.index.relation_id(relation)
    rows = []
    for start in range(0, len(h_ids), 256):
        hb = h_ids[start:start + 256]
        scores = lp.model.score_all_tails(hb, torch.full_like(hb, r_id), t_ids)
        order = torch.argsort(scores, dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(1, len(t_ids) + 1).expand_as(order).contiguous())
        for i, h in enumerate(hb.tolist()):
            for j, t in enumerate(t_ids.tolist()):
                if h != t and (h, r_id, t) not in lp._known:
                    rows.append((h, t, float(scores[i, j]), int(ranks[i, j])))
    ents = lp.index.entities
    df = pd.DataFrame(rows, columns=["h", "t", "score", "rank_for_head"])
    df = df.nlargest(top_n, "score").reset_index(drop=True)
    df.insert(0, "head", [ents[i] for i in df.pop("h")])
    df.insert(1, "relation", relation)
    df.insert(2, "tail", [ents[i] for i in df.pop("t")])
    return df


def find_nearest_neighbours(lp: LinkPredictor, entity: str, top_n: int = 10,
                            candidates: list[str] | None = None) -> list[tuple[str, float]]:
    """Entities with the highest cosine similarity to ``entity``."""
    pool = candidates if candidates is not None else lp.index.entities
    pool = [e for e in pool if e != entity]
    emb = _unit(lp.get_embeddings(pool))
    q = _unit(lp.get_embeddings([entity]))[0]
    sims = emb @ q
    order = np.argsort(-sims)[:top_n]
    return [(pool[i], float(sims[i])) for i in order]


def find_duplicates(lp: LinkPredictor, entities: list[str], threshold: float = 0.9) -> pd.DataFrame:
    """Pairs of entities whose embeddings have cosine similarity ``>= threshold``."""
    emb = _unit(lp.get_embeddings(entities))
    sims = emb @ emb.T
    i, j = np.where(np.triu(sims, k=1) >= threshold)
    df = pd.DataFrame({"entity_a": [entities[a] for a in i], "entity_b": [entities[b] for b in j],
                       "similarity": sims[i, j]})
    return df.sort_values("similarity", ascending=False).reset_index(drop=True)


def find_clusters(lp: LinkPredictor, entities: list[str], n_clusters: int = 5, n_init: int = 10,
                  max_iter: int = 100, seed: int = 0) -> pd.DataFrame:
    """K-means (k-means++ init, cosine geometry) over entity embeddings."""
    X = _unit(lp.get_embeddings(entities))
    rng = np.random.default_rng(seed)
    best_inertia, best_labels = np.inf, None
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, n_clusters):
            d = np.min(((X[:, None] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            centers.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
        centers = np.array(centers)
        for _ in range(max_iter):
            labels = ((X[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
            new = np.array([X[labels == c].mean(0) if (labels == c).any() else centers[c]
                            for c in range(n_clusters)])
            if np.allclose(new, centers):
                break
            centers = new
        inertia = ((X - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return pd.DataFrame({"entity": entities, "cluster": best_labels})


def project_2d(lp: LinkPredictor, entities: list[str]) -> pd.DataFrame:
    """2-D PCA projection of entity embeddings, ready to scatter-plot."""
    X = lp.get_embeddings(entities)
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    xy = X @ vt[:2].T
    return pd.DataFrame({"entity": entities, "x": xy[:, 0], "y": xy[:, 1]})
