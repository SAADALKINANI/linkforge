"""High-level, scikit-learn-style API: ``LinkPredictor``."""

from __future__ import annotations

import copy
import time
from collections import defaultdict

import numpy as np
import torch

from . import __version__
from .evaluation import compute_ranks, summarize
from .losses import get_loss
from .mapping import TripleIndex
from .models import EmbeddingModel, get_model_class


class NegativeSampler:
    """Generates corruptions of positive triples.

    ``uniform`` replaces the head or tail with any entity. ``typed`` (local
    closed-world) only uses entities seen in that position for the same
    relation, so a *customer* is never swapped for a *region*. Typed
    negatives are harder and usually train better models on business graphs
    with many entity types.
    """

    def __init__(self, train_ids: np.ndarray, n_entities: int, strategy: str = "uniform",
                 generator: torch.Generator | None = None):
        if strategy not in {"uniform", "typed"}:
            raise ValueError("strategy must be 'uniform' or 'typed'")
        self.n_entities = n_entities
        self.strategy = strategy
        self.gen = generator
        if strategy == "typed":
            heads, tails = defaultdict(set), defaultdict(set)
            for h, r, t in train_ids:
                heads[int(r)].add(int(h))
                tails[int(r)].add(int(t))
            self.heads = {r: torch.tensor(sorted(v)) for r, v in heads.items()}
            self.tails = {r: torch.tensor(sorted(v)) for r, v in tails.items()}

    def _random_entities(self, rels: torch.Tensor, eta: int, pool: dict) -> torch.Tensor:
        if self.strategy == "uniform":
            return torch.randint(self.n_entities, (len(rels), eta), generator=self.gen)
        out = torch.empty(len(rels), eta, dtype=torch.long)
        for r in rels.unique().tolist():
            rows = (rels == r).nonzero(as_tuple=True)[0]
            cands = pool[r]
            idx = torch.randint(len(cands), (len(rows), eta), generator=self.gen)
            out[rows] = cands[idx]
        return out

    def corrupt(self, batch: torch.Tensor, eta: int) -> torch.Tensor:
        """Return ``(B, eta, 3)`` corruptions of ``batch`` (``(B, 3)``)."""
        rels = batch[:, 1]
        neg = batch.unsqueeze(1).repeat(1, eta, 1)
        replace_head = torch.rand(len(batch), eta, generator=self.gen) < 0.5
        new_heads = self._random_entities(rels, eta, getattr(self, "heads", {}))
        new_tails = self._random_entities(rels, eta, getattr(self, "tails", {}))
        neg[..., 0] = torch.where(replace_head, new_heads, neg[..., 0])
        neg[..., 2] = torch.where(replace_head, neg[..., 2], new_tails)
        return neg


class LinkPredictor:
    """Train a knowledge-graph embedding model and use it to predict links.

    Example:
        >>> lp = LinkPredictor(model="ComplEx", k=100, epochs=100)
        >>> lp.fit(train_triples)                       # (N, 3) array of strings
        >>> lp.evaluate(test_triples, filter_triples=[train_triples, test_triples])
        >>> lp.predict_tails("Acme Corp", "buys_from", top_n=5)
    """

    def __init__(self, model: str = "ComplEx", k: int = 100, eta: int = 20, epochs: int = 100,
                 batch_size: int = 512, lr: float = 1e-3, loss: str = "multiclass_nll",
                 loss_params: dict | None = None, regularizer_weight: float = 0.0,
                 regularizer_p: int = 2, model_params: dict | None = None,
                 negative_sampling: str = "uniform", seed: int = 0, verbose: bool = True):
        self.config = dict(model=model, k=k, eta=eta, epochs=epochs, batch_size=batch_size, lr=lr,
                           loss=loss, loss_params=loss_params or {},
                           regularizer_weight=regularizer_weight, regularizer_p=regularizer_p,
                           model_params=model_params or {}, negative_sampling=negative_sampling,
                           seed=seed)
        get_model_class(model)  # fail fast on typos
        get_loss(loss)
        self.verbose = verbose
        self.index: TripleIndex | None = None
        self.model: EmbeddingModel | None = None
        self.history: list[dict] = []
        self._known: set[tuple[int, int, int]] = set()

    # ------------------------------------------------------------- training
    def fit(self, X, X_valid=None, early_stopping: bool = False, patience: int = 3,
            eval_every: int = 10, valid_filter: list | None = None) -> "LinkPredictor":
        """Learn embeddings from ``X``, an ``(N, 3)`` array of string triples.

        With ``early_stopping=True`` the filtered validation MRR on ``X_valid``
        is checked every ``eval_every`` epochs; training stops after
        ``patience`` checks without improvement and the best weights are kept.
        """
        cfg = self.config
        X = np.asarray(X, dtype=object).reshape(-1, 3)
        torch.manual_seed(cfg["seed"])
        gen = torch.Generator().manual_seed(cfg["seed"])

        self.index = TripleIndex.from_triples(X)
        train_ids = self.index.to_ids(X)
        self._known = set(map(tuple, train_ids.tolist()))
        model_cls = get_model_class(cfg["model"])
        self.model = model_cls(self.index.n_entities, self.index.n_relations, k=cfg["k"],
                               seed=cfg["seed"], **cfg["model_params"])
        loss_fn = get_loss(cfg["loss"])
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"])
        sampler = NegativeSampler(train_ids, self.index.n_entities, cfg["negative_sampling"], gen)

        valid_ids = valid_filters = None
        if X_valid is not None:
            X_valid = np.asarray(X_valid, dtype=object).reshape(-1, 3)
            keep = self.index.has_triple_labels(X_valid)
            if not keep.all() and self.verbose:
                print(f"[linkforge] dropping {(~keep).sum()} validation triples with unseen entities")
            valid_ids = self.index.to_ids(X_valid[keep])
            extra = [self._ids_known(f) for f in (valid_filter or [])]
            valid_filters = [train_ids, valid_ids, *extra]

        data = torch.as_tensor(train_ids, dtype=torch.long)
        best_mrr, best_state, bad_checks = -1.0, None, 0
        self.history = []
        for epoch in range(1, cfg["epochs"] + 1):
            self.model.train()
            t0, total = time.time(), 0.0
            perm = torch.randperm(len(data), generator=gen)
            for start in range(0, len(data), cfg["batch_size"]):
                batch = data[perm[start:start + cfg["batch_size"]]]
                neg = sampler.corrupt(batch, cfg["eta"])
                pos_s = self.model(batch)
                neg_s = self.model(neg.reshape(-1, 3)).reshape(len(batch), cfg["eta"])
                loss = loss_fn(pos_s, neg_s, **cfg["loss_params"])
                if cfg["regularizer_weight"] > 0:
                    loss = loss + cfg["regularizer_weight"] * self.model.regularization(batch, cfg["regularizer_p"])
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.model.post_step()
                total += loss.item() * len(batch)
            record = {"epoch": epoch, "loss": total / len(data), "seconds": time.time() - t0}

            if valid_ids is not None and len(valid_ids) and (epoch % eval_every == 0 or epoch == cfg["epochs"]):
                ranks = compute_ranks(self.model, valid_ids, filter_ids=valid_filters)
                record["valid_mrr"] = summarize(ranks)["mrr"]
                if record["valid_mrr"] > best_mrr:
                    best_mrr, bad_checks = record["valid_mrr"], 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    bad_checks += 1
            self.history.append(record)
            if self.verbose and (epoch == 1 or epoch % max(1, cfg["epochs"] // 10) == 0 or "valid_mrr" in record):
                msg = f"[linkforge] epoch {epoch:4d}  loss {record['loss']:.4f}"
                if "valid_mrr" in record:
                    msg += f"  valid MRR {record['valid_mrr']:.4f}"
                print(msg)
            if early_stopping and bad_checks >= patience:
                if self.verbose:
                    print(f"[linkforge] early stopping at epoch {epoch} (best valid MRR {best_mrr:.4f})")
                break
        if early_stopping and best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    # ----------------------------------------------------------- inference
    def _check_fitted(self):
        if self.model is None or self.index is None:
            raise RuntimeError("Model is not fitted yet. Call fit() or LinkPredictor.load().")

    def _ids_known(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=object).reshape(-1, 3)
        return self.index.to_ids(X[self.index.has_triple_labels(X)])

    @torch.no_grad()
    def score(self, X) -> np.ndarray:
        """Raw plausibility scores for ``(N, 3)`` string triples (higher = more likely)."""
        self._check_fitted()
        ids = torch.as_tensor(self.index.to_ids(X), dtype=torch.long)
        return self.model(ids).numpy()

    def evaluate(self, X_test, filter_triples: list | None = None, corrupt: str = "both",
                 strategy: str = "realistic") -> dict:
        """Filtered link-prediction metrics on ``X_test``.

        ``filter_triples`` is a list of arrays of all known true facts (train,
        valid, test). Test triples with entities unseen in training are skipped.
        Returns a dict with ``mrr``, ``mr``, ``hits@1/3/10``, ``n`` and ``ranks``.
        """
        self._check_fitted()
        X_test = np.asarray(X_test, dtype=object).reshape(-1, 3)
        keep = self.index.has_triple_labels(X_test)
        test_ids = self.index.to_ids(X_test[keep])
        filters = [self._ids_known(f) for f in filter_triples] if filter_triples else None
        ranks = compute_ranks(self.model, test_ids, filter_ids=filters, corrupt=corrupt, strategy=strategy)
        out = summarize(ranks)
        out.update(n=int(keep.sum()), skipped=int((~keep).sum()), ranks=ranks)
        return out

    @torch.no_grad()
    def predict_tails(self, head: str, relation: str, top_n: int = 10, exclude_known: bool = True,
                      candidates: list[str] | None = None) -> list[tuple[str, float]]:
        """Most likely ``t`` for the query ``(head, relation, ?)``."""
        return self._predict(head, relation, top_n, exclude_known, candidates, side="tail")

    @torch.no_grad()
    def predict_heads(self, relation: str, tail: str, top_n: int = 10, exclude_known: bool = True,
                      candidates: list[str] | None = None) -> list[tuple[str, float]]:
        """Most likely ``h`` for the query ``(?, relation, tail)``."""
        return self._predict(tail, relation, top_n, exclude_known, candidates, side="head")

    def _predict(self, anchor, relation, top_n, exclude_known, candidates, side):
        self._check_fitted()
        a = torch.tensor([self.index.entity_id(anchor)])
        r = torch.tensor([self.index.relation_id(relation)])
        if candidates is None:
            cand = torch.arange(self.index.n_entities)
        else:
            cand = torch.tensor([self.index.entity_id(c) for c in candidates])
        if side == "tail":
            scores = self.model.score_all_tails(a, r, cand)[0]
        else:
            scores = self.model.score_all_heads(r, a, cand)[0]
        ents = self.index.entities
        results = []
        for i in torch.argsort(scores, descending=True).tolist():
            e = int(cand[i])
            triple = (int(a), int(r), e) if side == "tail" else (e, int(r), int(a))
            if e == int(a) or (exclude_known and triple in self._known):
                continue
            results.append((ents[e], float(scores[i])))
            if len(results) == top_n:
                break
        return results

    def get_embeddings(self, names: list[str], kind: str = "entity") -> np.ndarray:
        """Embedding vectors for entity or relation names."""
        self._check_fitted()
        if kind == "entity":
            ids = [self.index.entity_id(n) for n in names]
            return self.model.ent_emb.detach()[ids].numpy()
        if kind == "relation":
            ids = [self.index.relation_id(n) for n in names]
            return self.model.rel_emb.detach()[ids].numpy()
        raise ValueError("kind must be 'entity' or 'relation'")

    def is_known(self, head: str, relation: str, tail: str) -> bool:
        """True if the triple was part of the training data."""
        self._check_fitted()
        try:
            return tuple(self.index.to_ids([[head, relation, tail]])[0].tolist()) in self._known
        except KeyError:
            return False

    # -------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        self._check_fitted()
        torch.save({
            "linkforge_version": __version__,
            "config": self.config,
            "index": self.index.to_dict(),
            "state_dict": self.model.state_dict(),
            "known": torch.tensor(sorted(self._known), dtype=torch.long).reshape(-1, 3),
            "history": self.history,
        }, path)

    @classmethod
    def load(cls, path: str, verbose: bool = True) -> "LinkPredictor":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        lp = cls(**blob["config"], verbose=verbose)
        lp.index = TripleIndex.from_dict(blob["index"])
        model_cls = get_model_class(lp.config["model"])
        lp.model = model_cls(lp.index.n_entities, lp.index.n_relations, k=lp.config["k"],
                             **lp.config["model_params"])
        lp.model.load_state_dict(blob["state_dict"])
        lp.model.eval()
        lp._known = set(map(tuple, blob["known"].tolist()))
        lp.history = blob.get("history", [])
        return lp
