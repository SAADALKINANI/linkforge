"""Mapping between string labels and the integer ids used by the models."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class TripleIndex:
    """Bidirectional vocabulary for entities and relations."""

    def __init__(self, entities: Iterable[str] = (), relations: Iterable[str] = ()):
        self.ent_to_id: dict[str, int] = {}
        self.rel_to_id: dict[str, int] = {}
        for e in entities:
            self._add(self.ent_to_id, e)
        for r in relations:
            self._add(self.rel_to_id, r)

    @staticmethod
    def _add(table: dict[str, int], key: str) -> int:
        if key not in table:
            table[key] = len(table)
        return table[key]

    @classmethod
    def from_triples(cls, *triple_sets: np.ndarray) -> "TripleIndex":
        idx = cls()
        for triples in triple_sets:
            for h, r, t in np.asarray(triples, dtype=object):
                idx._add(idx.ent_to_id, str(h))
                idx._add(idx.rel_to_id, str(r))
                idx._add(idx.ent_to_id, str(t))
        return idx

    # ----------------------------------------------------------------- props
    @property
    def entities(self) -> list[str]:
        return list(self.ent_to_id)

    @property
    def relations(self) -> list[str]:
        return list(self.rel_to_id)

    @property
    def n_entities(self) -> int:
        return len(self.ent_to_id)

    @property
    def n_relations(self) -> int:
        return len(self.rel_to_id)

    # ------------------------------------------------------------ conversion
    def entity_id(self, name: str) -> int:
        try:
            return self.ent_to_id[name]
        except KeyError:
            raise KeyError(f"Entity '{name}' was not seen during training") from None

    def relation_id(self, name: str) -> int:
        try:
            return self.rel_to_id[name]
        except KeyError:
            raise KeyError(f"Relation '{name}' was not seen during training") from None

    def to_ids(self, triples: np.ndarray) -> np.ndarray:
        """Convert ``(N, 3)`` string triples to an ``(N, 3)`` int64 array."""
        triples = np.asarray(triples, dtype=object).reshape(-1, 3)
        out = np.empty(triples.shape, dtype=np.int64)
        for i, (h, r, t) in enumerate(triples):
            out[i] = (self.entity_id(str(h)), self.relation_id(str(r)), self.entity_id(str(t)))
        return out

    def to_labels(self, ids: np.ndarray) -> np.ndarray:
        ents, rels = self.entities, self.relations
        ids = np.asarray(ids).reshape(-1, 3)
        return np.array([[ents[h], rels[r], ents[t]] for h, r, t in ids], dtype=object)

    def has_triple_labels(self, triples: np.ndarray) -> np.ndarray:
        """Boolean mask: which triples only use known entities and relations."""
        triples = np.asarray(triples, dtype=object).reshape(-1, 3)
        return np.array([str(h) in self.ent_to_id and str(r) in self.rel_to_id
                         and str(t) in self.ent_to_id for h, r, t in triples], dtype=bool)

    def to_dict(self) -> dict:
        return {"entities": self.entities, "relations": self.relations}

    @classmethod
    def from_dict(cls, d: dict) -> "TripleIndex":
        return cls(d["entities"], d["relations"])
