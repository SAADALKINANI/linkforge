"""Loading, splitting and generating knowledge graphs."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def load_from_csv(path: str, sep: str = "\t", header: bool = False,
                  columns: tuple[int | str, int | str, int | str] = (0, 1, 2)) -> np.ndarray:
    """Read triples from a delimited file into an ``(N, 3)`` object array.

    ``columns`` picks the subject, predicate and object columns (by position,
    or by name when ``header=True``). Duplicate rows and rows with blanks are
    dropped.
    """
    df = pd.read_csv(path, sep=sep, header=0 if header else None, dtype=str, keep_default_na=False)
    cols = [df.columns[c] if isinstance(c, int) else c for c in columns]
    df = df[cols].apply(lambda s: s.str.strip())
    df = df[(df != "").all(axis=1)].drop_duplicates()
    return df.to_numpy(dtype=object)


def save_to_csv(triples: np.ndarray, path: str, sep: str = "\t") -> None:
    pd.DataFrame(np.asarray(triples, dtype=object)).to_csv(path, sep=sep, header=False, index=False)


def train_test_split_no_unseen(X: np.ndarray, test_size: int | float = 0.1, seed: int = 0,
                               relations: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split triples so every entity and relation in the test set also appears in training.

    Embedding models can only score entities they have an embedding for, so
    a naive random split would silently leave test triples unscorable.

    Args:
        test_size: number of triples, or a fraction of ``len(X)``.
        relations: only draw test triples from these relations (e.g. the
            business question you care about, such as ``buys``).
    """
    X = np.asarray(X, dtype=object).reshape(-1, 3)
    n_test = int(round(test_size * len(X))) if isinstance(test_size, float) else int(test_size)
    ent_count = Counter(X[:, 0].tolist() + X[:, 2].tolist())
    rel_count = Counter(X[:, 1].tolist())
    rng = np.random.default_rng(seed)
    test_idx = []
    for i in rng.permutation(len(X)):
        if len(test_idx) >= n_test:
            break
        h, r, t = X[i]
        if relations is not None and r not in relations:
            continue
        need = 2 if h == t else 1
        if ent_count[h] > need and ent_count[t] > need and rel_count[r] > 1:
            ent_count[h] -= 1
            ent_count[t] -= 1
            rel_count[r] -= 1
            test_idx.append(i)
    if len(test_idx) < n_test:
        raise ValueError(f"Could only place {len(test_idx)} of {n_test} triples in the test set "
                         "without creating unseen entities. Use a smaller test_size.")
    mask = np.zeros(len(X), dtype=bool)
    mask[test_idx] = True
    return X[~mask], X[mask]


# --------------------------------------------------------------------------
# Synthetic enterprise graph: customers, suppliers, products and transactions
# --------------------------------------------------------------------------

REGIONS = ["North America", "Latin America", "Western Europe", "Nordics", "Middle East", "APAC"]
INDUSTRY_CATEGORIES = {
    "Retail": ["Logistics", "Packaging", "Analytics Software"],
    "Banking": ["Cybersecurity", "Cloud Services", "Analytics Software"],
    "Healthcare": ["Medical Devices", "Cloud Services", "Facilities"],
    "Energy": ["Raw Materials", "Facilities", "Logistics"],
    "Telecom": ["Network Hardware", "Cloud Services", "Cybersecurity"],
    "Manufacturing": ["Raw Materials", "Packaging", "Logistics"],
    "Pharma": ["Medical Devices", "Raw Materials", "Analytics Software"],
    "Public Sector": ["Cybersecurity", "Facilities", "Network Hardware"],
}
CATEGORIES = sorted({c for cats in INDUSTRY_CATEGORIES.values() for c in cats})
_ADJ = ["Northwind", "Bluepeak", "Ironbridge", "Silverline", "Crestview", "Redwood", "Harbor",
        "Summit", "Evergreen", "Granite", "Lumen", "Atlas", "Meridian", "Orchid", "Vantage", "Kestrel"]
_NOUN = {"customer": ["Group", "Holdings", "Partners", "Corp", "Co", "Ltd"],
         "supplier": ["Supply", "Systems", "Solutions", "Industries", "Works", "Networks"]}


def _names(kind: str, n: int, rng: np.random.Generator, suffix: str = "") -> list[str]:
    combos = [f"{a} {b}" for a in _ADJ for b in _NOUN[kind]]
    rng.shuffle(combos)
    return [f"{combos[i % len(combos)]}{suffix}" + (f" {i // len(combos) + 1}" if i >= len(combos) else "")
            for i in range(n)]


def generate_enterprise_graph(n_customers: int = 300, n_suppliers: int = 80, n_products: int = 150,
                              purchases_per_customer: tuple[int, int] = (5, 10),
                              seed: int = 42) -> np.ndarray:
    """Synthesise a consulting-style knowledge graph with real latent structure.

    Customers belong to an industry and region; each industry favours a few
    product categories; suppliers specialise in categories and serve
    regions. Purchases (``buys``) and sourcing relationships
    (``sources_from``) follow those patterns with some noise, so a good model
    can recover held-out transactions from the rest of the graph.

    Relations: ``located_in``, ``operates_in``, ``belongs_to``, ``supplies``,
    ``buys``, ``sources_from``, ``competes_with``.
    """
    rng = np.random.default_rng(seed)
    triples: set[tuple[str, str, str]] = set()
    industries = list(INDUSTRY_CATEGORIES)

    # Products → categories
    products = [f"PRD-{i:04d}" for i in range(n_products)]
    prod_cat = {p: CATEGORIES[i % len(CATEGORIES)] for i, p in enumerate(products)}
    for p, c in prod_cat.items():
        triples.add((p, "belongs_to", c))
    by_cat = {c: [p for p in products if prod_cat[p] == c] for c in CATEGORIES}

    # Suppliers → region, specialised categories, products they supply
    suppliers = _names("supplier", n_suppliers, rng)
    sup_region, sup_products = {}, {}
    for i, s in enumerate(suppliers):
        sup_region[s] = REGIONS[i % len(REGIONS)]
        triples.add((s, "located_in", sup_region[s]))
        cats = rng.choice(CATEGORIES, size=rng.integers(1, 3), replace=False)
        prods = set()
        for c in cats:
            k = min(len(by_cat[c]), int(rng.integers(3, 7)))
            prods.update(rng.choice(by_cat[c], size=k, replace=False).tolist())
        sup_products[s] = prods
        for p in prods:
            triples.add((s, "supplies", p))
    # make sure every product has at least one supplier
    for p in products:
        if not any(p in ps for ps in sup_products.values()):
            s = suppliers[int(rng.integers(n_suppliers))]
            sup_products[s].add(p)
            triples.add((s, "supplies", p))
    prod_suppliers = {p: [s for s in suppliers if p in sup_products[s]] for p in products}

    # Competitors: suppliers sharing a region and at least one category
    for i, a in enumerate(suppliers):
        cats_a = {prod_cat[p] for p in sup_products[a]}
        for b in suppliers[i + 1:]:
            if sup_region[a] == sup_region[b] and cats_a & {prod_cat[p] for p in sup_products[b]}:
                triples.add((a, "competes_with", b))
                triples.add((b, "competes_with", a))

    # Customers → region, industry, purchases and sourcing
    customers = _names("customer", n_customers, rng, suffix="")
    for i, c in enumerate(customers):
        region = REGIONS[int(rng.integers(len(REGIONS)))]
        industry = industries[i % len(industries)]
        triples.add((c, "located_in", region))
        triples.add((c, "operates_in", industry))
        favourite = INDUSTRY_CATEGORIES[industry]
        for _ in range(int(rng.integers(*purchases_per_customer, endpoint=True))):
            cat = rng.choice(favourite) if rng.random() < 0.85 else rng.choice(CATEGORIES)
            p = str(rng.choice(by_cat[cat]))
            triples.add((c, "buys", p))
            local = [s for s in prod_suppliers[p] if sup_region[s] == region]
            pool = local if local and rng.random() < 0.8 else prod_suppliers[p]
            triples.add((c, "sources_from", str(rng.choice(pool))))

    return np.array(sorted(triples), dtype=object)


def load_enterprise_demo(seed: int = 42, valid_size: int = 200, test_size: int = 400,
                         target_relations: tuple[str, ...] = ("buys", "sources_from"),
                         **graph_kwargs) -> dict[str, np.ndarray]:
    """Generated enterprise graph split into ``train`` / ``valid`` / ``test``.

    Validation and test triples are drawn only from ``target_relations`` (the
    transactional links a consultant would want to predict).
    """
    X = generate_enterprise_graph(seed=seed, **graph_kwargs)
    rest, test = train_test_split_no_unseen(X, test_size, seed=seed, relations=list(target_relations))
    train, valid = train_test_split_no_unseen(rest, valid_size, seed=seed + 1, relations=list(target_relations))
    return {"train": train, "valid": valid, "test": test}
