"""End-to-end consulting scenario on the synthetic enterprise graph.

1. Train ComplEx on customers, suppliers, products, regions and transactions.
2. Measure how well it recovers held-out purchases and sourcing links.
3. Surface cross-sell opportunities and likely supplier relationships.
4. Find look-alike customers and segment them.

Run:  python examples/supply_chain_demo.py
"""

import numpy as np

from linkforge import (LinkPredictor, discover_facts, find_clusters, find_nearest_neighbours,
                       load_enterprise_demo)

data = load_enterprise_demo(seed=42)
train, valid, test = data["train"], data["valid"], data["test"]
print(f"train {len(train)} | valid {len(valid)} | test {len(test)} triples\n")

lp = LinkPredictor(model="ComplEx", k=100, epochs=100, eta=20, lr=5e-3,
                   loss="multiclass_nll", regularizer_weight=1e-4, seed=0)
lp.fit(train, X_valid=valid, early_stopping=True, eval_every=10, patience=3)

# ---------------------------------------------------------------- evaluate
res = lp.evaluate(test, filter_triples=[train, valid, test], corrupt="tail")
n_ent = lp.index.n_entities
print(f"\nPredicting the missing end of held-out links among {n_ent} entities:")
print(f"  MRR      {res['mrr']:.3f}   (random guessing ≈ {np.mean(1 / np.arange(1, n_ent + 1)):.3f})")
print(f"  Hits@10  {res['hits@10']:.3f}   (random guessing ≈ {10 / n_ent:.3f})")

# ------------------------------------------------- business questions
customers = sorted({h for h, r, _ in train if r == "operates_in"})
products = sorted({h for h, r, _ in train if r == "belongs_to"})
suppliers = sorted({h for h, r, _ in train if r == "supplies"})
industry = {h: t for h, r, t in train if r == "operates_in"}
category = {h: t for h, r, t in train if r == "belongs_to"}

print("\nTop cross-sell opportunities (customer likely to buy product it hasn't bought):")
xsell = discover_facts(lp, "buys", customers, products, top_n=8)
xsell["industry"] = xsell["head"].map(industry)
xsell["category"] = xsell["tail"].map(category)
print(xsell[["head", "industry", "tail", "category", "score"]].to_string(index=False))

c = customers[0]
print(f"\nRecommended new suppliers for {c} ({industry[c]}):")
for s, score in lp.predict_tails(c, "sources_from", top_n=5, candidates=suppliers):
    print(f"  {s:30s} {score:.3f}")

print(f"\nCustomers that look most like {c}:")
for other, sim in find_nearest_neighbours(lp, c, top_n=5, candidates=customers):
    print(f"  {other:30s} {industry[other]:15s} cos={sim:.3f}")

seg = find_clusters(lp, customers, n_clusters=8, seed=0)
seg["industry"] = seg["entity"].map(industry)
print("\nCustomer segments vs. industry (embedding clusters rediscover industries):")
print(seg.groupby("cluster")["industry"].agg(lambda s: s.value_counts().index[0] + f" ({s.value_counts().iloc[0]}/{len(s)})").to_string())

lp.save("enterprise_complex.pt")
print("\nSaved model → enterprise_complex.pt")
