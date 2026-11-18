# LinkForge

**Link prediction and knowledge discovery on knowledge graphs, in PyTorch.**

LinkForge learns vector embeddings of the entities and relations in a knowledge graph. It then uses them to predict links that are missing from the graph. Typical uses are which customer is likely to buy which product, which supplier a company probably sources from, or which records are duplicates of each other.

It is inspired by [Accenture's AmpliGraph](https://github.com/Accenture/AmpliGraph). LinkForge is a smaller, independent implementation built on PyTorch with a scikit-learn-style API.

```python
from linkforge import LinkPredictor, load_enterprise_demo

data = load_enterprise_demo()                       # customers, suppliers, products, transactions
lp = LinkPredictor(model="ComplEx", k=100, epochs=100, lr=5e-3)
lp.fit(data["train"], X_valid=data["valid"], early_stopping=True)

lp.evaluate(data["test"], filter_triples=[data["train"], data["valid"], data["test"]])
# {'mrr': 0.16, 'hits@10': 0.42, ...}   random guessing: mrr 0.012, hits@10 0.018

lp.predict_tails("Atlas Co", "sources_from", top_n=5)
# [('Summit Supply', 5.15), ('Redwood Solutions', 4.73), ...]
```

## Features

| Area | What's included |
|---|---|
| **Models** | `TransE`, `DistMult`, `ComplEx`, `HolE` (FFT circular correlation) |
| **Losses** | pairwise margin, binary NLL, multiclass NLL, self-adversarial (RotatE-style) |
| **Training** | negative sampling (uniform or type-constrained), Adam, Lp regularisation, early stopping on validation MRR with best-weight restore |
| **Evaluation** | standard *filtered* protocol, with head/tail/both corruption and optimistic/pessimistic/realistic tie handling. Reports MRR, MR, Hits@1/3/10 |
| **Discovery** | `discover_facts` (rank unseen triples), `find_nearest_neighbours`, `find_duplicates`, `find_clusters` (k-means++), `project_2d` (PCA) |
| **Data** | TSV/CSV loading, a train/test split that guarantees no unseen entities, and a synthetic enterprise graph generator |
| **Tooling** | `linkforge` CLI, save/load to a single `.pt` file, pytest suite |

## Install

```bash
git clone https://github.com/SAADALKINANI/linkforge.git
cd linkforge
pip install -e ".[dev]"
pytest
```

## Data format

A knowledge graph is a list of `(subject, predicate, object)` triples, one per line, tab-separated:

```
Atlas Co        operates_in   Energy
Atlas Co        buys          PRD-0030
Summit Supply   supplies      PRD-0030
```

## Command line

```bash
linkforge demo-data --out data/enterprise
linkforge train    --train data/enterprise/train.tsv --valid data/enterprise/valid.tsv \
                   --model ComplEx --epochs 100 --lr 0.005 --early-stopping --out model.pt
linkforge evaluate --model model.pt --test data/enterprise/test.tsv \
                   --filter data/enterprise/train.tsv data/enterprise/valid.tsv --corrupt tail
linkforge predict  --model model.pt --head "Atlas Co" --relation buys --top 10
```

## The consulting demo

`examples/supply_chain_demo.py` runs a full scenario on a generated graph. The graph has 300 customers in 8 industries and 6 regions, 80 suppliers, 150 products in 9 categories, and about 5,600 triples. Held-out purchases and sourcing links are the test set. The script does the following:

- It recovers held-out links with about 13× the MRR of random guessing.
- It lists cross-sell opportunities: unseen `buys` links, which mostly fall in each industry's favoured categories.
- It recommends suppliers for a given customer.
- It segments customers. The embedding clusters rediscover the 8 industries with roughly 90–100% purity, without ever being told which features matter.

```bash
python examples/supply_chain_demo.py
```

## Choosing a model

| Model | Score | Good at | Limitation |
|---|---|---|---|
| TransE | `-‖h + r − t‖` | hierarchies, 1-to-1 links | struggles with 1-to-N / N-to-N |
| DistMult | `⟨h, r, t⟩` | fast, strong baseline | symmetric only: can't tell `a→b` from `b→a` |
| ComplEx | `Re⟨h, r, t̄⟩` | symmetric **and** directed relations | 2× memory |
| HolE | `r · (h ⋆ t)` | compact and expressive | equivalent to ComplEx, slower |

On the demo graph, ComplEx and TransE do best. Use `negative_sampling="uniform"` (the default) when you evaluate against all entities. `"typed"` negatives only contrast entities of the same role, so the model never learns that a region can't be a product. That hurts ranking against the full entity set.

## Project layout

```
linkforge/
  models.py      TransE, DistMult, ComplEx, HolE (broadcast-friendly scoring)
  losses.py      pairwise, nll, multiclass_nll, self_adversarial
  predictor.py   LinkPredictor: fit / evaluate / predict / save / load, NegativeSampler
  evaluation.py  filtered ranking, MRR, MR, Hits@N
  discovery.py   discover_facts, neighbours, duplicates, clusters, PCA
  datasets.py    CSV I/O, no-unseen split, synthetic enterprise graph
  mapping.py     string ↔ id vocabulary
  cli.py         command-line interface
examples/        end-to-end consulting scenario
tests/           pytest suite
```

## References

- Bordes et al. (2013). *Translating Embeddings for Modeling Multi-relational Data* (TransE).
- Yang et al. (2015). *Embedding Entities and Relations for Learning and Inference in Knowledge Bases* (DistMult).
- Trouillon et al. (2016). *Complex Embeddings for Simple Link Prediction* (ComplEx).
- Nickel et al. (2016). *Holographic Embeddings of Knowledge Graphs* (HolE).
- Sun et al. (2019). *RotatE* (self-adversarial negative sampling).

## License

Apache 2.0. See [LICENSE](LICENSE).
