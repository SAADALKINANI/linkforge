import numpy as np
import pytest
import torch

from linkforge import (LinkPredictor, MODELS, compute_ranks, discover_facts, find_clusters,
                       find_nearest_neighbours, generate_enterprise_graph, hits_at_n_score,
                       load_from_csv, mrr_score, save_to_csv, train_test_split_no_unseen)
from linkforge.cli import main as cli_main
from linkforge.losses import LOSSES


@pytest.fixture(scope="module")
def graph():
    return generate_enterprise_graph(n_customers=60, n_suppliers=20, n_products=40, seed=1)


@pytest.mark.parametrize("name", list(MODELS))
def test_score_all_matches_pointwise(name):
    m = MODELS[name](12, 3, k=8, seed=0)
    h, r = torch.tensor([0, 5]), torch.tensor([1, 2])
    all_t = m.score_all_tails(h, r)
    all_h = m.score_all_heads(r, h)
    for b in range(2):
        tails = torch.stack([h[b].repeat(12), r[b].repeat(12), torch.arange(12)], 1)
        heads = torch.stack([torch.arange(12), r[b].repeat(12), h[b].repeat(12)], 1)
        assert torch.allclose(all_t[b], m(tails), atol=1e-5)
        assert torch.allclose(all_h[b], m(heads), atol=1e-5)


def test_distmult_symmetric_complex_not():
    d, c = MODELS["DistMult"](4, 1, k=6, seed=0), MODELS["ComplEx"](4, 1, k=6, seed=0)
    fwd, bwd = torch.tensor([[0, 0, 1]]), torch.tensor([[1, 0, 0]])
    assert torch.allclose(d(fwd), d(bwd))
    assert not torch.allclose(c(fwd), c(bwd))


@pytest.mark.parametrize("loss", list(LOSSES))
def test_losses_finite_and_prefer_correct_order(loss):
    good = LOSSES[loss](torch.tensor([5.0, 5.0]), torch.full((2, 4), -5.0))
    bad = LOSSES[loss](torch.tensor([-5.0, -5.0]), torch.full((2, 4), 5.0))
    assert torch.isfinite(good) and good < bad


def test_metrics():
    ranks = np.array([1, 2, 4, 10])
    assert mrr_score(ranks) == pytest.approx((1 + 0.5 + 0.25 + 0.1) / 4)
    assert hits_at_n_score(ranks, 3) == 0.5


def test_filtered_ranking_removes_known_facts():
    m = MODELS["DistMult"](5, 1, k=4, seed=0)
    with torch.no_grad():
        m.ent_emb.zero_()
        m.rel_emb.fill_(1.0)
        m.ent_emb[:, 0] = torch.tensor([1.0, 5.0, 4.0, 3.0, 0.0])
    test = np.array([[0, 0, 3]])  # entities 1 and 2 score higher than 3
    raw = compute_ranks(m, test, corrupt="tail")
    filt = compute_ranks(m, test, filter_ids=[np.array([[0, 0, 1], [0, 0, 2]]), test], corrupt="tail")
    assert raw[0] == 3 and filt[0] == 1


def test_split_has_no_unseen(graph):
    train, test = train_test_split_no_unseen(graph, 50, seed=0, relations=["buys"])
    assert len(test) == 50 and set(test[:, 1]) == {"buys"}
    seen = set(train[:, 0]) | set(train[:, 2])
    assert set(test[:, 0]) <= seen and set(test[:, 2]) <= seen


def test_train_beats_random_and_roundtrip(graph, tmp_path):
    train, test = train_test_split_no_unseen(graph, 40, seed=0, relations=["buys"])
    lp = LinkPredictor(model="ComplEx", k=32, epochs=40, lr=5e-3, verbose=False)
    lp.fit(train, X_valid=test, early_stopping=True, eval_every=10)
    res = lp.evaluate(test, filter_triples=[train, test], corrupt="tail")
    n_ent = lp.index.n_entities
    random_mrr = np.mean(1 / np.arange(1, n_ent + 1))
    assert res["mrr"] > 3 * random_mrr

    path = tmp_path / "m.pt"
    lp.save(str(path))
    lp2 = LinkPredictor.load(str(path), verbose=False)
    np.testing.assert_allclose(lp.score(test[:5]), lp2.score(test[:5]), rtol=1e-6)
    head, rel, _ = test[0]
    preds = lp2.predict_tails(head, rel, top_n=5)
    assert len(preds) == 5 and all(not lp2.is_known(head, rel, t) for t, _ in preds)


def test_discovery(graph):
    lp = LinkPredictor(model="DistMult", k=16, epochs=5, verbose=False).fit(graph)
    customers = sorted({h for h, r, _ in graph if r == "operates_in"})
    products = sorted({t for _, r, t in graph if r == "buys"})
    df = discover_facts(lp, "buys", customers[:10], products, top_n=7)
    assert len(df) == 7 and list(df.columns[:3]) == ["head", "relation", "tail"]
    assert not any(lp.is_known(h, "buys", t) for h, t in zip(df["head"], df["tail"]))
    assert len(find_nearest_neighbours(lp, customers[0], top_n=3)) == 3
    assert find_clusters(lp, customers, n_clusters=3)["cluster"].nunique() <= 3


def test_csv_roundtrip_and_cli(graph, tmp_path, capsys):
    train, test = train_test_split_no_unseen(graph, 30, seed=0, relations=["buys"])
    save_to_csv(train, tmp_path / "train.tsv")
    save_to_csv(test, tmp_path / "test.tsv")
    assert len(load_from_csv(tmp_path / "train.tsv")) == len(train)
    model = str(tmp_path / "m.pt")
    cli_main(["train", "--train", str(tmp_path / "train.tsv"), "--out", model, "--epochs", "3", "--k", "8"])
    cli_main(["evaluate", "--model", model, "--test", str(tmp_path / "test.tsv"),
              "--filter", str(tmp_path / "train.tsv")])
    cli_main(["predict", "--model", model, "--head", test[0][0], "--relation", "buys", "--top", "3"])
    out = capsys.readouterr().out
    assert '"mrr"' in out and "Top 3" in out
