"""Command-line interface: ``linkforge <command> ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .datasets import load_enterprise_demo, load_from_csv, save_to_csv
from .models import MODELS
from .predictor import LinkPredictor


def _cmd_demo_data(a):
    os.makedirs(a.out, exist_ok=True)
    for split, X in load_enterprise_demo(seed=a.seed).items():
        path = os.path.join(a.out, f"{split}.tsv")
        save_to_csv(X, path)
        print(f"wrote {len(X):6d} triples → {path}")


def _cmd_train(a):
    X = load_from_csv(a.train, sep=a.sep)
    X_valid = load_from_csv(a.valid, sep=a.sep) if a.valid else None
    lp = LinkPredictor(model=a.model, k=a.k, eta=a.eta, epochs=a.epochs, batch_size=a.batch_size,
                       lr=a.lr, loss=a.loss, regularizer_weight=a.reg, negative_sampling=a.negatives,
                       seed=a.seed)
    lp.fit(X, X_valid=X_valid, early_stopping=a.early_stopping and X_valid is not None,
           eval_every=a.eval_every)
    lp.save(a.out)
    print(f"saved model → {a.out}")


def _cmd_evaluate(a):
    lp = LinkPredictor.load(a.model, verbose=False)
    X_test = load_from_csv(a.test, sep=a.sep)
    filters = [load_from_csv(f, sep=a.sep) for f in a.filter] + [X_test] if a.filter else None
    res = lp.evaluate(X_test, filter_triples=filters, corrupt=a.corrupt)
    res.pop("ranks")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, indent=2))


def _cmd_predict(a):
    lp = LinkPredictor.load(a.model, verbose=False)
    if (a.head is None) == (a.tail is None):
        sys.exit("Give exactly one of --head or --tail")
    if a.head is not None:
        preds = lp.predict_tails(a.head, a.relation, top_n=a.top)
        print(f"Top {a.top} for ({a.head}, {a.relation}, ?)")
    else:
        preds = lp.predict_heads(a.relation, a.tail, top_n=a.top)
        print(f"Top {a.top} for (?, {a.relation}, {a.tail})")
    for i, (e, s) in enumerate(preds, 1):
        print(f"{i:3d}. {e:40s} {s: .4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="linkforge", description="Link prediction for knowledge graphs")
    p.add_argument("--version", action="version", version=f"linkforge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo-data", help="write the synthetic enterprise graph as TSV files")
    d.add_argument("--out", default="data/enterprise")
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=_cmd_demo_data)

    t = sub.add_parser("train", help="train a model on a TSV of triples")
    t.add_argument("--train", required=True)
    t.add_argument("--valid")
    t.add_argument("--out", default="model.pt")
    t.add_argument("--model", default="ComplEx", choices=list(MODELS))
    t.add_argument("--k", type=int, default=100)
    t.add_argument("--eta", type=int, default=20)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--loss", default="multiclass_nll",
                   choices=["pairwise", "nll", "multiclass_nll", "self_adversarial"])
    t.add_argument("--reg", type=float, default=0.0, help="L2 regularisation weight")
    t.add_argument("--negatives", default="uniform", choices=["uniform", "typed"])
    t.add_argument("--early-stopping", action="store_true")
    t.add_argument("--eval-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("evaluate", help="filtered MRR / Hits@N on a test TSV")
    e.add_argument("--model", required=True)
    e.add_argument("--test", required=True)
    e.add_argument("--filter", nargs="*", default=[], help="TSVs of known true triples (train, valid)")
    e.add_argument("--corrupt", default="both", choices=["head", "tail", "both"])
    e.set_defaults(func=_cmd_evaluate)

    q = sub.add_parser("predict", help="top-N completions for (head, relation, ?) or (?, relation, tail)")
    q.add_argument("--model", required=True)
    q.add_argument("--relation", required=True)
    q.add_argument("--head")
    q.add_argument("--tail")
    q.add_argument("--top", type=int, default=10)
    q.set_defaults(func=_cmd_predict)

    for sp in (t, e):
        sp.add_argument("--sep", default="\t")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
