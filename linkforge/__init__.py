"""LinkForge: link prediction and knowledge discovery on knowledge graphs."""

__version__ = "0.1.0"

from .datasets import (  # noqa: E402
    generate_enterprise_graph,
    load_enterprise_demo,
    load_from_csv,
    save_to_csv,
    train_test_split_no_unseen,
)
from .discovery import discover_facts, find_clusters, find_duplicates, find_nearest_neighbours, project_2d  # noqa: E402
from .evaluation import compute_ranks, hits_at_n_score, mr_score, mrr_score  # noqa: E402
from .models import MODELS, ComplEx, DistMult, HolE, TransE  # noqa: E402
from .predictor import LinkPredictor  # noqa: E402

__all__ = [
    "LinkPredictor", "TransE", "DistMult", "ComplEx", "HolE", "MODELS",
    "load_from_csv", "save_to_csv", "train_test_split_no_unseen",
    "generate_enterprise_graph", "load_enterprise_demo",
    "compute_ranks", "mrr_score", "mr_score", "hits_at_n_score",
    "discover_facts", "find_nearest_neighbours", "find_duplicates", "find_clusters", "project_2d",
]
