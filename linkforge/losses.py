"""Loss functions for training with negative sampling.

Each loss receives ``pos`` with shape ``(B,)`` and ``neg`` with shape
``(B, eta)``: the scores of positive triples and of the ``eta`` corruptions
generated for each of them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_loss(pos: torch.Tensor, neg: torch.Tensor, margin: float = 1.0, **_) -> torch.Tensor:
    """Margin ranking loss: ``max(0, margin - f(pos) + f(neg))``."""
    return F.relu(margin - pos.unsqueeze(1) + neg).mean()


def nll_loss(pos: torch.Tensor, neg: torch.Tensor, **_) -> torch.Tensor:
    """Binary log-loss: positives labelled 1, corruptions labelled 0."""
    return F.softplus(-pos).mean() + F.softplus(neg).mean()


def multiclass_nll_loss(pos: torch.Tensor, neg: torch.Tensor, **_) -> torch.Tensor:
    """Softmax cross-entropy where the positive must win against its corruptions."""
    logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
    return (torch.logsumexp(logits, dim=1) - pos).mean()


def self_adversarial_loss(pos: torch.Tensor, neg: torch.Tensor, margin: float = 3.0,
                          alpha: float = 0.5, **_) -> torch.Tensor:
    """Self-adversarial negative sampling loss (Sun et al., 2019, RotatE).

    Harder negatives (higher scores) get more weight; weights are not
    back-propagated through.
    """
    weights = torch.softmax(alpha * neg, dim=1).detach()
    pos_term = -F.logsigmoid(margin + pos)
    neg_term = -(weights * F.logsigmoid(-neg - margin)).sum(1)
    return (pos_term + neg_term).mean()


LOSSES = {
    "pairwise": pairwise_loss,
    "nll": nll_loss,
    "multiclass_nll": multiclass_nll_loss,
    "self_adversarial": self_adversarial_loss,
}


def get_loss(name: str):
    try:
        return LOSSES[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {', '.join(LOSSES)}") from None
