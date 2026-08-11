"""
Supervised Contrastive Loss (Khosla et al. 2020, "Supervised Contrastive Learning")

Adapted from the unsupervised NT-Xent loss in pretrain.py (SimCLR): the same
cosine-similarity/temperature scaffolding, but the single fixed positive per
anchor (its augmented twin) is replaced by a label-driven positive mask (every
other same-label sample in the batch), each anchor normalized by its own
positive count (SupCon eq. 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Linear -> ReLU -> Linear -> L2-normalize, projecting backbone features
    into the space SupConLoss operates in. Mirrors pretrain.py's SimCLRModel
    projector, extended to two layers (common SupCon-paper choice)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=1)


class SupConLoss(nn.Module):
    """Supervised contrastive loss over a single batch of already-projected,
    L2-normalized embeddings (no two-view/augmentation requirement -- one
    embedding per sample, positives are same-label samples within the batch).

    Anchors with no same-label peer in the batch contribute 0 loss (there is
    nothing to pull together for them) rather than raising or producing NaN.
    A batch with fewer than 2 samples, or where every anchor is label-unique,
    likewise returns 0.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n = embeddings.size(0)
        if n < 2:
            return embeddings.new_zeros(())

        device = embeddings.device
        labels = labels.view(-1, 1)
        same_label = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        positive_mask = same_label.masked_fill(self_mask, 0.0)

        anchors_with_positive = positive_mask.sum(dim=1) > 0
        if not anchors_with_positive.any():
            return embeddings.new_zeros(())

        sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        sim = sim.masked_fill(self_mask, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        # The diagonal is -inf (self-similarity, excluded from the denominator
        # above) and always gets multiplied by positive_mask's 0 there -- but
        # 0 * -inf is NaN in IEEE754, not 0, so it must be zeroed explicitly
        # before that multiplication rather than relying on the mask alone.
        log_prob = log_prob.masked_fill(self_mask, 0.0)

        positive_counts = positive_mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts

        loss_per_anchor = -mean_log_prob_pos[anchors_with_positive]
        return loss_per_anchor.mean()
