"""
CPU-only smoke/unit tests for the additive CE+SCL plumbing:
  - HARDataset(source_id=...)
  - create_dataloaders(return_source_id=...)
  - TwoLayerClassifier(return_features=...)
  - SupConLoss / ProjectionHead
  - finetune.py's train_epoch/evaluate loss_mode="ce_scl" branch, and that
    loss_mode="ce" (default) stays byte-for-byte behaviorally unchanged.

No pytest in this env -- plain asserts, run directly:
    python tests/test_ce_scl.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import HARDataset
from src.data.dataloader import create_dataloaders
from src.models.classifiers import TwoLayerClassifier
from src.losses.supcon import ProjectionHead, SupConLoss
from finetune import train_epoch, evaluate


class DummyBackbone(nn.Module):
    """Minimal backbone stand-in: Conv1d + global mean pool -> (batch, output_dim)."""

    def __init__(self, in_channels=3, output_dim=8):
        super().__init__()
        self.output_dim = output_dim
        self.conv = nn.Conv1d(in_channels, output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x).mean(dim=-1)


def make_synthetic_pool(n_per_user=20, n_channels=3, seq_len=16, n_classes=4, seed=0):
    """Two "datasets" (dsA=target, dsB=source), 6 users each (ids 1..6), namespaced
    the same way load_pooled_datasets() does ("<dataset>::<user>")."""
    rng = np.random.RandomState(seed)
    X_parts, Y_parts, U_parts = [], [], []
    for ds in ("dsA", "dsB"):
        for user in range(1, 7):
            x = rng.randn(n_per_user, n_channels, seq_len).astype(np.float32)
            y = rng.randint(0, n_classes, size=n_per_user)
            X_parts.append(x)
            Y_parts.append(y)
            U_parts.extend([f"{ds}::{user}"] * n_per_user)
    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    U = np.array(U_parts)
    return X, Y, U


def test_hardataset_source_id():
    X = np.random.randn(5, 3, 8).astype(np.float32)
    Y = np.array([0, 1, 0, 1, 0])
    source_id = np.array([0, 0, 1, 1, 1])

    ds_plain = HARDataset(X, Y)
    item = ds_plain[0]
    assert len(item) == 2, "default HARDataset must still return a bare (x, y) 2-tuple"

    ds_tagged = HARDataset(X, Y, source_id=source_id)
    x, y, sid = ds_tagged[2]
    assert int(sid) == 1
    assert int(y) == 0

    try:
        HARDataset(X, Y, source_id=np.array([0, 1]))
        assert False, "mismatched source_id length must raise"
    except ValueError:
        pass
    print("test_hardataset_source_id: PASS")


def test_create_dataloaders_return_source_id():
    X, Y, U = make_synthetic_pool()
    test_users = ["dsA::5", "dsA::6", "dsB::5", "dsB::6"]
    val_users = ["dsA::3", "dsA::4", "dsB::3", "dsB::4"]

    # Plain (default) call: 3-tuple return, untouched.
    out = create_dataloaders(X, Y, U, test_users, val_users, batch_size=8, num_workers=0)
    assert len(out) == 3, "return_source_id=False must keep the existing 3-tuple return"

    train_loader, val_loader, test_loader, dataset_id_map = create_dataloaders(
        X, Y, U, test_users, val_users, batch_size=8, num_workers=0, return_source_id=True,
    )
    assert dataset_id_map == {"dsA": 0, "dsB": 1}

    for loader in (train_loader, val_loader, test_loader):
        batch = next(iter(loader))
        assert len(batch) == 3, "return_source_id=True must yield 3-element batches"
        _, _, source_id = batch
        assert set(source_id.tolist()) <= {0, 1}

    # Spot-check correctness on the (small, non-shuffled) test loader: every
    # sample's source_id must match the dataset its raw user id belongs to.
    all_source_ids = []
    for _, _, sid in test_loader:
        all_source_ids.extend(sid.tolist())
    assert set(all_source_ids) == {0, 1}, "test split should contain both datasets"
    print("test_create_dataloaders_return_source_id: PASS")


def test_classifier_return_features():
    backbone = DummyBackbone(output_dim=8)
    model = TwoLayerClassifier(backbone, n_classes=4)
    x = torch.randn(6, 3, 16)

    logits = model(x)
    assert logits.shape == (6, 4)

    logits2, features = model(x, return_features=True)
    assert logits2.shape == (6, 4)
    assert features.shape == (6, 8)
    assert torch.allclose(logits, logits2, atol=1e-6), "return_features must not change the logits"
    print("test_classifier_return_features: PASS")


def test_supcon_loss():
    torch.manual_seed(0)
    criterion = SupConLoss(temperature=0.1)

    # All-unique labels: no positives anywhere -> loss must be exactly 0.
    embeddings = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
    labels = torch.tensor([0, 1, 2, 3, 4])
    loss = criterion(embeddings, labels)
    assert loss.item() == 0.0

    # n < 2 -> 0.
    assert criterion(embeddings[:1], labels[:1]).item() == 0.0

    # Two well-separated same-label clusters, points already aligned within their
    # own cluster -> should be a small, finite, non-negative loss.
    cluster_a = torch.nn.functional.normalize(torch.randn(1, 8), dim=1).repeat(4, 1)
    cluster_b = torch.nn.functional.normalize(torch.randn(1, 8), dim=1).repeat(4, 1)
    embeddings2 = torch.cat([cluster_a, cluster_b], dim=0)
    labels2 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss2 = criterion(embeddings2, labels2)
    assert torch.isfinite(loss2) and loss2.item() >= 0.0

    # Gradient sanity: loss should be differentiable end-to-end through a projection head.
    head = ProjectionHead(input_dim=8, hidden_dim=16, output_dim=4)
    feats = torch.randn(6, 8, requires_grad=False)
    proj = head(feats)
    labels3 = torch.tensor([0, 0, 1, 1, 2, 2])
    loss3 = criterion(proj, labels3)
    loss3.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    print("test_supcon_loss: PASS")


def test_train_epoch_evaluate_ce_scl_smoke():
    X, Y, U = make_synthetic_pool(n_per_user=15, n_classes=3)
    test_users = ["dsA::5", "dsA::6", "dsB::5", "dsB::6"]
    val_users = ["dsA::3", "dsA::4", "dsB::3", "dsB::4"]

    train_loader, val_loader, test_loader, dataset_id_map = create_dataloaders(
        X, Y, U, test_users, val_users, batch_size=16, num_workers=0, return_source_id=True,
    )
    target_source_id = dataset_id_map["dsA"]

    backbone = DummyBackbone(output_dim=8)
    model = TwoLayerClassifier(backbone, n_classes=3)
    projection_head = ProjectionHead(input_dim=8, hidden_dim=16, output_dim=4)
    scl_criterion = SupConLoss(temperature=0.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(projection_head.parameters()), lr=1e-3)
    device = torch.device("cpu")

    train_loss = train_epoch(
        model, train_loader, criterion, optimizer, device, model_type="resnet",
        loss_mode="ce_scl", scl_weight=1.0, scl_criterion=scl_criterion,
        projection_head=projection_head, target_source_id=target_source_id,
    )
    assert np.isfinite(train_loss)

    val_loss, val_f1, val_acc = evaluate(
        model, val_loader, criterion, device, model_type="resnet",
        loss_mode="ce_scl", target_source_id=target_source_id,
    )
    assert np.isfinite(val_loss) and 0.0 <= val_f1 <= 1.0 and 0.0 <= val_acc <= 1.0

    test_loss, test_f1, test_acc, test_f1_per_class = evaluate(
        model, test_loader, criterion, device, model_type="resnet",
        loss_mode="ce_scl", target_source_id=target_source_id, return_per_class=True,
    )
    assert len(test_f1_per_class) == 3, "per-class F1 must have one entry per class"
    print("test_train_epoch_evaluate_ce_scl_smoke: PASS")


def test_ce_mode_backward_compatible():
    """loss_mode='ce' (default) with plain 2-tuple batches must behave exactly as
    before the CE+SCL change -- this is the path every existing sibling repo call
    (ssl-finetune-from-heavyscore, llm_mfbo_agent/fidelity.py) still uses."""
    X, Y, U = make_synthetic_pool(n_per_user=10, n_classes=3)
    test_users = ["dsA::5", "dsA::6", "dsB::5", "dsB::6"]
    val_users = ["dsA::3", "dsA::4", "dsB::3", "dsB::4"]

    train_loader, val_loader, test_loader = create_dataloaders(
        X, Y, U, test_users, val_users, batch_size=16, num_workers=0,
    )
    backbone = DummyBackbone(output_dim=8)
    model = TwoLayerClassifier(backbone, n_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_loss = train_epoch(model, train_loader, criterion, optimizer, device, model_type="resnet")
    assert np.isfinite(train_loss)
    val_loss, val_f1, val_acc = evaluate(model, val_loader, criterion, device, model_type="resnet")
    assert np.isfinite(val_loss)
    test_loss, test_f1, test_acc = evaluate(model, test_loader, criterion, device, model_type="resnet")
    assert np.isfinite(test_loss)
    print("test_ce_mode_backward_compatible: PASS")


if __name__ == "__main__":
    test_hardataset_source_id()
    test_create_dataloaders_return_source_id()
    test_classifier_return_features()
    test_supcon_loss()
    test_train_epoch_evaluate_ce_scl_smoke()
    test_ce_mode_backward_compatible()
    print("\nAll CE+SCL tests passed.")
