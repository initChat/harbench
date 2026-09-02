#!/usr/bin/env python3
"""
HARBench Evaluation Script

Fine-tune and evaluate models on HAR datasets.
Based on scripts/finetune/finetune_main.py.

Supported models (14 total):
  ResNet-based (SSL-Wearables):
    - resnet      : Random init baseline
    - mtl         : Multi-Task Learning pretrained
    - harnet      : HARNet (OxWearables official, torch.hub)
    - simclr      : SimCLR pretrained
    - moco        : MoCo pretrained
    - timechannel : Masked Resnet (time+channel masking)
    - timemask    : Masked Resnet (time masking only)
    - cpc         : Contrastive Predictive Coding

  Transformer-based:
    - selfpab     : SelfPAB (STFT + Transformer)
    - limubert    : LIMU-BERT
    - imumae      : IMU-Video-MAE (ECCV 2024)

  Foundation Models (require additional dependencies):
    - patchtst    : PatchTST (pip install transformers)
    - moment      : MOMENT (pip install momentfm)

Usage:
    python finetune.py --model mtl --dataset dsads --sensors LeftArm LeftLeg
    python finetune.py --model harnet --dataset dsads --sensors LeftArm LeftLeg --data_ratio 0.1
    python finetune.py --model mtl --zeroshot
    python finetune.py --model resnet --zeroshot-supervised
"""

import argparse
import json
import os
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# All imports from artifact/src (standalone, no parent directory dependency)
from src.data.dataloader import load_dataset, load_pooled_datasets, create_dataloaders, dataset_group_names
from src.data.dataset import HARDataset
from src.models import (
    Resnet, NDeviceResnet, TwoLayerClassifier,
    LIMUBert, IMUVideoMAE,
    SelfPAB, MultiDeviceMaskedResnet, MultiDeviceResnetCPC,
)
from src.utils import macro_f1_score, accuracy, per_class_f1
from src.losses.supcon import ProjectionHead, SupConLoss

# Optional imports for foundation models
try:
    from transformers import PatchTSTConfig, PatchTSTForClassification
    HAS_PATCHTST = True
except ImportError:
    HAS_PATCHTST = False

try:
    from momentfm import MOMENTPipeline
    HAS_MOMENT = True
except ImportError:
    HAS_MOMENT = False


# =============================================================================
# Model Configuration
# =============================================================================

# Supported models and their configurations
# Based on scripts/finetune/finetune_main.py get_backbone()
MODELS = {
    # SSL-Wearables ResNet variants
    "resnet": {
        "type": "resnet",
        "description": "1D ResNet backbone (scratch, random init)",
    },
    "mtl": {
        "type": "resnet",
        "weights": "pretrained/mtl.pth",
        "description": "Multi-Task Learning pretrained ResNet",
    },
    "harnet": {
        "type": "harnet",
        "description": "HARNet (OxWearables official, torch.hub)",
    },
    "simclr": {
        "type": "resnet",
        "weights": "pretrained/simclr.pth",
        "description": "SimCLR pretrained ResNet",
    },
    "moco": {
        "type": "resnet",
        "weights": "pretrained/moco.pth",
        "description": "MoCo pretrained ResNet",
    },

    # Masked Resnet variants
    "timechannel": {
        "type": "maskedresnet",
        "weights": "pretrained/timechannel.pth",
        "description": "Masked Resnet (time+channel masking)",
    },
    "timemask": {
        "type": "maskedresnet",
        "weights": "pretrained/timemask.pth",
        "description": "Masked Resnet (time masking only)",
    },

    # CPC
    "cpc": {
        "type": "cpc",
        "weights": "pretrained/cpc.pth",
        "description": "Contrastive Predictive Coding ResNet",
    },

    # Transformer-based models
    "selfpab": {
        "type": "selfpab",
        "weights": "pretrained/selfpab.ckpt",
        "description": "SelfPAB Transformer encoder (STFT + Transformer)",
    },
    "limubert": {
        "type": "limubert",
        "weights": "pretrained/limubert.pt",
        "description": "LIMU-BERT Transformer encoder",
    },
    "imumae": {
        "type": "imumae",
        "weights": "pretrained/imumae.pth",
        "description": "IMU-Video-MAE encoder (ECCV 2024)",
    },

    # Foundation Models (require additional dependencies)
    "patchtst": {
        "type": "patchtst",
        "description": "PatchTST (Time Series Foundation Model)",
    },
    "moment": {
        "type": "moment",
        "description": "MOMENT Foundation Model",
    },
}


# =============================================================================
# Constants
# =============================================================================

SEED = 42

FOLDS = [
    {"test": [1, 2], "val": [3, 4]},
    {"test": [3, 4], "val": [5, 6]},
    {"test": [5, 6], "val": [7, 8]},
    {"test": [7, 8], "val": [1, 2]},
]

# run_finetune_pooled(): fallback split fractions for a pooled dataset that
# doesn't have >=4 distinct users to hold out by identity (see the random
# per-window split branch below).
WINDOW_SPLIT_TEST_FRAC = 0.2
WINDOW_SPLIT_VAL_FRAC = 0.2

# Zero-shot: Common activity mapping (6 classes)
ZEROSHOT_DATASETS = ["dsads", "mhealth", "pamap2"]
ZEROSHOT_SUPPORT = ["forthtrace", "realdisp", "realworld", "selfback", "ward"]

# Activity mapping from original dataset labels to common 6 classes (0-indexed)
# Common classes: 0=Static, 1=Walking, 2=Running, 3=Stairs, 4=Jumping, 5=Cycling
# Based on src/data/zero_shot_mapping.py (converted from 1-indexed to 0-indexed)
ACTIVITY_MAPPING = {
    "dsads": {
        # Based on dataset_info.py labels
        0: 0, 1: 0, 2: 0, 3: 0,  # Sitting, Standing, Lying(Back/Right) -> Static
        4: 3, 5: 3,  # StairsUp, StairsDown -> Stairs
        6: 0,  # Standing(Elevator) -> Static
        8: 1, 9: 1, 10: 1,  # Walking variations -> Walking
        11: 2,  # Running -> Running
        14: 5, 15: 5,  # Cycling variations -> Cycling
        17: 4,  # Jumping -> Jumping
    },
    "mhealth": {
        0: 0, 1: 0, 2: 0,  # Standing, Sitting, Lying -> Static
        3: 1,  # Walking -> Walking
        4: 3,  # StairsUp -> Stairs
        8: 5,  # Cycling -> Cycling
        9: 2, 10: 2,  # Jogging, Running -> Running
        11: 4,  # JumpFrontBack -> Jumping
    },
    "pamap2": {
        # Based on dataset_info.py: 0=lying, 1=sitting, 2=standing, 3=walking, 4=running, 5=cycling, 7=ascending stairs, 8=descending stairs, 11=rope jumping
        0: 0, 1: 0, 2: 0,  # lying, sitting, standing -> Static
        3: 1,  # walking -> Walking
        4: 2,  # running -> Running
        5: 5,  # cycling -> Cycling
        7: 3, 8: 3,  # ascending/descending stairs -> Stairs
        11: 4,  # rope jumping -> Jumping
    },
    "forthtrace": {
        0: 0, 1: 0, 2: 0,  # Stand, Sit, Sit and Talk -> Static
        3: 1, 4: 1,  # Walk, Walk and Talk -> Walking
        5: 3, 6: 3,  # Climb stairs variations -> Stairs
    },
    "realdisp": {
        0: 1,  # Walking -> Walking
        1: 2, 2: 2,  # Jogging, Running -> Running
        3: 4, 4: 4, 5: 4, 6: 4, 7: 4,  # Jump variations -> Jumping
        32: 5,  # Cycling -> Cycling
    },
    "realworld": {
        # Based on dataset_info.py: 0=ClimbingDown, 1=ClimbingUp, 2=Jumping, 3=Lying, 4=Running, 5=Sitting, 6=Standing, 7=Walking
        0: 3, 1: 3,  # ClimbingDown, ClimbingUp -> Stairs
        2: 4,  # Jumping -> Jumping
        3: 0, 5: 0, 6: 0,  # Lying, Sitting, Standing -> Static
        4: 2,  # Running -> Running
        7: 1,  # Walking -> Walking
    },
    "selfback": {
        0: 3, 1: 3,  # upstairs, downstairs -> Stairs
        2: 1, 3: 1, 4: 1,  # walk slow/mod/fast -> Walking
        5: 2,  # jogging -> Running
        6: 0, 7: 0, 8: 0,  # standing, sitting, lying -> Static
    },
    "ward": {
        0: 0, 1: 0, 2: 0,  # RestStanding, RestSitting, RestLying -> Static
        3: 1,  # WalkFoward -> Walking
        8: 3, 9: 3,  # GoUp/DownStairs -> Stairs
        10: 2,  # Jog -> Running
        11: 4,  # Jump -> Jumping
    },
}

ZEROSHOT_SENSORS = {
    "dsads": ["LeftArm", "LeftLeg"],
    "mhealth": ["RightWrist", "LeftAnkle"],
    "pamap2": ["hand", "ankle"],
    "forthtrace": ["LeftWrist", "RightThigh"],
    "realdisp": ["LeftLowerArm", "LeftThigh"],
    "realworld": ["Forearm", "Thigh"],
    "selfback": ["Wrist", "Thigh"],
    "ward": ["LeftArm", "LeftAnkle"],
}



# =============================================================================
# Core Functions
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device, model_type="resnet", max_iterations=None,
                 loss_mode="ce", scl_weight=1.0, scl_criterion=None, projection_head=None,
                 target_source_id=None):
    """loss_mode="ce_scl" expects `loader` to yield (inputs, labels, source_id) batches
    (i.e. its dataset was built with return_source_id=True) and requires model_type to
    be a plain backbone+TwoLayerClassifier model ("patchtst"/"moment" foundation-model
    wrappers don't expose return_features and aren't supported in this mode).
    L_CE is computed only on samples whose source_id == target_source_id (the target
    dataset); L_SCL only on the remaining (source/baseline) samples. Either term is
    skipped (contributes 0) if its subset is empty in a given batch, rather than
    raising on a degenerate mini-batch."""
    model.train()
    total_loss = 0.0
    total = 0

    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        if max_iterations is not None and i >= max_iterations:
            break

        if loss_mode == "ce_scl":
            inputs, labels, source_id = batch
            source_id = source_id.to(device)
        else:
            inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Special handling for foundation models
        if model_type == "patchtst":
            # PatchTST expects (batch, seq_len, channels)
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs).prediction_logits
        elif model_type == "moment":
            outputs = model.forward(x_enc=inputs).logits
        elif loss_mode == "ce_scl":
            outputs, features = model(inputs, return_features=True)
        else:
            outputs = model(inputs)

        if loss_mode == "ce_scl":
            target_mask = source_id == target_source_id
            source_mask = ~target_mask
            ce_loss = (
                criterion(outputs[target_mask], labels[target_mask])
                if target_mask.any() else outputs.new_zeros(())
            )
            if source_mask.any():
                projected = projection_head(features[source_mask])
                scl_loss = scl_criterion(projected, labels[source_mask])
            else:
                scl_loss = outputs.new_zeros(())
            loss = ce_loss + scl_weight * scl_loss
        else:
            loss = criterion(outputs, labels)

        # A degenerate ce_scl batch (e.g. the loader's trailing partial batch
        # is a single non-target sample) can leave BOTH ce_loss and scl_loss
        # as disconnected new_zeros() fallbacks -- their sum has no grad_fn,
        # which crashes loss.backward(). Such a batch carries no learning
        # signal either way, so skip the step instead of raising.
        if not loss.requires_grad:
            continue

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

    return total_loss / total if total > 0 else 0


def evaluate(model, loader, criterion, device, model_type="resnet", loss_mode="ce", target_source_id=None,
             return_per_class=False, n_classes=None, eval_label_ids=None):
    """loss_mode="ce_scl": `loader` yields (inputs, labels, source_id) batches. Both
    the reported loss AND F1/accuracy/per-class-F1 are computed over target-dataset
    samples only (falling back to every sample in the loader if the target
    contributes none), matching train_epoch's L_CE term -- so a trial's reported
    metrics reflect the target dataset alone, not the pooled baseline+target set
    (see optimal_subset_selection .claude/260825_task.md task 1: F1 used to be
    pooled-wide regardless of loss_mode, which mis-scoped every downstream reward).

    eval_label_ids, when given, is the exact label set macro_f1/per-class-F1 are
    scored and averaged over -- pass the target dataset's own class ids here (see
    run_finetune_pooled's target_label_ids), NOT the full pooled taxonomy, or
    every class only a baseline can have silently drags macro_f1 down by a factor
    that depends on how many baselines got pooled, not on target performance (see
    .claude/260826_task.md reward dilution fix). Falls back to n_classes (the
    classifier head size, i.e. the full pooled taxonomy) when eval_label_ids isn't
    given -- e.g. non-ce_scl callers where there's no single "target" to restrict to.
    """
    model.eval()
    total_loss = 0.0
    total_loss_count = 0
    all_preds = []
    all_labels = []
    target_preds = []
    target_labels = []

    with torch.no_grad():
        for batch in loader:
            if loss_mode == "ce_scl":
                inputs, labels, source_id = batch
                source_id = source_id.to(device)
            else:
                inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Special handling for foundation models
            if model_type == "patchtst":
                inputs = inputs.permute(0, 2, 1)
                outputs = model(inputs).prediction_logits
            elif model_type == "moment":
                outputs = model.forward(x_enc=inputs).logits
            else:
                outputs = model(inputs)

            if loss_mode == "ce_scl":
                target_mask = source_id == target_source_id
                loss_labels = labels[target_mask] if target_mask.any() else labels
                loss_outputs = outputs[target_mask] if target_mask.any() else outputs
                loss = criterion(loss_outputs, loss_labels)
                total_loss += loss.item() * loss_labels.size(0)
                total_loss_count += loss_labels.size(0)
            else:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_loss_count += inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if loss_mode == "ce_scl" and target_mask.any():
                target_preds.extend(predicted[target_mask].cpu().numpy())
                target_labels.extend(labels[target_mask].cpu().numpy())

    if loss_mode == "ce_scl" and target_labels:
        all_preds, all_labels = target_preds, target_labels

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    loss = total_loss / total_loss_count if total_loss_count > 0 else 0
    if eval_label_ids is not None:
        labels_arg = eval_label_ids
    else:
        labels_arg = list(range(n_classes)) if n_classes is not None else None
    f1 = macro_f1_score(all_labels, all_preds, labels=labels_arg)
    acc = accuracy(all_labels, all_preds)

    if return_per_class:
        return loss, f1, acc, per_class_f1(all_labels, all_preds, labels=labels_arg)
    return loss, f1, acc


def create_backbone(model_type, weights_path, num_sensors, in_channels, device):
    """Create backbone based on model type.

    Based on scripts/finetune/finetune_main.py get_backbone() function.
    """
    if model_type == "resnet":
        # NDeviceResnet: Multi-device ResNet with shared weights (SSL-Wearables style)
        backbone = NDeviceResnet(
            state_dict_path=weights_path,
            num_devices=num_sensors,
            device=device,
        )
    elif model_type == "maskedresnet":
        # MultiDeviceMaskedResnet: Masked Reconstruction Model
        backbone = MultiDeviceMaskedResnet(
            device=device,
            num_devices=num_sensors,
            state_dict_path=weights_path,
        )
    elif model_type == "cpc":
        # MultiDeviceResnetCPC: Contrastive Predictive Coding
        backbone = MultiDeviceResnetCPC(
            device=device,
            num_devices=num_sensors,
            state_dict_path=weights_path,
        )
    elif model_type == "selfpab":
        # SelfPAB: STFT + Transformer encoder
        backbone = SelfPAB(
            device=device,
            num_devices=num_sensors,
            checkpoint_path=weights_path if weights_path and weights_path.endswith(".ckpt") else None,
        )
    elif model_type == "limubert":
        # LIMU-BERT: Transformer encoder with automatic resampling
        backbone = LIMUBert(
            feature_num=in_channels,
            hidden=72,
            hidden_ff=144,
            n_layers=4,
            n_heads=4,
            seq_len=150,  # Input: 30Hz, 150 frames
            target_seq_len=120,  # Target: 20Hz for pretrained weights
            emb_norm=True,
            pretrained_path=weights_path if weights_path and weights_path.endswith(".pt") else None,
            device=device,
        )
    elif model_type == "imumae":
        # IMU-Video-MAE: Spectrogram + ViT encoder (ECCV 2024)
        backbone = IMUVideoMAE(
            in_channels=in_channels,
            seq_len=150,
            pretrained_path=weights_path if weights_path and weights_path.endswith(".pth") else None,
            device=device,
        )
    elif model_type == "harnet":
        # HARNet: OxWearables official pretrained model via torch.hub
        # Multi-sensor support: each sensor processed by a separate HARNet
        class NDeviceHARNet(nn.Module):
            def __init__(self, num_devices: int = 1):
                super().__init__()
                self.num_devices = num_devices
                self.output_dim = 512 * num_devices
                self.feature_extractors = nn.ModuleList()
                for _ in range(num_devices):
                    harnet = torch.hub.load(
                        'OxWearables/ssl-wearables', 'harnet5',
                        class_num=5, pretrained=True
                    ).feature_extractor
                    self.feature_extractors.append(harnet)

            def forward(self, x):
                # x: (batch, num_devices * 3, seq_len)
                outputs = []
                for i in range(self.num_devices):
                    x_i = x[:, i*3:(i+1)*3, :]
                    out = self.feature_extractors[i](x_i)
                    outputs.append(out)
                return torch.cat(outputs, dim=1)

        backbone = NDeviceHARNet(num_devices=num_sensors)
    elif model_type == "patchtst":
        # PatchTST: Return None, handled separately in train_model
        if not HAS_PATCHTST:
            raise ImportError("PatchTST requires: pip install transformers")
        return None  # Special case: full model created in train_model
    elif model_type == "moment":
        # MOMENT: Return None, handled separately in train_model
        if not HAS_MOMENT:
            raise ImportError("MOMENT requires: pip install momentfm")
        return None  # Special case: full model created in train_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return backbone


def _extract_backbone_state_dict(model, model_type):
    """Extract a state dict that create_backbone() can reload via --weights.

    NDeviceResnet/MultiDeviceMaskedResnet/MultiDeviceResnetCPC each wrap num_sensors
    per-sensor encoders that were all loaded from the same state_dict_path (see
    create_backbone()); saving the whole wrapped backbone would prefix every key with
    "feature_extractors.0."/"resnet_encoders.0." and no longer match what those classes'
    strict per-encoder load_state_dict() expects. Saving encoder 0 keeps the round trip
    key-compatible.
    """
    backbone = model.backbone
    if model_type == "resnet":
        return backbone.feature_extractors[0].state_dict()
    if model_type in ("maskedresnet", "cpc"):
        return backbone.resnet_encoders[0].state_dict()
    return backbone.state_dict()


def train_model(train_loader, val_loader, test_loader, n_classes, num_sensors,
                weights_path, device, args, model_type="resnet", log_func=None,
                return_backbone=False, target_source_id=None, eval_label_ids=None):
    """Train and evaluate a model.

    `args.loss_mode == "ce_scl"` requires `target_source_id` (the int id, from
    create_dataloaders(return_source_id=True)'s dataset_id_map, of the target
    dataset within the pool) and only supports plain backbone+TwoLayerClassifier
    models -- raises on "patchtst"/"moment", which don't expose return_features.

    eval_label_ids: forwarded to every evaluate() call (both the per-epoch
    val-F1 used for early stopping/checkpoint selection, and the final test
    F1) -- see evaluate()'s docstring. Keeps checkpoint selection and the
    final reported metrics scored against the same class set.
    """
    if log_func is None:
        log_func = print
    in_channels = num_sensors * 3
    loss_mode = getattr(args, "loss_mode", "ce")
    if loss_mode == "ce_scl" and model_type in ("patchtst", "moment"):
        raise NotImplementedError(
            f"loss_mode='ce_scl' is not supported for model_type={model_type!r} "
            "(foundation-model wrappers don't expose return_features); use a "
            "resnet-family model_type instead."
        )
    if loss_mode == "ce_scl" and target_source_id is None:
        raise ValueError("loss_mode='ce_scl' requires target_source_id")

    # Special handling for foundation models that don't use backbone + classifier pattern
    if model_type == "patchtst":
        config = PatchTSTConfig(
            num_input_channels=in_channels,
            num_targets=n_classes,
            context_length=150,
            patch_length=16,
            stride=16,
            use_cls_token=True,
        )
        model = PatchTSTForClassification(config=config)
        model = model.to(device)
    elif model_type == "moment":
        # MOMENT foundation model
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={
                'task_name': 'classification',
                'n_channels': in_channels,
                'num_class': n_classes,
            }
        )
        model.init()  # Initialize classification head
        model = model.to(device)
    else:
        # Standard backbone + classifier pattern
        backbone = create_backbone(model_type, weights_path, num_sensors, in_channels, device)
        model = TwoLayerClassifier(backbone, n_classes=n_classes)
        model = model.to(device)

    projection_head = None
    scl_criterion = None
    if loss_mode == "ce_scl":
        projection_head = ProjectionHead(model.backbone.output_dim).to(device)
        scl_criterion = SupConLoss(temperature=getattr(args, "scl_temperature", 0.1))

    optimizer_params = list(model.parameters())
    if projection_head is not None:
        optimizer_params += list(projection_head.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    import time
    start_time = time.time()

    max_iter = getattr(args, 'max_iterations', None)
    scl_weight = getattr(args, "scl_weight", 1.0)
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, model_type, max_iterations=max_iter,
            loss_mode=loss_mode, scl_weight=scl_weight, scl_criterion=scl_criterion,
            projection_head=projection_head, target_source_id=target_source_id,
        )
        val_loss, val_f1, val_acc = evaluate(
            model, val_loader, criterion, device, model_type,
            loss_mode=loss_mode, target_source_id=target_source_id, n_classes=n_classes,
            eval_label_ids=eval_label_ids,
        )
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        is_best = val_f1 > best_val_f1
        best_marker = " *" if is_best else ""
        n_batches = len(train_loader)

        if not args.quiet:
            log_func(f"  Epoch {epoch+1}/{args.epochs}: 100%|{'█'*10}| {n_batches}/{n_batches} [{epoch_time:.2f}s] train_loss={train_loss:.4f} val_loss={val_loss:.4f}{best_marker}")

        if is_best:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            if not args.quiet:
                log_func(f"  Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_f1, test_acc, test_f1_per_class = evaluate(
        model, test_loader, criterion, device, model_type,
        loss_mode=loss_mode, target_source_id=target_source_id, return_per_class=True, n_classes=n_classes,
        eval_label_ids=eval_label_ids,
    )
    # Per-class val F1 on the same frozen best_model_state, alongside the
    # scalar best_val_f1 already tracked per-epoch above -- lets a caller use
    # validation performance (not test_f1) as a reward/comparison signal
    # without an extra ad hoc split (see .claude/260902_task.md task 5 and
    # run_finetune_pooled's --target_val_users handling).
    _, _, _, val_f1_per_class = evaluate(
        model, val_loader, criterion, device, model_type,
        loss_mode=loss_mode, target_source_id=target_source_id, return_per_class=True, n_classes=n_classes,
        eval_label_ids=eval_label_ids,
    )

    metrics = {
        "test_f1": float(test_f1),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "best_val_f1": float(best_val_f1),
        # Auxiliary metric, logged unconditionally (not just under loss_mode="ce_scl")
        # so it's available from a pipeline's very first trial onward, before any
        # macro-F1 -> per-class-F1 reward-scalarization switch needs it.
        "test_f1_per_class": [float(v) for v in test_f1_per_class],
        "val_f1_per_class": [float(v) for v in val_f1_per_class],
    }

    if return_backbone:
        return metrics, model
    return metrics


# =============================================================================
# Mode: Finetune
# =============================================================================

def run_finetune(args):
    """Standard fine-tuning with 4-fold CV."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get model configuration
    model_config = MODELS.get(args.model, MODELS["resnet"])
    model_type = model_config["type"]

    # Determine weights path (CLI arg > model config > None)
    weights_path = args.weights
    if weights_path is None and "weights" in model_config:
        weights_path = model_config["weights"]

    # Setup logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.model}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Using device: {device}")
    log(f"Model: {args.model} ({model_config['description']})")
    log(f"Weights: {weights_path or 'scratch (random init)'}")
    log(f"Loading dataset: {args.dataset}")
    log(f"Sensors: {args.sensors}")

    # Load data
    X, Y, U = load_dataset(args.dataset, args.sensors, args.data_root)
    log(f"Data shape: X={X.shape}, Y={Y.shape}, U={U.shape}")

    n_classes = len(np.unique(Y))
    num_sensors = len(args.sensors)
    log(f"Classes: {n_classes}, Sensors: {num_sensors}")

    fold_results = []

    for fold_idx, fold in enumerate(FOLDS):
        log(f"\n{'='*60}")
        log(f"Fold {fold_idx + 1}/4: test_users={fold['test']}, val_users={fold['val']}")
        log(f"{'='*60}")

        train_loader, val_loader, test_loader = create_dataloaders(
            X, Y, U, fold["test"], fold["val"],
            batch_size=args.batch_size, data_ratio=args.data_ratio,
            max_samples_per_epoch=args.max_samples_per_epoch
        )

        result = train_model(
            train_loader, val_loader, test_loader,
            n_classes, num_sensors,
            weights_path, device, args,
            model_type=model_type,
            log_func=log
        )

        log(f"Fold {fold_idx + 1} Result: F1={result['test_f1']:.4f}, Acc={result['test_acc']:.4f}")
        fold_results.append(result)

    # Aggregate
    mean_f1 = np.mean([r["test_f1"] for r in fold_results])
    std_f1 = np.std([r["test_f1"] for r in fold_results])
    mean_acc = np.mean([r["test_acc"] for r in fold_results])
    std_acc = np.std([r["test_acc"] for r in fold_results])

    log(f"\n{'='*60}")
    log(f"Final Results (4-fold CV)")
    log(f"{'='*60}")
    log(f"Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    log(f"Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

    results = {
        "mode": "finetune",
        "model": args.model,
        "model_type": model_type,
        "dataset": args.dataset,
        "sensors": args.sensors,
        "weights": weights_path,
        "seed": args.seed,
        "n_classes": n_classes,
        "num_sensors": num_sensors,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
            "optimizer": "Adam",
            "data_ratio": args.data_ratio,
        },
        "fold_results": fold_results,
        "summary": {
            "mean_f1": float(mean_f1),
            "std_f1": float(std_f1),
            "mean_acc": float(mean_acc),
            "std_acc": float(std_acc),
        },
        "timestamp": timestamp,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to: {results_path}")
    log_file.close()
    return results


# =============================================================================
# Mode: Finetune (single split)
# =============================================================================

def run_finetune_single_split(args):
    """Fine-tune on one split (FOLDS[0]) instead of the full 4-fold CV.

    Used when only one trained backbone is needed (see --save_backbone) and paying
    for 4 folds of training would be wasted work.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_config = MODELS.get(args.model, MODELS["resnet"])
    model_type = model_config["type"]

    weights_path = args.weights
    if weights_path is None and "weights" in model_config:
        weights_path = model_config["weights"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.model}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Using device: {device}")
    log(f"Model: {args.model} ({model_config['description']})")
    log(f"Weights: {weights_path or 'scratch (random init)'}")
    log(f"Loading dataset: {args.dataset}")
    log(f"Sensors: {args.sensors}")

    # Load data
    X, Y, U = load_dataset(args.dataset, args.sensors, args.data_root)
    log(f"Data shape: X={X.shape}, Y={Y.shape}, U={U.shape}")

    n_classes = len(np.unique(Y))
    num_sensors = len(args.sensors)
    log(f"Classes: {n_classes}, Sensors: {num_sensors}")

    fold = FOLDS[0]
    if args.custom_test_users and args.custom_val_users:
        fold = {"test": args.custom_test_users, "val": args.custom_val_users}
    log(f"\n{'='*60}")
    log(f"Single split: test_users={fold['test']}, val_users={fold['val']}")
    log(f"{'='*60}")

    train_loader, val_loader, test_loader = create_dataloaders(
        X, Y, U, fold["test"], fold["val"],
        batch_size=args.batch_size, data_ratio=args.data_ratio,
        max_samples_per_epoch=args.max_samples_per_epoch
    )

    result, model = train_model(
        train_loader, val_loader, test_loader,
        n_classes, num_sensors,
        weights_path, device, args,
        model_type=model_type,
        log_func=log,
        return_backbone=True,
    )

    log(f"Result: F1={result['test_f1']:.4f}, Acc={result['test_acc']:.4f}")

    if args.save_backbone:
        os.makedirs(os.path.dirname(args.save_backbone) or ".", exist_ok=True)
        torch.save(_extract_backbone_state_dict(model, model_type), args.save_backbone)
        log(f"Backbone saved to: {args.save_backbone}")

    results = {
        "mode": "finetune_single_split",
        "model": args.model,
        "model_type": model_type,
        "dataset": args.dataset,
        "sensors": args.sensors,
        "weights": weights_path,
        "seed": args.seed,
        "n_classes": n_classes,
        "num_sensors": num_sensors,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
            "optimizer": "Adam",
            "data_ratio": args.data_ratio,
        },
        "split": fold,
        "result": result,
        "backbone_path": args.save_backbone,
        "timestamp": timestamp,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to: {results_path}")
    log_file.close()
    return results


# =============================================================================
# Mode: Finetune (pooled multi-baseline)
# =============================================================================

def _pooled_train_val_test_masks(pairs, U, target_dataset=None, target_val_users=None):
    """Position-based 2-test+2-val-users-per-dataset split (falling back to a
    random per-window split for datasets with <4 distinct users) -- see
    run_finetune_pooled's inline comments for the full rationale (FOLDS[0]'s
    literal ids assume every preprocessed dataset uses raw user ids 1..8,
    which is false in practice). Shared by run_finetune_pooled and
    run_finetune_multi_candidate so this fairly intricate rule has exactly
    one implementation.

    target_dataset/target_val_users: when both are given, `target_dataset`'s
    val role is pinned to `target_val_users` (e.g. optimal_subset_selection's
    held_out.reserve_target() val_ids) instead of this function's own
    auto-picked "next 2 sorted ids" -- and it gets no auto test role at all
    (search-time reward should come from validation performance, not a 4th,
    undeclared subject role -- see .claude/260902_task.md task 5). Every
    other pair (every baseline) keeps the auto-pick untouched.

    Returns (test_mask, val_mask, test_users, val_users, window_split_datasets).
    """
    dataset_raw_users = defaultdict(set)
    for u in U:
        ds, _, raw = u.partition("::")
        dataset_raw_users[ds].add(raw)
    ds_prefix = np.array([u.split("::", 1)[0] for u in U])

    test_users, val_users = [], []
    window_split_test_mask = np.zeros(len(U), dtype=bool)
    window_split_val_mask = np.zeros(len(U), dtype=bool)
    window_split_datasets = []
    for pair in pairs:
        ds = pair["dataset"]
        if target_val_users is not None and ds == target_dataset:
            val_users += [f"{ds}::{uid}" for uid in target_val_users]
            continue
        raw_ids = sorted(dataset_raw_users[ds], key=int)
        if len(raw_ids) < 4:
            window_split_datasets.append(ds)
            ds_indices = np.where(ds_prefix == ds)[0]
            rng = np.random.RandomState(SEED)
            perm = rng.permutation(ds_indices)
            n = len(perm)
            n_test = max(1, round(n * WINDOW_SPLIT_TEST_FRAC))
            n_val = max(1, round(n * WINDOW_SPLIT_VAL_FRAC))
            window_split_test_mask[perm[:n_test]] = True
            window_split_val_mask[perm[n_test:n_test + n_val]] = True
            continue
        test_users += [f"{ds}::{u}" for u in raw_ids[:2]]
        val_users += [f"{ds}::{u}" for u in raw_ids[2:4]]

    test_mask = np.isin(U, test_users) | window_split_test_mask
    val_mask = np.isin(U, val_users) | window_split_val_mask
    return test_mask, val_mask, test_users, val_users, window_split_datasets


def _resolve_target_label_ids(target_dataset, pairs, group_names):
    """Label ids (indices into group_names) belonging to target_dataset only
    -- scores ce_scl macro_f1/per-class-F1 against just the target's own
    classes, not the full pooled taxonomy (see run_finetune_pooled's inline
    comments and .claude/260826_task.md: scoring against the pooled taxonomy
    made macro_f1 diluted by how many baselines got pooled, not by target
    performance). Shared by run_finetune_pooled and run_finetune_multi_candidate."""
    target_pair = next(p for p in pairs if p["dataset"] == target_dataset)
    target_groups = dataset_group_names(target_dataset, target_pair.get("label_map"))
    group_to_idx = {g: i for i, g in enumerate(group_names)}
    return sorted(group_to_idx[g] for g in target_groups if g in group_to_idx)


def run_finetune_pooled(args):
    """Fine-tune once on data pooled from multiple baseline (dataset, sensors,
    data_root) pairs, listed in a manifest file, instead of a single dataset.

    See .claude/260714_plan_finetune_moreSensor.md (design 4) in
    ssl-finetune-from-heavyscore for the full design discussion.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_config = MODELS.get(args.model, MODELS["resnet"])
    model_type = model_config["type"]

    weights_path = args.weights
    if weights_path is None and "weights" in model_config:
        weights_path = model_config["weights"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.model}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    with open(args.baseline_manifest) as f:
        pairs = json.load(f)

    log(f"Using device: {device}")
    log(f"Model: {args.model} ({model_config['description']})")
    log(f"Weights: {weights_path or 'scratch (random init)'}")
    log(f"Loading pooled datasets from manifest: {args.baseline_manifest}")
    log(f"Pairs: {[(p['dataset'], p['sensors']) for p in pairs]}")

    dataset_weights = None
    if getattr(args, "use_manifest_weights", False):
        dataset_weights = {
            p["dataset"]: p["weight"]["requested"] for p in pairs if "weight" in p
        }
        log(f"--use_manifest_weights: dataset_weights={dataset_weights}")

    # Load and pool data
    X, Y, U, group_names = load_pooled_datasets(pairs)
    log(f"Data shape: X={X.shape}, Y={Y.shape}, U={U.shape}")

    n_classes = len(group_names)
    num_sensors = 1  # load_pooled_datasets() requires exactly one sensor per pair
    log(f"Classes: {n_classes} ({group_names}), Sensors: {num_sensors}")

    # FOLDS[0]'s literal ids ([1,2]/[3,4]) assumed every preprocessed dataset
    # uses raw user ids 1..8 -- false in practice (e.g. har70plus at some
    # data_roots uses raw ids 501..518), which silently produced empty val/test
    # splits. Instead, reserve the first 2 sorted raw user ids per dataset for
    # test and the next 2 for val -- position-based, so every pooled dataset
    # still contributes exactly 2 test + 2 val users regardless of its actual
    # id numbering. Extra per-window overrides for datasets that don't have
    # >=4 distinct users -- a per-user split is impossible for them, so
    # instead of raising, fall back to a random per-window split within that
    # dataset only. This does NOT invent fake user identities; it just means
    # that dataset's held-out split isn't subject-independent the way every
    # other pooled dataset's is (there aren't enough real subjects to make it
    # so). See _pooled_train_val_test_masks().
    test_mask, val_mask, test_users, val_users, window_split_datasets = (
        _pooled_train_val_test_masks(
            pairs, U,
            target_dataset=getattr(args, "target_dataset", None),
            target_val_users=getattr(args, "target_val_users", None),
        )
    )

    log(f"\n{'='*60}")
    log(f"Pooled split (2 test + 2 val users per dataset, position-based): "
        f"test={test_users}, val={val_users}")
    if window_split_datasets:
        log(f"Datasets with <4 distinct users, using random per-window split "
            f"instead ({WINDOW_SPLIT_TEST_FRAC:.0%} test / {WINDOW_SPLIT_VAL_FRAC:.0%} val, "
            f"seed={SEED}): {window_split_datasets} "
            f"(test={int(window_split_test_mask.sum())} windows, "
            f"val={int(window_split_val_mask.sum())} windows)")
    log(f"{'='*60}")

    loss_mode = getattr(args, "loss_mode", "ce")
    dataset_id_map = None
    target_source_id = None
    target_label_ids = None
    if loss_mode == "ce_scl":
        train_loader, val_loader, test_loader, dataset_id_map = create_dataloaders(
            X, Y, U, test_users, val_users,
            batch_size=args.batch_size, data_ratio=args.data_ratio,
            max_samples_per_epoch=args.max_samples_per_epoch,
            test_mask=test_mask, val_mask=val_mask,
            return_source_id=True,
            dataset_weights=dataset_weights, log_func=log,
        )
        target_dataset = getattr(args, "target_dataset", None)
        if not target_dataset:
            raise ValueError("loss_mode='ce_scl' requires --target_dataset")
        if target_dataset not in dataset_id_map:
            raise KeyError(
                f"--target_dataset {target_dataset!r} not found among pooled manifest "
                f"datasets {sorted(dataset_id_map)!r}"
            )
        target_source_id = dataset_id_map[target_dataset]

        # Score macro_f1/per-class-F1 against only the target's own classes,
        # not the full pooled taxonomy (group_names spans target + every
        # baseline in this trial's manifest) -- see evaluate()'s docstring
        # and .claude/260826_task.md: scoring against the pooled taxonomy
        # made macro_f1 diluted by how many baselines got pooled, not by
        # target performance. See _resolve_target_label_ids().
        target_label_ids = _resolve_target_label_ids(target_dataset, pairs, group_names)
        log(f"Scoring against target's own classes: {len(target_label_ids)}/{n_classes} "
            f"({[group_names[i] for i in target_label_ids]})")
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            X, Y, U, test_users, val_users,
            batch_size=args.batch_size, data_ratio=args.data_ratio,
            max_samples_per_epoch=args.max_samples_per_epoch,
            test_mask=test_mask, val_mask=val_mask,
        )

    result, model = train_model(
        train_loader, val_loader, test_loader,
        n_classes, num_sensors,
        weights_path, device, args,
        model_type=model_type,
        log_func=log,
        return_backbone=True,
        target_source_id=target_source_id,
        eval_label_ids=target_label_ids,
    )

    target_val_users = getattr(args, "target_val_users", None)
    if target_val_users is not None:
        # test_loader now holds zero target samples (the target's val role was
        # pinned above, so it got no auto test role at all) -- evaluate()'s
        # target-only scoring would otherwise silently fall back to scoring
        # over every sample in the loader (i.e. baseline classes) and mislabel
        # that as the target's test_f1. Null it out rather than report a
        # number that looks like a target metric but isn't one.
        for key in ("test_f1", "test_f1_per_class", "test_acc", "test_loss"):
            result[key] = None
        log("--target_val_users set: target has no search-time test role, so "
            "test_f1/test_f1_per_class/test_acc/test_loss are nulled out here "
            "(would otherwise silently score baseline classes) -- use "
            "best_val_f1/val_f1_per_class instead.")
        log(f"Result: val_F1={result['best_val_f1']:.4f}")
    else:
        log(f"Result: F1={result['test_f1']:.4f}, Acc={result['test_acc']:.4f}")

    if args.save_backbone:
        os.makedirs(os.path.dirname(args.save_backbone) or ".", exist_ok=True)
        torch.save(_extract_backbone_state_dict(model, model_type), args.save_backbone)
        log(f"Backbone saved to: {args.save_backbone}")

    results = {
        "mode": "finetune_pooled",
        "model": args.model,
        "model_type": model_type,
        "baseline_manifest": args.baseline_manifest,
        "pairs": pairs,
        "group_names": group_names,
        "weights": weights_path,
        "seed": args.seed,
        "n_classes": n_classes,
        "num_sensors": num_sensors,
        "loss_mode": loss_mode,
        "target_dataset": getattr(args, "target_dataset", None),
        "target_label_ids": target_label_ids,
        "target_group_names": [group_names[i] for i in target_label_ids] if target_label_ids is not None else None,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
            "optimizer": "Adam",
            "data_ratio": args.data_ratio,
            "scl_weight": getattr(args, "scl_weight", None) if loss_mode == "ce_scl" else None,
            "scl_temperature": getattr(args, "scl_temperature", None) if loss_mode == "ce_scl" else None,
        },
        "split": {"test": test_users, "val": val_users},
        "result": result,
        "backbone_path": args.save_backbone,
        "timestamp": timestamp,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to: {results_path}")
    log_file.close()
    return results


# =============================================================================
# Mode: Multi-candidate (native winner + companions, held-out eval, one process)
# =============================================================================

def run_finetune_multi_candidate(args):
    """Native replacement for optimal_subset_selection/candidate_ablation.py's
    external orchestration: one process trains+held-out-evaluates
    prior_backbone plus N candidates (winner + companions) against the SAME
    already-loaded target data, instead of 2*(N+1) separate finetune.py
    subprocess launches (candidate_ablation.py's design) each reloading
    pooled dataset arrays from disk with no reuse across candidates.

    --candidates_manifest_json: path to a JSON list of {"candidate_id", "S",
    "w", "manifest"}. "manifest" is exactly manifest_builder.build_manifest()'s
    existing [{dataset, sensors, data_root, label_map}, ...] shape -- the
    same contract --baseline_manifest's file content already has, just
    inlined per candidate instead of one file per trial. "S"/"w" ride along
    purely as report metadata: this function never resolves a raw subset+
    weight into a manifest itself (rank_md parsing, label-map resolution,
    weighted-view symlinking) -- that stays an optimal_subset_selection-only
    concern, kept out of this repo's standalone/no-parent-dependency design.

    --dataset/--sensors/--data_root here are the target's FULL (non-search-
    view) data root -- held-out subjects must be physically present so
    --custom_test_users/--custom_val_users can exclude them from training
    and test only on them (same contract held_out.run_held_out_eval() used).
    --weights is the prior pretrained backbone every candidate's pooled
    trial starts from (same role as candidate_ablation.py's --weights).

    Determinism: unlike the old per-candidate-subprocess design (a fresh
    process gets a fresh seed automatically), everything here runs in one
    process, so set_seed(args.seed) is re-called before every train_model()
    call below -- without it, RNG state would carry over between stages and
    make results run-order-dependent, breaking parity with the old numbers.
    GPU memory: explicit `del model` + torch.cuda.empty_cache() after each
    stage, since 2*(N+1) models now train sequentially in one process
    instead of in separate processes that each get memory reclaimed on exit.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_config = MODELS.get(args.model, MODELS["resnet"])
    model_type = model_config["type"]

    prior_weights_path = args.weights
    if prior_weights_path is None and "weights" in model_config:
        prior_weights_path = model_config["weights"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.model}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Using device: {device}")
    log(f"Model: {args.model} ({model_config['description']})")
    log(f"Prior backbone: {prior_weights_path or 'scratch (random init)'}")
    log(f"Target (held-out eval): {args.dataset}, sensors={args.sensors}")

    with open(args.candidates_manifest_json) as f:
        candidates = json.load(f)

    # Target's full data, loaded ONCE and reused for every held-out-eval
    # stage below (prior_backbone's and every candidate's) -- the main
    # efficiency win over candidate_ablation.py's per-candidate subprocess
    # reloads.
    X_t, Y_t, U_t = load_dataset(args.dataset, args.sensors, args.data_root)
    log(f"Target data shape: X={X_t.shape}, Y={Y_t.shape}, U={U_t.shape}")
    n_classes_t = len(np.unique(Y_t))
    num_sensors_t = len(args.sensors)
    held_out_fold = {"test": args.custom_test_users, "val": args.custom_val_users}
    log(f"Held-out fold: test_users={held_out_fold['test']}, val_users={held_out_fold['val']}")

    def _held_out_eval(weights_path, tag):
        """Same single-split finetune+eval held_out.run_held_out_eval()
        already does as a subprocess: fine-tune `weights_path` once more on
        everyone except held_out_fold's test/val users, test ONLY on the
        held-out (test) users. Plain CE (not ce_scl) -- a single-dataset
        finetune has no source/baseline pool to run SCL against."""
        set_seed(args.seed)
        args.loss_mode = "ce"
        train_loader, val_loader, test_loader = create_dataloaders(
            X_t, Y_t, U_t, held_out_fold["test"], held_out_fold["val"],
            batch_size=args.batch_size, data_ratio=args.data_ratio,
            max_samples_per_epoch=args.max_samples_per_epoch,
        )
        result, model = train_model(
            train_loader, val_loader, test_loader, n_classes_t, num_sensors_t,
            weights_path, device, args, model_type=model_type,
            log_func=log, return_backbone=True,
        )
        log(f"[{tag}] held-out eval: F1={result['test_f1']:.4f}, Acc={result['test_acc']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def _train_pooled_candidate(pairs, backbone_out_path, tag):
        """Same CE+SCL pooled trial run_finetune_pooled/trial_runner.run_trial()
        already do as a subprocess, given a candidate's manifest directly
        instead of a --baseline_manifest file. train_model() reads loss_mode
        off `args` (getattr(args, "loss_mode", "ce")), same as
        run_finetune_pooled -- unlike that function, this mode has no
        --loss_mode CLI flag of its own (ce_scl is the only mode a pooled
        multi-candidate trial makes sense under), so it's set explicitly
        here rather than left at argparse's "ce" default. Reset to "ce" by
        _held_out_eval() before its own train_model() call, since dataloaders
        built without return_source_id=True there yield 2-tuples, not the
        3-tuples train_epoch's ce_scl branch expects."""
        set_seed(args.seed)
        args.loss_mode = "ce_scl"
        target_dataset_name = pairs[0]["dataset"]
        log(f"[{tag}] pairs: {[(p['dataset'], p['sensors']) for p in pairs]}")

        dataset_weights = None
        if getattr(args, "use_manifest_weights", False):
            dataset_weights = {
                p["dataset"]: p["weight"]["requested"] for p in pairs if "weight" in p
            }
            log(f"[{tag}] --use_manifest_weights: dataset_weights={dataset_weights}")

        X, Y, U, group_names = load_pooled_datasets(pairs)
        n_classes = len(group_names)
        num_sensors = 1  # load_pooled_datasets() requires exactly one sensor per pair

        test_mask, val_mask, test_users, val_users, window_split_datasets = (
            _pooled_train_val_test_masks(
                pairs, U,
                target_dataset=target_dataset_name,
                target_val_users=args.custom_val_users,
            )
        )
        if window_split_datasets:
            log(f"[{tag}] datasets with <4 distinct users, using random per-window "
                f"split instead: {window_split_datasets}")

        train_loader, val_loader, test_loader, dataset_id_map = create_dataloaders(
            X, Y, U, test_users, val_users,
            batch_size=args.batch_size, data_ratio=args.data_ratio,
            max_samples_per_epoch=args.max_samples_per_epoch,
            test_mask=test_mask, val_mask=val_mask,
            return_source_id=True,
            dataset_weights=dataset_weights, log_func=log,
        )
        if target_dataset_name not in dataset_id_map:
            raise KeyError(
                f"[{tag}] target dataset {target_dataset_name!r} not found among "
                f"pooled manifest datasets {sorted(dataset_id_map)!r}"
            )
        target_source_id = dataset_id_map[target_dataset_name]
        target_label_ids = _resolve_target_label_ids(target_dataset_name, pairs, group_names)
        log(f"[{tag}] scoring against target's own classes: {len(target_label_ids)}/{n_classes}")

        result, model = train_model(
            train_loader, val_loader, test_loader, n_classes, num_sensors,
            prior_weights_path, device, args, model_type=model_type,
            log_func=log, return_backbone=True,
            target_source_id=target_source_id, eval_label_ids=target_label_ids,
        )
        log(f"[{tag}] pooled trial: val_F1={result['best_val_f1']:.4f}")

        os.makedirs(os.path.dirname(backbone_out_path) or ".", exist_ok=True)
        torch.save(_extract_backbone_state_dict(model, model_type), backbone_out_path)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    results = [{
        "candidate_id": "prior_backbone", "weights_path": prior_weights_path, "baselines": {},
        **_held_out_eval(prior_weights_path, "prior_backbone"),
    }]
    log(json.dumps(results[-1]))

    for cand in candidates:
        cid = cand["candidate_id"]
        pairs = cand["manifest"]
        backbone_path = os.path.join(output_dir, f"{cid}_backbone.pth")

        pooled_result = _train_pooled_candidate(pairs, backbone_path, cid)
        held_out_result = _held_out_eval(backbone_path, cid)

        results.append({
            "candidate_id": cid, "weights_path": backbone_path,
            "baselines": dict(zip(cand["S"], cand["w"])),
            # search-time-only signal, kept for reference -- NOT the
            # reportable number, same caution as candidate_ablation.py's
            # pooled_search_time_macro_f1 field. best_val_f1, not test_f1:
            # the target now has no search-time test role once its val is
            # pinned via --target_val_users (see _pooled_train_val_test_masks).
            "pooled_search_time_val_f1": pooled_result["best_val_f1"],
            **held_out_result,
        })
        log(json.dumps(results[-1]))

    payload = {
        "mode": "finetune_multi_candidate",
        "dataset": args.dataset, "sensors": args.sensors,
        "held_out_ids": args.custom_test_users, "val_ids": args.custom_val_users,
        "candidates": results,
        "timestamp": timestamp,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)

    log(f"\nResults saved to: {results_path}")
    log_file.close()
    return payload


# =============================================================================
# Mode: Zero-shot
# =============================================================================

def load_and_map_dataset(dataset_name, data_root=None):
    """Load dataset and map labels to common activities."""
    sensors = ZEROSHOT_SENSORS.get(dataset_name, [])
    if not sensors:
        return None, None, None

    try:
        X, Y, U = load_dataset(dataset_name, sensors, data_root)
    except Exception as e:
        print(f"Warning: Failed to load {dataset_name}: {e}")
        return None, None, None

    mapping = ACTIVITY_MAPPING.get(dataset_name, {})
    Y_mapped = np.array([mapping.get(int(y), -1) for y in Y])

    valid_mask = Y_mapped >= 0
    X = X[valid_mask]
    Y_mapped = Y_mapped[valid_mask]
    U = U[valid_mask]

    return X, Y_mapped, U


def run_zeroshot(args):
    """Zero-shot (LODO) evaluation."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get model configuration
    model_config = MODELS.get(args.model, MODELS["resnet"])
    model_type = model_config["type"]

    # Determine weights path (CLI arg > model config > None)
    weights_path = args.weights
    if weights_path is None and "weights" in model_config:
        weights_path = model_config["weights"]

    # Create output directory and log file (same as finetune)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_suffix = f"_{args.zeroshot}" if args.zeroshot != "all" else ""
    output_dir = os.path.join(args.output_dir, f"{timestamp}_{args.model}_zeroshot{target_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # Multi-seed configuration
    seeds = [args.seed, args.seed + 1, args.seed + 2, args.seed + 3]
    num_seeds = len(seeds)

    log(f"Using device: {device}")
    log(f"Model: {args.model} ({model_config['description']})")
    log(f"Weights: {weights_path or 'scratch (random init)'}")
    log(f"Zero-shot LODO Evaluation ({num_seeds}-seed average: {seeds})")

    results = {}

    # Determine which targets to evaluate
    if args.zeroshot == "all":
        targets = ZEROSHOT_DATASETS
    else:
        if args.zeroshot not in ZEROSHOT_DATASETS:
            log(f"Error: Invalid target '{args.zeroshot}'. Choose from: {ZEROSHOT_DATASETS}")
            return {}
        targets = [args.zeroshot]

    for target in targets:
        log(f"\n{'='*60}")
        log(f"Target: {target} (Zero-shot)")
        log(f"{'='*60}")

        X_target, Y_target, _ = load_and_map_dataset(target, args.data_root)
        if X_target is None:
            continue

        # Load training data (excluding target)
        train_datasets = [d for d in ZEROSHOT_DATASETS + ZEROSHOT_SUPPORT if d != target]
        X_train_list, Y_train_list = [], []

        for dataset in train_datasets:
            X, Y, _ = load_and_map_dataset(dataset, args.data_root)
            if X is not None:
                X_train_list.append(X)
                Y_train_list.append(Y)
                log(f"  Loaded {dataset}: {X.shape[0]} samples")

        if not X_train_list:
            continue

        X_all = np.concatenate(X_train_list, axis=0)
        Y_all = np.concatenate(Y_train_list, axis=0)

        seed_f1s = []
        seed_accs = []

        for seed_idx, seed in enumerate(seeds):
            log(f"\n  --- Seed {seed} ({seed_idx+1}/{num_seeds}) ---")
            set_seed(seed)

            # Split into train/val (80/20)
            from sklearn.model_selection import train_test_split
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_all, Y_all, test_size=0.2, random_state=seed, stratify=Y_all
            )

            log(f"  Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
            log(f"  Target test: {X_target.shape[0]} samples")

            # Create datasets
            train_dataset = HARDataset(X_train, Y_train)
            val_dataset = HARDataset(X_val, Y_val)
            test_dataset = HARDataset(X_target, Y_target)

            # WeightedRandomSampler for class-balanced training
            from collections import Counter
            class_count = Counter(Y_train)
            class_weights = {cls: 1.0 / count for cls, count in class_count.items()}
            sample_weights = np.array([class_weights[y] for y in Y_train])
            sample_weights = torch.from_numpy(sample_weights).float()

            # samples_per_epoch: min(train_size, max_samples) or train_size if None
            if args.max_samples_per_epoch is not None:
                samples_per_epoch = min(len(Y_train), args.max_samples_per_epoch)
            else:
                samples_per_epoch = len(Y_train)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=samples_per_epoch,
                replacement=True
            )

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            # Create model
            num_sensors = len(ZEROSHOT_SENSORS[target])
            in_channels = num_sensors * 3
            n_classes = 6  # Zero-shot uses 6 common classes

            # Special handling for foundation models (same as train_model)
            if model_type == "patchtst":
                config = PatchTSTConfig(
                    num_input_channels=in_channels,
                    num_targets=n_classes,
                    context_length=150,
                    patch_length=16,
                    stride=16,
                    use_cls_token=True,
                )
                model = PatchTSTForClassification(config=config)
                model = model.to(device)
            elif model_type == "moment":
                model = MOMENTPipeline.from_pretrained(
                    "AutonLab/MOMENT-1-small",
                    model_kwargs={
                        'task_name': 'classification',
                        'n_channels': in_channels,
                        'num_class': n_classes,
                    }
                )
                model.init()
                model = model.to(device)
            else:
                # Standard backbone + classifier pattern
                backbone = create_backbone(model_type, weights_path, num_sensors, in_channels, device)
                model = TwoLayerClassifier(backbone, n_classes=n_classes)
                model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            criterion = nn.CrossEntropyLoss()

            best_f1 = 0.0
            best_state = None
            patience_counter = 0
            n_batches = len(train_loader)

            for epoch in range(args.epochs):
                import time
                start_time = time.time()

                # Train
                model.train()
                total_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # Special handling for foundation models
                    if model_type == "patchtst":
                        inputs_t = inputs.permute(0, 2, 1)
                        outputs = model(inputs_t).prediction_logits
                    elif model_type == "moment":
                        outputs = model.forward(x_enc=inputs).logits
                    else:
                        outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                train_loss = total_loss / len(train_loader)

                # Evaluate on validation (for model selection)
                val_loss, val_f1, val_acc = evaluate(model, val_loader, criterion, device, model_type)

                epoch_time = time.time() - start_time
                is_best = val_f1 > best_f1
                best_marker = " *" if is_best else ""

                log(f"  Epoch {epoch+1}/{args.epochs}: 100%|{'█'*10}| {n_batches}/{n_batches} [{epoch_time:.2f}s] train_loss={train_loss:.4f} val_loss={val_loss:.4f}{best_marker}")

                if is_best:
                    best_f1 = val_f1
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    log(f"  Early stopping at epoch {epoch+1}")
                    break

                scheduler.step()

            if best_state:
                model.load_state_dict(best_state)
            # Final evaluation on test (target dataset)
            _, f1, acc = evaluate(model, test_loader, criterion, device, model_type)

            log(f"  Seed {seed} Result: F1={f1:.4f}, Acc={acc:.4f}")
            seed_f1s.append(f1)
            seed_accs.append(acc)

        # Average over seeds
        mean_f1 = float(np.mean(seed_f1s))
        mean_acc = float(np.mean(seed_accs))
        std_f1 = float(np.std(seed_f1s))
        log(f"Target {target} Result: F1={mean_f1:.4f} (±{std_f1:.4f}), Acc={mean_acc:.4f}")
        results[target] = {"f1": mean_f1, "acc": mean_acc, "std_f1": std_f1, "seed_f1s": [float(f) for f in seed_f1s]}

    # Summary
    if results:
        mean_f1 = np.mean([r["f1"] for r in results.values()])

        log(f"\n{'='*60}")
        log(f"Zero-shot Results Summary ({num_seeds}-seed average)")
        log(f"{'='*60}")
        for dataset, r in results.items():
            log(f"  {dataset}: F1={r['f1']:.4f} (±{r['std_f1']:.4f})")
        log(f"  Average: F1={mean_f1:.4f}")

        final_results = {
            "mode": "zeroshot",
            "model": args.model,
            "model_type": model_type,
            "weights": weights_path,
            "seeds": seeds,
            "dataset_results": results,
            "summary": {"mean_f1": float(mean_f1)},
            "timestamp": timestamp,
        }

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)

        log(f"\nResults saved to: {results_path}")
        log_file.close()
        return final_results

    log_file.close()
    return {}


# =============================================================================
# Mode: Zero-shot Supervised (Upper Bound Reference)
# =============================================================================

def run_zeroshot_supervised(args):
    """Supervised evaluation on zero-shot target datasets (DSADS, MHEALTH, PAMAP2).

    This provides the upper bound reference for zero-shot evaluation by training
    and testing on the same target dataset using 4-fold CV with 6 common activity classes.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get model configuration
    model_config = MODELS.get(args.model, MODELS["resnet"])
    model_type = model_config["type"]

    # Determine weights path (CLI arg > model config > None)
    weights_path = args.weights
    if weights_path is None and "weights" in model_config:
        weights_path = model_config["weights"]

    # Create output directory and log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{timestamp}_{args.model}_zeroshot_supervised")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Using device: {device}")
    log(f"Model: {args.model} ({model_config['description']})")
    log(f"Weights: {weights_path or 'scratch (random init)'}")
    log(f"Zero-shot Supervised Evaluation (Upper Bound)")

    results = {}

    for target in ZEROSHOT_DATASETS:
        log(f"\n{'='*60}")
        log(f"Target: {target} (Supervised - 4-fold CV)")
        log(f"{'='*60}")

        # Load and map dataset to common 6 classes
        X, Y, U = load_and_map_dataset(target, args.data_root)
        if X is None:
            log(f"  Skipping {target}: failed to load")
            continue

        log(f"  Data shape: X={X.shape}, Y={Y.shape}, U={U.shape}")
        log(f"  Classes: {len(np.unique(Y))}, Sensors: {len(ZEROSHOT_SENSORS[target])}")

        num_sensors = len(ZEROSHOT_SENSORS[target])
        n_classes = 6  # Common activity classes

        fold_results = []

        for fold_idx, fold in enumerate(FOLDS):
            log(f"\n  Fold {fold_idx + 1}/4: test_users={fold['test']}, val_users={fold['val']}")

            train_loader, val_loader, test_loader = create_dataloaders(
                X, Y, U, fold["test"], fold["val"],
                batch_size=args.batch_size, data_ratio=args.data_ratio,
                max_samples_per_epoch=args.max_samples_per_epoch
            )

            # Check if loaders have data
            if len(train_loader) == 0 or len(test_loader) == 0:
                log(f"    Skipping fold {fold_idx + 1}: insufficient data")
                continue

            result = train_model(
                train_loader, val_loader, test_loader,
                n_classes, num_sensors,
                weights_path, device, args,
                model_type=model_type,
                log_func=lambda msg: log(f"    {msg}")
            )

            log(f"    Fold {fold_idx + 1} Result: F1={result['test_f1']:.4f}, Acc={result['test_acc']:.4f}")
            fold_results.append(result)

        if fold_results:
            mean_f1 = np.mean([r["test_f1"] for r in fold_results])
            std_f1 = np.std([r["test_f1"] for r in fold_results])
            mean_acc = np.mean([r["test_acc"] for r in fold_results])

            log(f"\n  {target} Result: F1={mean_f1:.4f} +/- {std_f1:.4f}")
            results[target] = {
                "f1": float(mean_f1),
                "f1_std": float(std_f1),
                "acc": float(mean_acc),
                "fold_results": fold_results,
            }

    # Summary
    if results:
        mean_f1 = np.mean([r["f1"] for r in results.values()])

        log(f"\n{'='*60}")
        log(f"Zero-shot Supervised Results Summary (Upper Bound)")
        log(f"{'='*60}")
        for dataset, r in results.items():
            log(f"  {dataset}: F1={r['f1']:.4f} +/- {r['f1_std']:.4f}")
        log(f"  Average: F1={mean_f1:.4f}")

        final_results = {
            "mode": "zeroshot_supervised",
            "model": args.model,
            "model_type": model_type,
            "weights": weights_path,
            "seed": args.seed,
            "dataset_results": results,
            "summary": {"mean_f1": float(mean_f1)},
            "timestamp": timestamp,
        }

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)

        log(f"\nResults saved to: {results_path}")
        log_file.close()
        return final_results

    log_file.close()
    return {}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HARBench Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported models (13 total):

  ResNet-based (SSL-Wearables):
    resnet       - Random init baseline
    mtl          - Multi-Task Learning pretrained
    harnet       - HARNet (OxWearables official)
    simclr       - SimCLR pretrained
    moco         - MoCo pretrained
    timechannel  - Masked Resnet (time+channel)
    timemask     - Masked Resnet (time only)
    cpc          - Contrastive Predictive Coding

  Transformer-based:
    selfpab      - SelfPAB (STFT + Transformer)
    limubert     - LIMU-BERT
    imumae       - IMU-Video-MAE (ECCV 2024)

  Foundation Models:
    patchtst     - PatchTST (requires transformers)
    moment       - MOMENT (requires momentfm)

Examples:
  python finetune.py --model mtl --dataset dsads --sensors LeftArm LeftLeg
  python finetune.py --model harnet --dataset dsads --sensors LeftArm
  python finetune.py --model patchtst --dataset dsads --sensors LeftArm LeftLeg
"""
    )
    parser.add_argument("--model", type=str, default="resnet",
                        choices=list(MODELS.keys()),
                        help="Model to use (default: resnet)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--sensors", type=str, nargs="+", default=None, help="Sensor names")
    parser.add_argument("--weights", type=str, default=None,
                        help="Override pretrained weights path (optional)")
    parser.add_argument("--data_root", type=str, default=None, help="Data root path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--max_samples_per_epoch", type=int, default=3200,
                        help="Max samples per epoch (default=3200). Capped to training data size.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--data_ratio", type=float, default=1.0, help="Training data ratio (for few-shot)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress epoch-level output")
    parser.add_argument("--zeroshot", nargs="?", const="all", default=None,
                        help="Run zero-shot (LODO) evaluation. Optionally specify target: dsads, mhealth, pamap2, or 'all'")
    parser.add_argument("--zeroshot-supervised", action="store_true",
                        help="Run supervised evaluation on zero-shot target datasets (upper bound reference)")
    parser.add_argument("--single_split", action="store_true",
                        help="Train once on FOLDS[0] instead of the full 4-fold CV")
    parser.add_argument("--save_backbone", type=str, default=None,
                        help="Path to save the trained backbone's state_dict "
                             "(requires --single_split or --baseline_manifest)")
    parser.add_argument("--baseline_manifest", type=str, default=None,
                        help="Path to a JSON manifest (list of {dataset, sensors, data_root}) "
                             "to pool and fine-tune on jointly, instead of a single --dataset. "
                             "Trains once on a per-pair-namespaced FOLDS[0] split. Replaces "
                             "--dataset/--sensors/--data_root for this run.")
    parser.add_argument("--custom_test_users", type=int, nargs="+", default=None,
                        help="Override FOLDS[0]'s test users for --single_split with an explicit "
                             "user-id list (e.g. a permanent held-out set never used in any "
                             "regular fold). Requires --single_split and --custom_val_users. "
                             "Omitting this preserves today's behavior exactly (FOLDS[0]).")
    parser.add_argument("--custom_val_users", type=int, nargs="+", default=None,
                        help="Paired with --custom_test_users -- see its help.")
    parser.add_argument("--target_val_users", type=int, nargs="+", default=None,
                        help="With --baseline_manifest/--loss_mode ce_scl: pin the TARGET "
                             "dataset's validation users to this explicit id list instead of "
                             "_pooled_train_val_test_masks's auto position-based pick (first-2-"
                             "sorted-ids test / next-2 val). The target gets no auto test role "
                             "when this is set -- search-time reward/checkpoint-selection "
                             "should come from validation performance, not an extra ad hoc "
                             "split; the one honest generalization check remains "
                             "--custom_test_users at final-eval. Should be exactly the val ids "
                             "optimal_subset_selection's held_out.reserve_target() already "
                             "reserved for this run.")
    parser.add_argument("--loss_mode", type=str, default="ce", choices=["ce", "ce_scl"],
                        help="'ce' (default): plain cross-entropy. 'ce_scl': cross-entropy on "
                             "--target_dataset's own pooled samples plus a supervised contrastive "
                             "loss on the remaining (source/baseline) pooled samples. Only "
                             "supported together with --baseline_manifest and a resnet-family "
                             "--model (not patchtst/moment).")
    parser.add_argument("--target_dataset", type=str, default=None,
                        help="Required with --loss_mode ce_scl: the 'dataset' name (matching one "
                             "entry's \"dataset\" field in --baseline_manifest) whose samples get "
                             "the L_CE term; all other pooled entries get the L_SCL term.")
    parser.add_argument("--scl_weight", type=float, default=1.0,
                        help="Weight on the L_SCL term when --loss_mode ce_scl (L = L_CE + scl_weight * L_SCL).")
    parser.add_argument("--scl_temperature", type=float, default=0.1,
                        help="SupConLoss temperature when --loss_mode ce_scl.")
    parser.add_argument("--use_manifest_weights", action="store_true",
                        help="With --baseline_manifest, read each pair's \"weight\":{\"requested\": "
                             "w_i} field (written by optimal_subset_selection's/llm_mfbo_agent's "
                             "weighted_pool.py) and scale the ce_scl weighted sampler's per-dataset "
                             "draw mass by it, instead of pure uniform (dataset, class) balancing. "
                             "Default off -- existing callers that don't emit a \"weight\" field are "
                             "unaffected either way.")
    parser.add_argument("--candidates_manifest_json", type=str, default=None,
                        help="Path to a JSON list of {\"candidate_id\", \"S\", \"w\", \"manifest\"} "
                             "(\"manifest\" is a --baseline_manifest-shaped [{dataset, sensors, "
                             "data_root, label_map}, ...] list). Trains+held-out-evaluates "
                             "--weights (as 'prior_backbone') plus every candidate in one process, "
                             "reusing the target data loaded once via --dataset/--sensors/"
                             "--data_root (the FULL, non-search-view root) and --custom_test_users/"
                             "--custom_val_users as the held-out/val split. Mutually exclusive with "
                             "--baseline_manifest/--single_split. Native replacement for "
                             "optimal_subset_selection/candidate_ablation.py's external, "
                             "subprocess-per-candidate orchestration.")
    args = parser.parse_args()

    if args.save_backbone and not (args.single_split or args.baseline_manifest):
        parser.error("--save_backbone requires --single_split or --baseline_manifest")
    if args.single_split and args.baseline_manifest:
        parser.error("--single_split and --baseline_manifest are mutually exclusive")
    if args.loss_mode == "ce_scl" and not args.baseline_manifest:
        parser.error("--loss_mode ce_scl requires --baseline_manifest")
    if args.loss_mode == "ce_scl" and not args.target_dataset:
        parser.error("--loss_mode ce_scl requires --target_dataset")
    if args.target_val_users and not args.target_dataset:
        parser.error("--target_val_users requires --target_dataset")
    if bool(args.custom_test_users) != bool(args.custom_val_users):
        parser.error("--custom_test_users and --custom_val_users must be given together")
    if args.custom_test_users and not (args.single_split or args.candidates_manifest_json):
        parser.error("--custom_test_users/--custom_val_users require --single_split or --candidates_manifest_json")
    if args.candidates_manifest_json and (args.single_split or args.baseline_manifest):
        parser.error("--candidates_manifest_json is mutually exclusive with --single_split/--baseline_manifest")
    if args.candidates_manifest_json and not (args.dataset and args.sensors and args.data_root):
        parser.error("--candidates_manifest_json requires --dataset/--sensors/--data_root (the target's full data root)")
    if args.candidates_manifest_json and not args.weights:
        parser.error("--candidates_manifest_json requires --weights (the prior backbone)")
    if args.candidates_manifest_json and not args.custom_test_users:
        parser.error("--candidates_manifest_json requires --custom_test_users/--custom_val_users (the held-out/val split)")

    set_seed(args.seed)

    if args.zeroshot:
        args.output_dir = os.path.join(args.output_dir, "zeroshot")
        run_zeroshot(args)
    elif getattr(args, 'zeroshot_supervised', False):
        args.output_dir = os.path.join(args.output_dir, "zeroshot_supervised")
        run_zeroshot_supervised(args)
    elif args.candidates_manifest_json:
        args.output_dir = os.path.join(args.output_dir, "finetune_multi_candidate")
        run_finetune_multi_candidate(args)
    elif args.baseline_manifest:
        args.output_dir = os.path.join(args.output_dir, "finetune_pooled")
        run_finetune_pooled(args)
    elif args.single_split:
        if args.dataset is None or args.sensors is None:
            parser.error("--dataset and --sensors are required")
        args.output_dir = os.path.join(args.output_dir, "finetune_single_split")
        run_finetune_single_split(args)
    else:
        if args.dataset is None or args.sensors is None:
            parser.error("--dataset and --sensors are required")
        args.output_dir = os.path.join(args.output_dir, "finetune")
        run_finetune(args)


if __name__ == "__main__":
    main()
