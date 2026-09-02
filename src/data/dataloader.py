"""
HARBench Data Loading

Load data from processed_strict format.
Structure: {data_root}/{dataset}/USER{id}/{sensor}/{modality}/X.npy, Y.npy
"""

import os
import re
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from .dataset import HARDataset


# Default data root (processed format)
# Priority: environment variable > artifact/har-datasets
DEFAULT_DATA_ROOT = os.environ.get(
    "HARBENCH_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "../../har-datasets/data/processed")
)

# har-datasets' own package is also named "src" (relative to its root), which
# is already bound in sys.modules for *this* package tree -- inserting
# har-datasets/ and importing "src.dataset_taxonomy" would silently resolve to
# the wrong "src". Instead put har-datasets/src itself on sys.path and import
# its bare top-level modules, which is exactly the layout dataset_taxonomy.py's
# import fallback (`except ImportError: from dataset_info import DATASETS`)
# is written to support.
_HAR_DATASETS_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../har-datasets/src")
)
if _HAR_DATASETS_SRC not in sys.path:
    sys.path.insert(0, _HAR_DATASETS_SRC)

from dataset_info import DATASETS
from dataset_taxonomy import get_dataset_label_mapping


# =============================================================================
# Usable class definitions (based on const.py usable_labels and processed_strict activity_map)
#
# Important: const.py labels definition and processed_strict activity_map use different ordering.
# Here, we reference each preprocessor's activity_map and map activity names from
# const.py usable_labels to processed_strict class IDs.
# =============================================================================
USABLE_CLASSES = {
    # Daily
    # DSADS: All 19 classes used
    "dsads": None,

    # FORTHTRACE: 11 classes used
    "forthtrace": None,

    # HARTH: 12 -> 10 classes
    # processed_strict: {walking:0, running:1, shuffling:2, stairs(ascending):3,
    #                   stairs(descending):4, standing:5, sitting:6, lying:7,
    #                   cycling(sit):8, cycling(stand):9, cycling(sit,inactive):10,
    #                   cycling(stand,inactive):11}
    # usable: walking, shuffling, stairs(ascending), stairs(descending), standing,
    #         sitting, lying, cycling(sit), cycling(stand), cycling(sit,inactive)
    # Excluded: running(1), cycling(stand,inactive)(11)
    "harth": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    # IMWSHA: All 11 classes used
    "imwsha": None,

    # PAAL: 24 -> 11 classes (matching const.py n_usable_classes=11)
    # processed_strict: mapped in order 0-23
    # usable: brush_teeth(4), brush_hair(5), take_off_a_jacket(6), put_on_a_jacket(7),
    #         put_on_a_shoe(8), writing(14), type_on_a_keyboard(16),
    #         washing_dishes(20), dusting(21), ironing(22)
    # Note: washing_dishes is duplicated (20 and 23), effectively using 10 classes
    "paal": [4, 5, 6, 7, 8, 14, 16, 20, 21, 22],

    # PAMAP2: processed_strict has only 12 classes (matching const.py n_usable_classes=12)
    # All classes used
    "pamap2": None,

    # SELFBACK: All 9 classes used
    "selfback": None,

    # UCAEHAR: 8 -> 6 classes
    # processed_strict: {WALKING:0, RUNNING:1, STANDING:2, SITTING:3, LYING:4,
    #                   DRINKING:5, WALKING_UPSTAIRS:6, WALKING_DOWNSTAIRS:7}
    # usable: STANDING(2), SITTING(3), WALKING(0), LYING(4), WALKING_UPSTAIRS(6), RUNNING(1)
    # Excluded: DRINKING(5), WALKING_DOWNSTAIRS(7)
    "ucaehar": [0, 1, 2, 3, 4, 6],

    # USCHAD: All classes used (matching const.py n_usable_classes=12)
    "uschad": None,

    # WARD: All 13 classes used
    "ward": None,

    # REALWORLD: All 8 classes used
    "realworld": None,

    # Exercise
    # MEx: All 7 classes used
    "mex": None,

    # MHEALTH: All 12 classes used
    "mhealth": None,

    # REALDISP: All 33 classes used
    "realdisp": None,

    # Industry
    # LARa: 8 -> 6 classes
    # processed_strict: {Standing:0, Walking:1, Cart:2, Handling(upwards):3,
    #                   Handling(centred):4, Handling(downwards):5, Synchronization:6, None:7}
    # usable: Standing(0), Walking(1), Cart(2), Handling(upwards)(3),
    #         Handling(centred)(4), Synchronization(6)
    # Excluded: Handling(downwards)(5), None(7)
    "lara": [0, 1, 2, 3, 4, 6],

    # OPENPACK: All 10 classes used
    "openpack": None,

    # EXOSKELETONS: All 4 classes used
    "exoskeletons": None,

    # VTT_CONIOT: All 16 classes used
    "vtt_coniot": None,
}

def _resolve_dataset_dir(data_root, dataset):
    """Resolve a dataset name to its on-disk directory, tolerating case differences.

    Dataset dirs are conventionally lowercase, but some (e.g. "USC-HAD") are not,
    which breaks a naive `dataset.lower()` join on case-sensitive filesystems.
    """
    exact = os.path.join(data_root, dataset.lower())
    if os.path.exists(exact):
        return exact

    if os.path.isdir(data_root):
        target = dataset.lower()
        for entry in os.listdir(data_root):
            if entry.lower() == target:
                return os.path.join(data_root, entry)

    return exact



def load_dataset(dataset, sensors, data_root=None, modality="ACC"):
    """
    Load dataset (processed_strict format).

    Structure: {data_root}/{dataset}/USER{id}/{sensor}/{modality}/X.npy, Y.npy

    Args:
        dataset: Dataset name (e.g., "DSADS", "PAMAP2")
        sensors: List of sensor names (e.g., ["Chest", "Thigh"])
        data_root: Data root path
        modality: Modality ("ACC", "GYRO", "MAG")

    Returns:
        X: Sensor data (N, C, T)
        Y: Labels (N,)
        U: User IDs (N,)
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    # dataset_path = os.path.join(data_root, dataset.lower())
    dataset_path = _resolve_dataset_dir(data_root, dataset)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Get list of users
    users = sorted([d for d in os.listdir(dataset_path) if d.startswith("USER")])

    X_all, Y_all, U_all = [], [], []

    for user in users:
        user_id = int(user.replace("USER", "").lstrip("0") or "0")
        user_path = os.path.join(dataset_path, user)

        X_sensors = []
        Y_user = None

        for sensor in sensors:
            sensor_path = os.path.join(user_path, sensor, modality)

            if not os.path.exists(sensor_path):
                continue

            x_path = os.path.join(sensor_path, "X.npy")
            y_path = os.path.join(sensor_path, "Y.npy")

            if not os.path.exists(x_path):
                continue

            x = np.load(x_path)
            y = np.load(y_path)

            # float16 -> float32
            if x.dtype == np.float16:
                x = x.astype(np.float32)

            X_sensors.append(x)
            if Y_user is None:
                Y_user = y

        if not X_sensors:
            continue

        # If sample counts differ between sensors, align to minimum sample count
        min_samples = min(x.shape[0] for x in X_sensors)
        X_sensors = [x[:min_samples] for x in X_sensors]
        Y_user = Y_user[:min_samples]

        # Concatenate sensors (N, C1, T) + (N, C2, T) -> (N, C1+C2, T)
        X_user = np.concatenate(X_sensors, axis=1)
        n_samples = X_user.shape[0]

        X_all.append(X_user)
        Y_all.append(Y_user)
        U_all.append(np.full(n_samples, user_id))

    if not X_all:
        raise ValueError(f"No data found for dataset={dataset}, sensors={sensors}")

    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    U = np.concatenate(U_all, axis=0)

    # Exclude negative labels (unlabeled data)
    valid_mask = Y >= 0
    if not np.all(valid_mask):
        X = X[valid_mask]
        Y = Y[valid_mask]
        U = U[valid_mask]

    # If USABLE_CLASSES is defined, use only specified classes
    from collections import Counter
    dataset_lower = dataset.lower()
    usable_classes = USABLE_CLASSES.get(dataset_lower)

    if usable_classes is not None:
        # If explicit class list is defined
        class_counts = Counter(Y)
        excluded_classes = [cls for cls in class_counts.keys() if cls not in usable_classes]

        if excluded_classes:
            print(f"  Using predefined usable classes for {dataset_lower}")
            print(f"  Excluding classes: {sorted(excluded_classes)}")
            print(f"  Class counts: {dict(sorted(class_counts.items()))}")

            valid_mask = np.isin(Y, usable_classes)
            X = X[valid_mask]
            Y = Y[valid_mask]
            U = U[valid_mask]

            # Remap labels to consecutive integers starting from 0
            label_map = {old: new for new, old in enumerate(sorted(usable_classes))}
            Y = np.array([label_map[y] for y in Y])

            print(f"  Remaining classes: {len(usable_classes)}, samples: {len(Y)}")

    return X, Y, U


def _lookup_dataset_info_key(dataset_name):
    """Resolve a dataset name (as used on-disk, e.g. "har70plus"/"USC-HAD") to
    its dataset_info.py DATASETS key (e.g. "HAR70PLUS"/"USCHAD"), tolerating
    case and punctuation differences between the two naming conventions."""
    normalized = re.sub(r"[^A-Z0-9]", "", dataset_name.upper())
    for key in DATASETS:
        if re.sub(r"[^A-Z0-9]", "", key.upper()) == normalized:
            return key
    raise KeyError(f"No dataset_info.py DATASETS entry matches dataset {dataset_name!r}")


def _class_idx_to_taxonomy_group(dataset_name):
    """Return {class_idx: canonical_taxonomy_group} for a dataset, built from
    dataset_taxonomy.py's get_dataset_label_mapping() (raw_label -> group)
    composed with dataset_info.py's own idx -> raw_label labels dict.

    Only valid for datasets whose Y.npy class indices still match
    dataset_info.py's labels dict directly -- i.e. dataset_lower not present in
    load_dataset()'s USABLE_CLASSES remapping (confirmed true today for
    har70plus/hhar/uschad, see .claude/260714_plan_finetune_moreSensor.md
    design 2/3). A dataset that *is* remapped will raise KeyError below when
    its remapped indices are looked up, which is the correct fail-loud
    behavior rather than silently mislabeling pooled samples.
    """
    dataset_key = _lookup_dataset_info_key(dataset_name)
    label_to_group = get_dataset_label_mapping(dataset_key)  # {raw_label: group}, skips idx == -1
    idx_to_label = DATASETS[dataset_key]["labels"]
    return {idx: label_to_group[label] for idx, label in idx_to_label.items() if idx != -1}


def _resolve_idx_to_group(dataset_name, label_map=None):
    """{class_idx: canonical_taxonomy_group} for one dataset -- the
    caller-supplied label_map override when given, else
    _class_idx_to_taxonomy_group()'s dataset_info.py/dataset_taxonomy.py
    derivation. Shared by load_pooled_datasets()'s per-pair loop and
    dataset_group_names() so there's one resolution path."""
    if label_map is not None:
        return {int(k): v for k, v in label_map.items()}
    return _class_idx_to_taxonomy_group(dataset_name)


def dataset_group_names(dataset_name, label_map=None):
    """The set of canonical taxonomy groups a single dataset can produce
    (excludes "undefined"), using the same idx_to_group resolution
    load_pooled_datasets() applies per-pair -- the caller-supplied label_map
    override included. Used by finetune.py's run_finetune_pooled() to score
    a pooled trial's F1 against the target dataset's own classes only,
    instead of the full pooled taxonomy (see .claude/260826_task.md reward
    dilution fix)."""
    idx_to_group = _resolve_idx_to_group(dataset_name, label_map)
    return {group for group in idx_to_group.values() if group != "undefined"}


def load_pooled_datasets(pairs, modality="ACC"):
    """
    Load and pool several selected baseline (dataset, sensors, data_root) pairs
    into one joint training set, remapping each pair's own labels onto
    dataset_taxonomy.py's canonical activity groups so pairs from different
    datasets share one label space.

    See .claude/260714_plan_finetune_moreSensor.md (design 3) in
    ssl-finetune-from-heavyscore for the full design discussion.

    Args:
        pairs: list of dicts, each with keys "dataset", "sensors", "data_root"
               -- the same shape as the manifest finetune.py's pooled-training
               CLI mode is expected to consume (design 4). A pair may also
               carry an optional "label_map" ({class_idx (int or str) ->
               name}, JSON-safe) supplied by the caller -- when present, it's
               used in place of this module's own
               dataset_info.py/dataset_taxonomy.py-derived mapping for that
               pair only (opt-in, additive; pairs without it are unaffected
               and keep going through _class_idx_to_taxonomy_group as
               before). This lets a caller whose own preprocessing diverges
               from dataset_info.py's schema for a given dataset (e.g.
               optimal_subset_selection's capture24, which dataset_info.py
               still describes with a stale 4-class schema) supply its own
               accurate idx->name mapping instead of hard-crashing here.
        modality: passed through to load_dataset() for every pair.

    Returns:
        X: pooled sensor data (N, C, T)
        Y: dense int class ids in [0, len(group_names)), indexing group_names
        U: user ids namespaced per-dataset as "<dataset>::<raw_user_id>" so
           reused raw user-id numbers across datasets don't collide during
           train/val/test splitting
        group_names: sorted list of canonical taxonomy groups actually present
           across these pairs -- the classifier head for this pooled run must
           be sized len(group_names), not the full ACTIVITY_TAXONOMY, and this
           list is what makes row Y[i]'s class human-readable (group_names[Y[i]])
    """
    if not pairs:
        raise ValueError("load_pooled_datasets() requires at least one pair")

    X_parts, group_parts, U_parts = [], [], []

    for pair in pairs:
        dataset = pair["dataset"]
        sensors = pair["sensors"]
        data_root = pair.get("data_root")

        X, Y_idx, U = load_dataset(dataset, sensors, data_root, modality=modality)

        caller_label_map = pair.get("label_map")
        idx_to_group = _resolve_idx_to_group(dataset, caller_label_map)
        source_desc = "the caller-supplied label_map" if caller_label_map is not None else "dataset_info.py's labels"
        try:
            groups = np.array([idx_to_group[y] for y in Y_idx.tolist()])
        except KeyError as e:
            raise ValueError(
                f"{dataset}: class index {e} from Y.npy has no entry in "
                f"{source_desc} for this dataset -- either an unmapped label "
                f"or a USABLE_CLASSES-remapped index (pooling only supports "
                f"datasets whose indices match {source_desc} directly)"
            ) from e

        # "undefined" (null/transition labels) is an expected drop, not a
        # taxonomy gap -- classify_label_strict() (inside
        # get_dataset_label_mapping) already raised UncategorizedLabelError
        # for any real taxonomy gap, so nothing else needs filtering here.
        keep = groups != "undefined"
        X_parts.append(X[keep])
        group_parts.append(groups[keep])
        U_parts.append(np.array([f"{dataset}::{u}" for u in U[keep]]))

    X = np.concatenate(X_parts, axis=0)
    all_groups = np.concatenate(group_parts, axis=0)
    U = np.concatenate(U_parts, axis=0)

    group_names = sorted(set(all_groups.tolist()))
    group_to_idx = {group: i for i, group in enumerate(group_names)}
    Y = np.array([group_to_idx[group] for group in all_groups], dtype=np.int64)

    return X, Y, U, group_names


def create_dataloaders(X, Y, U, test_users, val_users, batch_size=64, num_workers=0, data_ratio=1.0,
                       use_weighted_sampler=True, max_samples_per_epoch=None,
                       test_mask=None, val_mask=None, return_source_id=False,
                       dataset_weights=None, log_func=None):
    """
    Create DataLoaders for train/val/test.

    Args:
        X: Sensor data (N, C, T)
        Y: Labels (N,)
        U: User IDs (N,)
        test_users: List of test users (e.g., [1, 2])
        val_users: List of validation users (e.g., [3, 4])
        batch_size: Batch size
        num_workers: Number of DataLoader workers (default 0 to avoid conflicts in parallel execution)
        data_ratio: Ratio of training data to use (0.0-1.0)
        use_weighted_sampler: Use WeightedRandomSampler to correct class imbalance
        max_samples_per_epoch: Maximum samples per epoch (None = use training data size)
        test_mask: Precomputed boolean test mask (N,), overrides test_users if given --
            callers use this when a per-user split isn't possible for part of the
            pooled data (e.g. run_finetune_pooled()'s random per-window fallback).
        val_mask: Precomputed boolean val mask (N,), overrides val_users if given.
        return_source_id: When True, derive a per-sample source-dataset id from U's
            "<dataset>::<user>" prefix (same string this function already splits for
            the weighted sampler), thread it into each HARDataset so batches carry a
            3rd (source_id) element, and additionally return a {dataset_name: int}
            id map as a 4th return value. Default off -- existing 3-tuple-unpacking
            callers are unaffected.
        dataset_weights: Optional {dataset_name: weight} to scale each source
            dataset's total per-epoch draw mass in the weighted sampler, on top
            of the existing (dataset, class) balancing. A dataset absent from
            the dict defaults to weight 1.0. None (default) preserves today's
            behavior exactly -- pure uniform (dataset, class) balancing.
        log_func: Optional callable(str). When dataset_weights is given, used to
            log the resulting expected per-dataset sampling fraction (see the
            (dataset, class)-grouping nuance in the docstring below) so it can
            be checked against the requested weights.

    Returns:
        train_loader, val_loader, test_loader
        train_loader, val_loader, test_loader, dataset_id_map  (if return_source_id=True)
    """
    from collections import Counter, defaultdict

    # Split data by user, unless the caller already computed explicit masks.
    test_mask = np.isin(U, test_users) if test_mask is None else test_mask
    val_mask = np.isin(U, val_users) if val_mask is None else val_mask
    train_mask = ~(test_mask | val_mask)

    # A caller-supplied test_users/val_users list that doesn't match any raw
    # ID actually present in U (e.g. finetune.py's FOLDS hardcodes 1-8, but
    # har70plus at some data_roots uses raw ids 501-518) used to silently
    # produce an empty val/test split here -- train_mask would be all-True
    # and training would proceed on 100% of the data with no error, only
    # surfacing later as a metrics crash or silently-meaningless test_f1/acc
    # computed on zero samples. Fail loudly at the actual point of mismatch
    # instead, with enough detail to fix the call site (pass explicit
    # test_mask/val_mask, or --custom_test_users/--custom_val_users where
    # the caller supports it).
    if not test_mask.any():
        raise ValueError(
            f"create_dataloaders: test_users={test_users!r} matches none of the raw user IDs "
            f"present in U (sample of actual IDs: {sorted(set(np.unique(U)))[:10]!r}) -- "
            "empty test split. Pass IDs that exist in this dataset, or provide test_mask explicitly."
        )
    if not val_mask.any():
        raise ValueError(
            f"create_dataloaders: val_users={val_users!r} matches none of the raw user IDs "
            f"present in U (sample of actual IDs: {sorted(set(np.unique(U)))[:10]!r}) -- "
            "empty val split. Pass IDs that exist in this dataset, or provide val_mask explicitly."
        )

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]

    dataset_id_map = None
    source_id_train = source_id_val = source_id_test = None
    if return_source_id:
        ds_prefix_all = np.array([str(u).split("::", 1)[0] if "::" in str(u) else "" for u in U])
        dataset_id_map = {name: i for i, name in enumerate(sorted(set(ds_prefix_all.tolist())))}
        ds_id_all = np.array([dataset_id_map[name] for name in ds_prefix_all], dtype=np.int64)
        source_id_train = ds_id_all[train_mask]
        source_id_val = ds_id_all[val_mask]
        source_id_test = ds_id_all[test_mask]

    # Save original training data count (for samples_per_epoch calculation in few-shot)
    n_train_original = len(X_train)

    # Few-shot: Stratified sampling (guarantee at least 1 sample per class)
    if data_ratio < 1.0:
        n_subset = max(1, int(n_train_original * data_ratio))

        # Select samples from each class
        unique_classes = np.unique(Y_train)
        selected_indices = []

        # First select at least 1 sample from each class
        for cls in unique_classes:
            cls_indices = np.where(Y_train == cls)[0]
            selected_indices.append(np.random.choice(cls_indices, 1)[0])

        # Select remaining samples randomly (without replacement)
        remaining = n_subset - len(selected_indices)
        if remaining > 0:
            all_indices = set(range(n_train_original))
            already_selected = set(selected_indices)
            available = list(all_indices - already_selected)
            if len(available) > 0:
                extra = np.random.choice(available, min(remaining, len(available)), replace=False)
                selected_indices.extend(extra.tolist())

        indices = np.array(selected_indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]
        if source_id_train is not None:
            source_id_train = source_id_train[indices]

    # Create datasets
    train_dataset = HARDataset(X_train, Y_train, source_id=source_id_train)
    val_dataset = HARDataset(X_val, Y_val, source_id=source_id_val)
    test_dataset = HARDataset(X_test, Y_test, source_id=source_id_test)

    # Use WeightedRandomSampler to correct class imbalance, keyed by (source
    # dataset, class) rather than class alone -- with pooled training data
    # (see load_pooled_datasets(), which namespaces U as "<dataset>::<user>"),
    # keying by class alone leaves each class's per-sample draw probability
    # split across its contributing datasets in proportion to their raw
    # counts, so a class dominated by one large dataset would still be
    # learned almost entirely from that dataset's samples. Keying by
    # (dataset, class) instead balances both axes at once. U without "::"
    # (single-dataset runs) collapses to one implicit dataset group, so this
    # is a no-op there -- identical to the old class-only weighting.
    if use_weighted_sampler:
        train_dataset_ids = [str(u).split("::", 1)[0] if "::" in str(u) else "" for u in U[train_mask]]
        group_count = Counter(zip(train_dataset_ids, Y_train.tolist()))
        if dataset_weights:
            group_weights = {
                group: dataset_weights.get(group[0], 1.0) / count
                for group, count in group_count.items()
            }
            if log_func is not None:
                # Per (dataset, class) group, total sampled mass = count * (w_ds / count) = w_ds,
                # so a dataset's total mass is w_ds * (number of its distinct classes) -- see
                # the (dataset, class)-grouping nuance in create_dataloaders' docstring.
                ds_mass = defaultdict(float)
                for (ds, cls), weight in group_weights.items():
                    ds_mass[ds] += weight * group_count[(ds, cls)]
                total_mass = sum(ds_mass.values())
                frac_str = ", ".join(
                    f"{ds}={mass / total_mass:.4f}" for ds, mass in sorted(ds_mass.items())
                )
                log_func(
                    f"create_dataloaders: dataset_weights requested={dataset_weights} -> "
                    f"expected per-dataset sampling fraction={{{frac_str}}}"
                )
        else:
            group_weights = {group: 1.0 / count for group, count in group_count.items()}
        sample_weights = np.array([
            group_weights[(ds, y)] for ds, y in zip(train_dataset_ids, Y_train.tolist())
        ])
        sample_weights = torch.from_numpy(sample_weights).float()

        # Determine number of samples: use original data count as baseline (ensures same update count in few-shot)
        # If max_samples_per_epoch is specified, use the smaller of it and original data count
        if max_samples_per_epoch is not None:
            samples_per_epoch = min(n_train_original, max_samples_per_epoch)
        else:
            samples_per_epoch = n_train_original

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=samples_per_epoch,
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if return_source_id:
        return train_loader, val_loader, test_loader, dataset_id_map
    return train_loader, val_loader, test_loader


# =============================================================================
# For Pretraining (Self-Supervised Learning)
# =============================================================================

class PretrainDataset(Dataset):
    """
    Dataset for pretraining.

    Randomly samples from each file and returns (batch_size, 3, 150).
    DataLoader uses batch_size=1, and squeeze(0) gets (batch_size, 3, 150).
    """

    def __init__(self, file_paths, sample_size=1000, block_size=None):
        """
        Args:
            file_paths: List of npy file paths
            sample_size: Number of windows to return per getitem (effective batch size)
            block_size: (Unused, kept for compatibility)
        """
        self.file_paths = file_paths
        self.sample_size = sample_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        X = np.load(path, mmap_mode='r')

        if X.dtype == np.float16:
            X = np.array(X, dtype=np.float32)

        num_samples = X.shape[0]

        # Random sampling
        if num_samples >= self.sample_size:
            indices = np.random.choice(num_samples, self.sample_size, replace=False)
        else:
            indices = np.random.choice(num_samples, self.sample_size, replace=True)

        data = X[indices]

        # Preprocessing
        data = np.nan_to_num(data, nan=0.0)
        data = np.clip(data, -10.0, 10.0)

        data = torch.tensor(data, dtype=torch.float32)
        return data, data  # For self-supervised learning (sample_size, 3, 150)


def collect_pretrain_files(datasets, sensors=None, data_root=None, modality="ACC"):
    """
    Collect file paths for pretraining.

    Args:
        datasets: List of dataset names, OR list of {"dataset": name, "data_root": path}
            dicts for datasets that don't all share one flat `data_root` (e.g. a
            node-split preprocessed layout where each dataset's real directory lives
            under a different node subdir) -- same pair shape load_pooled_datasets()
            already uses for finetune.py's --baseline_manifest. A dict entry's own
            "data_root" always wins; `data_root` below is only the fallback for plain
            name entries (or a dict entry that omits "data_root").
        sensors: List of sensor names (None = auto-detect all sensors)
        data_root: Data root path, used for any entry that doesn't supply its own
        modality: Modality

    Returns:
        file_paths: List of npy file paths
        skipped_datasets: List of requested dataset names that did not resolve
            to an on-disk directory (tolerant skip, not raised as an error)
    """
    file_paths = []
    skipped_datasets = []

    for entry in datasets:
        if isinstance(entry, dict):
            dataset = entry["dataset"]
            entry_root = entry.get("data_root") or data_root or DEFAULT_DATA_ROOT
        else:
            dataset = entry
            entry_root = data_root or DEFAULT_DATA_ROOT

        dataset_path = _resolve_dataset_dir(entry_root, dataset)
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} not found, skipping")
            skipped_datasets.append(dataset)
            continue

        users = sorted([d for d in os.listdir(dataset_path) if d.startswith("USER")])

        for user in users:
            user_path = os.path.join(dataset_path, user)

            # If sensors is None, auto-detect all sensors
            if sensors is None:
                available_sensors = [
                    d for d in os.listdir(user_path)
                    if os.path.isdir(os.path.join(user_path, d))
                ]
            else:
                available_sensors = sensors

            for sensor in available_sensors:
                x_path = os.path.join(user_path, sensor, modality, "X.npy")
                if os.path.exists(x_path):
                    file_paths.append(x_path)

    return file_paths, skipped_datasets


def create_pretrain_dataloaders(datasets, sensors, data_root=None, modality="ACC",
                                 batch_size=1000, num_workers=4,
                                 train_epoch_size=2000, val_epoch_size=100,
                                 val_ratio=0.1, seed=42, files_per_batch=4):
    """
    Create train/val DataLoaders for pretraining.

    Args:
        datasets: List of dataset names, OR list of {"dataset", "data_root"} dicts --
            see collect_pretrain_files()'s docstring
        sensors: List of sensor names
        data_root: Data root path (fallback for entries that don't supply their own)
        modality: Modality
        batch_size: Batch size (number of windows to read from one file)
        num_workers: Number of workers
        train_epoch_size: Number of batches per training epoch
        val_epoch_size: Number of batches per validation epoch
        val_ratio: Ratio of files for validation
        seed: Seed for splitting
        files_per_batch: Number of files to read simultaneously (4, same as LS-HAR)

    Returns:
        train_loader, val_loader, skipped_datasets
    """
    import random

    file_paths, skipped_datasets = collect_pretrain_files(datasets, sensors, data_root, modality)

    if not file_paths:
        raise ValueError("No data files found for pretraining")

    # Split files into train/val
    random.seed(seed)
    random.shuffle(file_paths)
    n_val = max(1, int(len(file_paths) * val_ratio))
    val_paths = file_paths[:n_val]
    train_paths = file_paths[n_val:]

    print(f"Found {len(file_paths)} files: {len(train_paths)} train, {len(val_paths)} val")
    print(f"Files per batch: {files_per_batch}, Windows per file: {batch_size}")
    print(f"Total windows per batch: {files_per_batch * batch_size}")

    # Train loader
    # LS-HAR: num_samples=1000, batch_size=4 → 250 batches/epoch
    train_dataset = PretrainDataset(train_paths, sample_size=batch_size)
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=train_epoch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=files_per_batch,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val loader
    val_dataset = PretrainDataset(val_paths, sample_size=batch_size)
    val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=val_epoch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=files_per_batch,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader, skipped_datasets
