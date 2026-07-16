---
marp: true
theme: default
paginate: true
title: finetune.py — Engineering Walkthrough
---

# `finetune.py`
### HARBench Evaluation Script — Engineering Walkthrough

Fine-tune & evaluate HAR (Human Activity Recognition) models across
14 backbones, 4 run modes, and 20+ datasets.

---

## Goal & Methodology

**Transfer learning evaluation for HAR**: take a pretrained backbone
(SSL-Wearables ResNet variants, transformers, foundation models),
attach a fresh classifier head, fine-tune on a labeled dataset.

Question it answers: *how well does this pretrained representation
transfer to downstream activity classification*, under different
data regimes:

- Full data (standard fine-tune)
- Few-shot (`--data_ratio`)
- Zero-shot / cross-dataset transfer (LODO)

---

## Four Run Modes

| Mode | Flag | What it does |
|---|---|---|
| `finetune` | *(default)* | 4-fold CV on one `(dataset, sensors)` pair |
| `finetune_single_split` | `--single_split` | Fold 0 only — cheap backbone extraction |
| `finetune_pooled` | `--baseline_manifest` | Joint training across multiple datasets |
| `zeroshot` / `zeroshot_supervised` | `--zeroshot` / `--zeroshot-supervised` | LODO transfer vs. in-dataset upper bound |

All four share the same backbone/train/eval machinery underneath.

---

## Data Flow (shared by all modes)

```
disk (processed_strict npy files)
   │
   ▼  dataloader.load_dataset() / load_pooled_datasets()
X (N,C,T), Y (N,), U (N,) arrays
   │
   ▼  create_dataloaders()
train / val / test DataLoaders   (user-split, class-balanced sampler)
   │
   ▼  create_backbone() + TwoLayerClassifier
model
   │
   ▼  train_model()  →  train_epoch() / evaluate() loop, early stopping
   │
   ▼
results.json + log.txt  →  results/<mode>/<timestamp>_<model>/
```

---

## Model Registry — `MODELS` dict (lines 78–149)

Static registry: `--model` name → `{type, weights, description}`.

- `type` selects which backbone class `create_backbone()` builds
- `weights` is the default pretrained checkpoint (override with `--weights`)

**14 models across 3 families:**
- ResNet-based: `resnet`, `mtl`, `harnet`, `simclr`, `moco`, `timechannel`, `timemask`, `cpc`
- Transformer-based: `selfpab`, `limubert`, `imumae`
- Foundation models: `patchtst`, `moment` (optional deps)

📌 **Adding a new model = add an entry here + branch in `create_backbone()`.**

---

## Core Function: `set_seed(seed)`

```python
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Seeds all RNG sources + forces deterministic cuDNN.
Called once in `main()` before any mode runs.

---

## Core Functions: `train_epoch` / `evaluate`

Generic train/eval loop over a `DataLoader`, with **special-casing for
foundation models**:

- `patchtst`: needs `(batch, seq_len, channels)` not `(batch, channels, seq_len)`;
  reads `.prediction_logits`
- `moment`: calls `.forward(x_enc=...)`; reads `.logits`
- everything else: plain `model(inputs)`

`evaluate()` returns `(loss, macro_f1, accuracy)` via `src/utils.py`.

`train_epoch()`'s `max_iterations` caps batches/epoch — paired with
`--max_samples_per_epoch` for quick/dev runs.

---

## `create_backbone(model_type, weights_path, num_sensors, in_channels, device)`

Factory building the **encoder only** (no classifier head):

| `model_type` | Class |
|---|---|
| `resnet` | `NDeviceResnet` — one ResNet encoder per sensor |
| `maskedresnet` | `MultiDeviceMaskedResnet` |
| `cpc` | `MultiDeviceResnetCPC` |
| `selfpab` | `SelfPAB` (STFT + Transformer) |
| `limubert` | `LIMUBert` (auto-resamples 30Hz→20Hz) |
| `imumae` | `IMUVideoMAE` |
| `harnet` | local `NDeviceHARNet` — torch.hub `harnet5` per sensor |
| `patchtst` / `moment` | returns `None` — built directly in `train_model` |

---

## `_extract_backbone_state_dict(model, model_type)`

Used only when `--save_backbone` is set.

**Why it exists:** `NDeviceResnet` / `MultiDeviceMaskedResnet` /
`MultiDeviceResnetCPC` replicate the *same* per-sensor encoder N times.

- Saving the whole wrapped module → keys prefixed
  `feature_extractors.0....` — **incompatible** with `create_backbone`'s
  strict per-encoder `load_state_dict`
- This function pulls out encoder `[0]`'s state dict only, so the
  saved file round-trips cleanly back through `--weights`

---

## `train_model(...)` — the shared driver

1. Build model (special-cased for `patchtst`/`moment`, else
   `backbone → TwoLayerClassifier`)
2. Adam optimizer + cosine LR schedule (`T_max=args.epochs`) +
   `CrossEntropyLoss`
3. Loop: train epoch → eval on val → track **best-val-F1 checkpoint**
   → early stop after `patience` epochs without improvement
4. Restore best checkpoint → evaluate once on test set
5. Return `{test_f1, test_acc, test_loss, best_val_f1}`
   (+ trained `model` if `return_backbone=True`)

---

## `run_finetune(args)`

Standard 4-fold CV on one `(dataset, sensors)` pair.

- Loads data once via `load_dataset`
- Loops over `FOLDS` (4 fixed splits of raw user IDs `1..8`)
- Per fold: `create_dataloaders` + `train_model`
- Averages F1 / accuracy across folds ± std
- Writes `results/finetune/<timestamp>_<model>/{results.json, log.txt}`

```python
FOLDS = [
    {"test": [1, 2], "val": [3, 4]},
    {"test": [3, 4], "val": [5, 6]},
    {"test": [5, 6], "val": [7, 8]},
    {"test": [7, 8], "val": [1, 2]},
]
```

---

## `run_finetune_single_split(args)`

Identical to `run_finetune`, but **only runs `FOLDS[0]`**.

**Why:** avoids paying for 4-fold training when the goal is just one
trained backbone to persist (`--save_backbone`), not a CV estimate.

---

## `run_finetune_pooled(args)`

Trains **one** model on data pooled from *multiple* `(dataset, sensors,
data_root)` pairs listed in a `--baseline_manifest` JSON file.

- `load_pooled_datasets()` remaps every pair's own labels onto a
  **shared canonical taxonomy** — classifier head size = number of
  taxonomy groups present
- ⚠️ Does **not** reuse `FOLDS`' literal ids `[1,2]/[3,4]` — some
  datasets don't use raw ids `1..8` (e.g. `har70plus` uses `501..518`)
- Instead: **per dataset in the pool**, first 2 sorted raw user IDs →
  test, next 2 → val (positional, not literal) — guarantees 2 test + 2
  val users per dataset regardless of actual ID numbering

---

## `run_zeroshot(args)` + `load_and_map_dataset(...)`

**Leave-One-Dataset-Out (LODO)** evaluation.

For each target in `ZEROSHOT_DATASETS = [dsads, mhealth, pamap2]`:
1. Map target's labels to a shared **6-class taxonomy**
   (`ACTIVITY_MAPPING`) — Static / Walking / Running / Stairs / Jumping / Cycling
2. Train on the union of *all other* zero-shot + support datasets
   (also label-mapped)
3. Test on the untouched target
4. Repeat over 4 seeds, average

⚠️ **Maintenance note:** this is a **hand-rolled training loop**, not a
call to `train_model`/`create_dataloaders` — a near-duplicate of
`train_model`'s loop body (lines 1087–1133). Changes to `train_model`
must be mirrored here manually.

---

## `run_zeroshot_supervised(args)`

The **upper-bound reference** for zero-shot evaluation.

- Same 6-class mapped target datasets (`dsads`, `mhealth`, `pamap2`)
- Standard **in-dataset 4-fold CV** (like `run_finetune`), not LODO
- Lets you compare: *zero-shot transfer F1* vs. *what's achievable
  training/testing on the same dataset*

---

## Two Separate Taxonomies — Don't Conflate

| | `ACTIVITY_MAPPING` | `dataset_taxonomy.get_dataset_label_mapping` |
|---|---|---|
| Used by | zero-shot mode only | pooled training (`load_pooled_datasets`) |
| Classes | 6 hand-authored | canonical taxonomy groups |
| Defined in | `finetune.py` (lines 172–232) | `har-datasets/src/dataset_taxonomy.py` |

Adding a new dataset to one system does **not** register it in the other.

---

## `main()` — CLI Wiring

```
--zeroshot            → run_zeroshot
--zeroshot-supervised → run_zeroshot_supervised
--baseline_manifest    → run_finetune_pooled
--single_split         → run_finetune_single_split
(default)              → run_finetune
```

Validated mutual exclusivity:
- `--save_backbone` requires `--single_split` or `--baseline_manifest`
- `--single_split` and `--baseline_manifest` can't combine

Each mode appends its own subdirectory under `--output_dir` (default `results/`).

---

## Supporting: `src/data/dataloader.py`

**`load_dataset(dataset, sensors, data_root, modality)`**
- Reads `{data_root}/{dataset}/USER{id}/{sensor}/{modality}/{X,Y}.npy`
- Concatenates sensors channel-wise, users row-wise
- Drops unlabeled rows (`Y < 0`)
- Applies `USABLE_CLASSES` allowlist + relabels to consecutive ints
- Returns `(X, Y, U)`

**`load_pooled_datasets(pairs, modality)`**
- Per pair: `load_dataset` → remap to canonical taxonomy groups
- Namespaces user IDs `"<dataset>::<raw_id>"` (avoids ID collisions
  across datasets)
- Drops `"undefined"` (transition/null) labels
- Returns `(X, Y, U, group_names)`

---

## Supporting: `create_dataloaders(...)`

- Splits by user → train / val / test
- Optional stratified few-shot subsampling (`data_ratio`)
- `WeightedRandomSampler` keyed by **(source dataset, class)**, not
  class alone

**Why (dataset, class) and not just class:** in pooled runs, keying by
class alone would let a class dominated by one large contributing
dataset be learned almost entirely from that dataset's samples.
Keying by `(dataset, class)` balances both axes.

For single-dataset runs this collapses to plain class-weighting — no
behavior change.

---

## Maintainer Cheat Sheet

- 🔁 `run_zeroshot`'s inline train loop duplicates `train_model` —
  refactor candidate; until then, mirror changes manually
- 🗂️ `FOLDS`' hardcoded ids `1..8` break for datasets with different
  raw ID ranges — already worked around in the pooled path, watch for
  it elsewhere
- 🧩 Foundation models (`patchtst`, `moment`) are optional deps and
  follow a different code path everywhere — adding a new such model
  touches `create_backbone`, `train_model`, `train_epoch`, `evaluate`,
  **and** `run_zeroshot`'s inline loop
- 🏷️ Two independent label-taxonomy systems exist (`ACTIVITY_MAPPING`
  vs. `dataset_taxonomy`) — know which mode you're extending

---

# Questions?

`finetune.py` · `src/data/dataloader.py` · `src/models/`
