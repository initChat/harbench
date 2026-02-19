# Contributing to HARBench

Thank you for your interest in contributing to HARBench! This guide walks you through adding a new model or method to the benchmark.

**Target audience**: Researchers and developers who want to evaluate their own HAR model on HARBench.

## Table of Contents

- [Development Setup](#development-setup)
- [Adding a New Model](#adding-a-new-model)
  - [Step 1: Implement Your Backbone](#step-1-implement-your-backbone)
  - [Step 2: Register Your Model](#step-2-register-your-model)
  - [Step 3: Export Your Class](#step-3-export-your-class)
  - [Step 4: (Optional) Add SSL Pretraining](#step-4-optional-add-ssl-pretraining)
  - [Step 5: Verify Your Model](#step-5-verify-your-model)
- [Code Style](#code-style)
- [Submitting Your Contribution](#submitting-your-contribution)

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/<your-username>/har-bench2.git
cd har-bench2
```

2. Set up the environment:

```bash
python -m venv .env
source .env/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

3. Preprocess a small dataset for testing:

```bash
python preprocess.py --dataset dsads --download
```

See the [README](README.md) for full setup instructions including Docker.

## Adding a New Model

Adding a model involves 3 required steps and 1 optional step:

| Step | File | Required |
|------|------|----------|
| 1. Implement backbone | `src/models/backbones.py` | Yes |
| 2. Register model | `finetune.py`, `run_benchmark.py` | Yes |
| 3. Export class | `src/models/__init__.py` | Yes |
| 4. SSL pretraining | `pretrain.py` | Optional |

### Step 1: Implement Your Backbone

Add your backbone class to `src/models/backbones.py`.

**Requirements:**

- Inherit from `nn.Module`
- Set `self.output_dim` in `__init__` — the classifier head uses this to determine input size
- Support multi-sensor input via `num_devices` parameter
- Input shape: `(batch, num_devices * 3, 150)` — 30 Hz sampling, 5-second window, 3 axes (x, y, z)
- Output shape: `(batch, output_dim)` or `(batch, output_dim, 1)`

**Skeleton:**

```python
class MyBackbone(nn.Module):
    """One-line description of your model."""

    def __init__(self, num_devices=1, pretrained_path=None, device="cpu"):
        super().__init__()
        self.num_devices = num_devices
        self.output_dim = 256 * num_devices  # Must be set

        # Your layers here
        # ...

        # Load pretrained weights if provided
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
            self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        """
        Args:
            x: (batch, num_devices * 3, 150)
        Returns:
            (batch, output_dim) or (batch, output_dim, 1)
        """
        # Split input by device, process each, concatenate
        outputs = []
        for i in range(self.num_devices):
            x_i = x[:, i*3:(i+1)*3, :]  # (batch, 3, 150)
            out = self.encoder(x_i)       # Replace with your own layers
            outputs.append(out)
        return torch.cat(outputs, dim=1)
```

**Existing examples to reference:**

- `NDeviceResnet` (line ~212) — standard multi-device CNN pattern with shared weights
- `SelfPAB` (line ~686) — Transformer-based model with STFT preprocessing and resampling
- `LIMUBert` (line ~356) — Transformer encoder with automatic resampling to different Hz

### Step 2: Register Your Model

Edit `finetune.py` and `run_benchmark.py` to make your model available via the `--model` flag.

**2a. Add to `MODELS` dict in `finetune.py`** (line ~78):

```python
MODELS = {
    # ... existing entries ...

    # Your model
    "mymodel": {
        "type": "mymodel",
        "weights": "pretrained/mymodel.pth",  # omit if no pretrained weights
        "description": "My Custom Model (brief description)",
    },
}
```

**2b. Add branch to `create_backbone()`** (line ~326):

```python
def create_backbone(model_type, weights_path, num_sensors, in_channels, device):
    # ... existing branches ...

    elif model_type == "mymodel":
        backbone = MyBackbone(
            num_devices=num_sensors,
            pretrained_path=weights_path,
            device=device,
        )

    # ... rest of function ...
```

**2c. Add to `MODELS` list in `run_benchmark.py`** (line ~44):

```python
MODELS = [
    # ... existing entries ...
    "mymodel",
]
```

> **Note:** Foundation models that don't follow the backbone + classifier pattern (like PatchTST and MOMENT) return `None` from `create_backbone()` and are handled directly in `train_model()` (line ~423). See the `patchtst` branch for an example.

### Step 3: Export Your Class

Edit `src/models/__init__.py` to make your class importable:

```python
from .backbones import (
    Resnet,
    NDeviceResnet,
    # ... existing imports ...
    MyBackbone,          # Add your class
)

__all__ = [
    "Resnet",
    "NDeviceResnet",
    # ... existing entries ...
    "MyBackbone",        # Add your class
]
```

### Step 4: (Optional) Add SSL Pretraining

If your model uses self-supervised pretraining, add it to `pretrain.py`.

**4a. Add to `SSL_METHODS`** (line ~48):

```python
SSL_METHODS = ["mtl", "simclr", "moco", "cpc", "timemask", "timechannel", "mymethod"]
```

**4b. Implement your SSL model class and loss function:**

The SSL model wraps a backbone and adds task-specific heads (e.g., projection head for contrastive learning). The loss function receives raw batches and is responsible for augmentation, forward pass, and loss computation.

```python
class MySSLModel(nn.Module):
    """SSL pretraining wrapper for your method."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # Must be stored as self.backbone (see 4e)
        self.projector = nn.Linear(backbone.output_dim, 256)

    def forward(self, x):
        features = self.backbone(x)
        features = features.reshape(features.size(0), -1)
        return self.projector(features)

def compute_mymethod_loss(model, batch, device, use_rotation=True):
    """
    Compute SSL loss.

    Args:
        model: Your SSL model instance
        batch: Raw input tensor (batch, channels, seq_len)
        device: torch device
        use_rotation: Whether to apply 3D rotation augmentation

    Returns:
        (loss, loss_value, metric) — metric can be 0.0 if unused
    """
    x = batch.to(device)
    if use_rotation:
        x = rotation_3d(x)
    # Your augmentation and loss computation here
    output = model(x)
    loss = ...
    return loss, loss.item(), 0.0
```

See `SimCLRModel` (line ~378) and `TimechannelModel` (line ~763) for concrete examples.

**4c. Register in `get_loss_function()`** (line ~867):

```python
loss_functions = {
    # ... existing entries ...
    "mymethod": compute_mymethod_loss,
}
```

**4d. Register in `create_model()`** (line ~945):

```python
elif method == "mymethod":
    model = MySSLModel(backbone)
```

**4e. Note on `get_backbone_state_dict()`** (line ~971):

After pretraining, `get_backbone_state_dict()` extracts the backbone weights for downstream fine-tuning. It checks for `model.backbone` by default, so **store your backbone as `self.backbone`** and it will work automatically. If your architecture uses a different attribute name (like MoCo's `encoder_q`), add a branch here.

**4f. Run pretraining:**

```bash
python pretrain.py --method mymethod --datasets DSADS PAMAP2 --sensors back thigh
```

Pretrained weights are saved to `results/pretrain/`. Copy the backbone weights to `pretrained/` for use in fine-tuning.

### Step 5: Verify Your Model

Make sure you have preprocessed at least `dsads` (see [Development Setup](#development-setup)), then run these commands:

```bash
# Fine-tune on a single dataset (quick sanity check)
python finetune.py --model mymodel --dataset dsads --sensors LeftArm LeftLeg --epochs 3

# Zero-shot evaluation (LODO across DSADS, MHEALTH, PAMAP2)
python finetune.py --model mymodel --zeroshot

# Full benchmark (all 5 metrics across 18 datasets)
python run_benchmark.py --model mymodel --eval all

# Run a specific evaluation type
python run_benchmark.py --model mymodel --eval average
python run_benchmark.py --model mymodel --eval zeroshot
```

If you have pretrained weights, place them in `pretrained/` and verify the `--weights` path works:

```bash
python finetune.py --model mymodel --weights pretrained/mymodel.pth --dataset dsads --sensors LeftArm
```

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Add a docstring to your backbone class describing the model architecture
- Use section separators for major code blocks:

```python
# =============================================================================
# My Model
# =============================================================================
```

- Use `torch.load(..., weights_only=True)` when loading state dicts
- Keep pretrained weight loading in `__init__` rather than `forward`

## Submitting Your Contribution

### 1. Open an Issue First

Before writing code, open a GitHub Issue to discuss your proposed model. This prevents duplicate work and allows maintainers to provide early feedback.

**Issue should include:**

- Model/method name and brief description
- Link to the paper or repository
- Whether pretrained weights are available
- Any additional dependencies required

### 2. Branch Naming

```
model/add-mymodel
fix/mymodel-loading-issue
docs/mymodel-description
```

### 3. Commit Messages

Follow the `<type>:<description>` format used in this repository:

```
feature: add MyModel backbone and pretrained weights
fix: mymodel weight loading for multi-device
docs: add MyModel to supported models table
```

Types: `feature`, `fix`, `update`, `docs`, `change`

### 4. Run the Full Benchmark

Before submitting a PR, run the complete benchmark and include **all 5 metric scores** in your PR description. The maintainer will calculate the final leaderboard scores from your raw results.

```bash
python run_benchmark.py --model mymodel --eval all
```

### 5. Pull Request Checklist

- [ ] GitHub Issue opened and linked in PR description
- [ ] Backbone implements `output_dim` and supports `num_devices`
- [ ] Model registered in `MODELS` dict, `create_backbone()`, and `run_benchmark.py`
- [ ] Class exported in `src/models/__init__.py`
- [ ] `python finetune.py --model mymodel --dataset dsads --sensors LeftArm` runs without errors
- [ ] Full benchmark results (`--eval all`) included in PR description
- [ ] Pretrained weights (if any) are available for download or included in `pretrained/`
- [ ] README `Supported Models` table updated with your model entry

### 6. PR Description Template

```
## What
Brief description of the model/method being added.

Closes #<issue-number>

## Reference
Link to the paper or repository.

## Changes
- Added `MyBackbone` to `src/models/backbones.py`
- Registered `mymodel` in `finetune.py` and `run_benchmark.py`
- Exported class in `src/models/__init__.py`
- Added pretrained weights to `pretrained/`

## Benchmark Results
| Metric | Score |
|--------|-------|
| Average Performance (F1) | 0.XX |
| Domain Robustness (F1) | 0.XX |
| Position Robustness (F1) | 0.XX |
| Few-shot Performance (F1) | 0.XX |
| Zero-shot Performance (F1) | 0.XX |
```
