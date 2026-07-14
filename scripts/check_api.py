"""
Smoke test for the HARBench public API (harbench.load_model / encode).

Run this in an environment that has torch + the bundled mtl weights:

    python scripts/check_api.py

It verifies:
  1. load_model("mtl") loads the bundled weights without error
  2. encode() returns the expected feature shape for raw windowed IMU input
  3. multi-sensor mode produces 512 * num_sensors features
  4. numpy in -> numpy out, and batching gives identical results
"""

import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import available_models, load_model

print("available_models():", available_models())

# --- single sensor: (N, 3, 150) -> (N, 512) -------------------------------
N, seq_len = 8, 150
x = np.random.randn(N, 3, seq_len).astype("float32")

model = load_model("mtl", num_sensors=1, device="cpu")
print("loaded:", model)

feats = model.encode(x)
assert isinstance(feats, np.ndarray), f"expected numpy, got {type(feats)}"
assert feats.shape == (N, 512), f"expected (N, 512), got {feats.shape}"
print("single-sensor encode ok:", feats.shape)

# --- batching must match a single pass ------------------------------------
feats_batched = model.encode(x, batch_size=3)
assert np.allclose(feats, feats_batched, atol=1e-5), "batched != single-pass"
print("batched encode matches single-pass ok")

# --- multi-sensor: (N, 6, 150) -> (N, 1024) -------------------------------
x2 = np.random.randn(N, 6, seq_len).astype("float32")
model2 = load_model("mtl", num_sensors=2, device="cpu")
feats2 = model2.encode(x2)
assert feats2.shape == (N, 1024), f"expected (N, 1024), got {feats2.shape}"
print("multi-sensor encode ok:", feats2.shape)

print("\nALL CHECKS PASSED")
