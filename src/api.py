"""
HARBench Public API

Lightweight entry point for using HARBench feature-extraction models on
your own IMU data, without going through the full training pipeline.

Example
-------
    import harbench

    model = harbench.load_model("mtl")
    feats = model.encode(x)        # x: (N, 3, 150)  ->  feats: (N, 512)

Input format
------------
`encode` expects windowed, channel-first IMU data shaped as:

    (num_windows, num_sensors * 3, sequence_length)

where each sensor contributes 3 channels (x, y, z), in the same order the
model was trained with. The pretrained `mtl` weights were trained on
30 Hz, 5-second windows (sequence_length = 150), so that is the
recommended window size, though the backbone accepts other lengths.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .models import NDeviceResnet

# Candidate directories holding the bundled pretrained weights. Resolved
# relative to this file so it works both from a git checkout (weights live in
# <repo>/pretrained) and from a pip-installed package (weights are bundled at
# harbench/pretrained, i.e. alongside this module).
_HERE = Path(__file__).resolve().parent
_PRETRAINED_DIRS = [
    _HERE / "pretrained",          # pip-installed: harbench/pretrained/mtl.pth
    _HERE.parent / "pretrained",   # git checkout:  <repo>/pretrained/mtl.pth
]


def _resolve_weights(filename: str) -> Path:
    """Return the first existing path to a bundled weight file."""
    for directory in _PRETRAINED_DIRS:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    # Fall back to the pip-installed location for the error message.
    return _PRETRAINED_DIRS[0] / filename

# Registry of models exposed through this API. Only weights bundled with the
# package are listed here.
_MODELS = {
    "mtl": {
        "type": "resnet",
        "weights": "mtl.pth",
        "description": "Multi-Task Learning pretrained 1D ResNet (SSL-Wearables style)",
    },
}


class FeatureExtractor:
    """Wraps a HARBench backbone for easy feature extraction.

    Returned by :func:`load_model`. Call :meth:`encode` to turn windowed
    IMU data into feature vectors.
    """

    def __init__(self, backbone: torch.nn.Module, name: str, device: str):
        self.backbone = backbone.to(device).eval()
        self.name = name
        self.device = device
        self.output_dim = backbone.output_dim

    @torch.no_grad()
    def encode(
        self,
        x: Union[np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Extract feature vectors from windowed IMU data.

        Args:
            x: Array shaped (num_windows, num_sensors * 3, sequence_length).
               Accepts a numpy array or a torch tensor.
            batch_size: Optional chunk size for processing large inputs.
               If None, the whole input is run in a single forward pass.

        Returns:
            numpy array of shape (num_windows, output_dim), where
            output_dim = 512 * num_sensors.
        """
        was_numpy = isinstance(x, np.ndarray)
        tensor = torch.as_tensor(x, dtype=torch.float32)

        if tensor.dim() != 3:
            raise ValueError(
                "encode expects a 3D input shaped "
                "(num_windows, num_sensors * 3, sequence_length), "
                f"got shape {tuple(tensor.shape)}"
            )

        if batch_size is None:
            feats = self._forward(tensor)
        else:
            chunks = [
                self._forward(tensor[i : i + batch_size])
                for i in range(0, tensor.size(0), batch_size)
            ]
            feats = torch.cat(chunks, dim=0)

        return feats.cpu().numpy() if was_numpy else feats

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.backbone(tensor.to(self.device))
        # Flatten (N, 512 * num_sensors, 1) -> (N, 512 * num_sensors)
        return out.reshape(out.size(0), -1)

    def __repr__(self) -> str:
        return (
            f"FeatureExtractor(name={self.name!r}, "
            f"output_dim={self.output_dim}, device={self.device!r})"
        )


def available_models() -> list:
    """Return the list of model names usable with :func:`load_model`."""
    return list(_MODELS)


def load_model(
    name: str = "mtl",
    num_sensors: int = 1,
    device: Optional[str] = None,
) -> FeatureExtractor:
    """Load a HARBench feature-extraction model with bundled pretrained weights.

    Args:
        name: Model name. See :func:`available_models` for the options.
        num_sensors: Number of IMU sensors. Each sensor adds 3 input channels
            (x, y, z) and 512 to the output feature dimension.
        device: Torch device string (e.g. "cuda", "cpu"). Defaults to
            "cuda" when available, otherwise "cpu".

    Returns:
        A :class:`FeatureExtractor` ready for :meth:`FeatureExtractor.encode`.
    """
    if name not in _MODELS:
        raise ValueError(
            f"Unknown model {name!r}. Available models: {available_models()}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = _MODELS[name]
    weights_path = _resolve_weights(config["weights"])
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Pretrained weights for {name!r} not found at {weights_path}. "
            "The weights are expected to ship with the package."
        )

    if config["type"] == "resnet":
        backbone = NDeviceResnet(
            state_dict_path=str(weights_path),
            num_devices=num_sensors,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported model type: {config['type']!r}")

    return FeatureExtractor(backbone, name=name, device=device)
