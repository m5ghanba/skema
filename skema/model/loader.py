"""
skema.model.loader
~~~~~~~~~~~~~~~~~~~
Downloads model weights from HuggingFace (if not cached) and returns
a ready-to-use SegModel instance (or a tuple of two for ensemble mode).
"""

import os
import urllib.request

import torch

from skema.model.architecture import SegModel

__version__ = "0.3.4"     # kept in sync with __init__.py

_HF_BASE = "https://huggingface.co/m5ghanba/SKeMa/resolve/main"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".skema")

OUT_CLASSES = 1


def _model_cache_path(filename: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, filename)


def _download_if_missing(url: str, local_path: str) -> None:
    if not os.path.exists(local_path):
        print(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, local_path)
        print("Download complete.")


def _load_single(url: str, filename: str, in_channels: int) -> SegModel:
    local_path = _model_cache_path(filename)
    _download_if_missing(url, local_path)
    model = SegModel("Unet", "tu-maxvit_tiny_tf_512", in_channels=in_channels, out_classes=OUT_CLASSES)
    model.load_state_dict(torch.load(local_path, map_location="cpu"))
    return model


def load_model(model_type: str = "model_full", use_bops_substrate: bool = False):
    """
    Return the appropriate model(s) for *model_type*.

    Parameters
    ----------
    model_type : str
        One of ``"model_full"``, ``"model_s2bandsandindices_only"``, ``"model_ensemble"``.
    use_bops_substrate : bool
        When True, use weights trained on BoPs substrate data (only affects
        ``model_full`` and ``model_ensemble``).

    Returns
    -------
    SegModel or (SegModel, SegModel)
        For ``model_ensemble`` a tuple ``(model_full, model_s2)`` is returned.
    """
    if model_type == "model_full":
        if use_bops_substrate:
            url      = f"{_HF_BASE}/model_full_bops_subs.pth"
            filename = f"model_full_bops_subs_v{__version__}.pth"
        else:
            url      = f"{_HF_BASE}/model_full_rf_subs.pth"
            filename = f"model_full_rf_subs_v{__version__}.pth"
        return _load_single(url, filename, in_channels=13)

    elif model_type == "model_s2bandsandindices_only":
        url      = f"{_HF_BASE}/modelS2Only.pth"
        filename = f"modelS2Only_v{__version__}.pth"
        return _load_single(url, filename, in_channels=10)

    elif model_type == "model_ensemble":
        print("Loading ensemble models...")
        if use_bops_substrate:
            url_full      = f"{_HF_BASE}/model_full_bops_subs.pth"
            filename_full = f"model_full_bops_subs_v{__version__}.pth"
        else:
            url_full      = f"{_HF_BASE}/model_full_rf_subs.pth"
            filename_full = f"model_full_rf_subs_v{__version__}.pth"

        model_full = _load_single(url_full, filename_full, in_channels=13)

        url_s2      = f"{_HF_BASE}/modelS2Only.pth"
        filename_s2 = f"modelS2Only_v{__version__}.pth"
        model_s2    = _load_single(url_s2, filename_s2, in_channels=10)

        print("Both models loaded successfully.")
        return (model_full, model_s2)

    else:
        raise ValueError(
            f"Invalid model_type '{model_type}'. "
            "Must be 'model_full', 'model_s2bandsandindices_only', or 'model_ensemble'."
        )