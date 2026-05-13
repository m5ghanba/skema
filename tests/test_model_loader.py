"""
Tests for skema.model.loader

We mock urllib.request.urlretrieve and torch.load so these tests run
without a network connection or real model weights.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from skema.model.architecture import SegModel
from skema.model.loader import _CACHE_DIR, load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_state_dict(in_channels: int):
    """
    Build a minimal state dict that matches the keys SegModel.load_state_dict
    actually cares about: just 'std' and 'mean' buffers plus an empty model.
    We patch load_state_dict itself, so we only need to make it not raise.
    """
    return {}   # patched load_state_dict ignores this


# ---------------------------------------------------------------------------
# Architecture / SegModel basic tests  (no weights needed)
# ---------------------------------------------------------------------------

class TestSegModelInstantiation:
    def test_s2_model_creates_without_error(self):
        model = SegModel("Unet", "tu-maxvit_tiny_tf_512", in_channels=10, out_classes=1)
        assert isinstance(model, SegModel)

    def test_full_model_creates_without_error(self):
        model = SegModel("Unet", "tu-maxvit_tiny_tf_512", in_channels=13, out_classes=1)
        assert isinstance(model, SegModel)

    def test_forward_returns_correct_shape(self):
        """Smoke-test forward pass with a tiny random input (CPU only)."""
        model = SegModel("Unet", "tu-maxvit_tiny_tf_512", in_channels=10, out_classes=1)
        model.eval()
        # Spatial dims must be divisible by 32; use 32×32
        x   = torch.zeros(1, 10, 32, 32)
        out = model(x)
        assert out.shape == (1, 1, 32, 32)

    def test_loss_fn_is_dice(self):
        model = SegModel("Unet", "tu-maxvit_tiny_tf_512", in_channels=10, out_classes=1)
        import segmentation_models_pytorch as smp
        assert isinstance(model.loss_fn, smp.losses.DiceLoss)


# ---------------------------------------------------------------------------
# load_model  (patched — no network, no disk)
# ---------------------------------------------------------------------------

class TestLoadModel:
    """Patches urllib download + torch.load so no network is required."""

    def _patch_load(self, in_channels: int):
        """Context-manager helper: patches download and weight loading."""
        from unittest.mock import patch as _patch

        def fake_load_state_dict(self, state_dict, strict=True, assign=False):
            pass  # do nothing

        return [
            _patch("urllib.request.urlretrieve"),
            _patch("torch.load", return_value={}),
            _patch.object(SegModel, "load_state_dict", fake_load_state_dict),
            _patch("os.path.exists", return_value=True),   # pretend file already cached
        ]

    def test_load_s2_model_returns_segmodel(self):
        patches = self._patch_load(in_channels=10)
        with patches[0], patches[1], patches[2], patches[3]:
            model = load_model("model_s2bandsandindices_only")
        assert isinstance(model, SegModel)

    def test_load_full_model_rf_returns_segmodel(self):
        patches = self._patch_load(in_channels=13)
        with patches[0], patches[1], patches[2], patches[3]:
            model = load_model("model_full", use_bops_substrate=False)
        assert isinstance(model, SegModel)

    def test_load_full_model_bops_returns_segmodel(self):
        patches = self._patch_load(in_channels=13)
        with patches[0], patches[1], patches[2], patches[3]:
            model = load_model("model_full", use_bops_substrate=True)
        assert isinstance(model, SegModel)

    def test_load_ensemble_returns_tuple(self):
        patches = self._patch_load(in_channels=10)
        with patches[0], patches[1], patches[2], patches[3]:
            result = load_model("model_ensemble")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(m, SegModel) for m in result)

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="Invalid model_type"):
            load_model("totally_made_up_type")

    def test_download_skipped_when_file_exists(self):
        patches = self._patch_load(in_channels=10)
        with patches[0] as mock_download, patches[1], patches[2], patches[3]:
            load_model("model_s2bandsandindices_only")
        # os.path.exists returns True, so urlretrieve should NOT be called
        mock_download.assert_not_called()