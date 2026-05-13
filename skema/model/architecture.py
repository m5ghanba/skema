"""
skema.model.architecture
~~~~~~~~~~~~~~~~~~~~~~~~~
PyTorch-Lightning segmentation model wrapping SMP's U-Net + MaxViT-Tiny encoder.
"""

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

T_MAX = 50  # cosine-annealing period (epochs); kept here so it's easy to change


class SegModel(pl.LightningModule):
    """
    Binary semantic segmentation model.

    Parameters
    ----------
    arch : str
        SMP architecture name, e.g. ``"Unet"``.
    encoder_name : str
        Timm/SMP encoder name, e.g. ``"tu-maxvit_tiny_tf_512"``.
    in_channels : int
        Number of input channels (10 for S2-only, 13 for full model).
    out_classes : int
        Number of output classes (1 for binary segmentation).
    """

    def __init__(self, arch: str, encoder_name: str, in_channels: int, out_classes: int, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_weights=None,
            **kwargs,
        )
        # Encoder preprocessing params (kept for compatibility; not applied during inference)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std",  torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.training_step_outputs   = []
        self.validation_step_outputs = []
        self.test_step_outputs       = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # -> logits
        return self.model(image)

    # ------------------------------------------------------------------
    # Shared train/val/test logic
    # ------------------------------------------------------------------

    def shared_step(self, batch, stage: str) -> dict:
        image, mask = batch
        assert image.ndim == 4, "Expected (B, C, H, W)"
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, "Spatial dims must be divisible by 32"
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss        = self.loss_fn(logits_mask, mask)
        prob_mask   = logits_mask.sigmoid()
        pred_mask   = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, outputs: list, stage: str) -> None:
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        self.log_dict({
            f"{stage}_per_image_iou": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise"),
            f"{stage}_dataset_iou":   smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_precision":     smp.metrics.precision(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_recall":        smp.metrics.recall(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_f1_score":      smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro"),
        }, prog_bar=True)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, "train")
        self.training_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, "test")
        self.test_step_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }