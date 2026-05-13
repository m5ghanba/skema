"""skema.model — SegModel architecture and weight loader."""

from skema.model.architecture import SegModel
from skema.model.loader import load_model

__all__ = ["SegModel", "load_model"]