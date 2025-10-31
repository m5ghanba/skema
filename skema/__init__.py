"""
SKeMa: Satellite-based Kelp Mapping using Semantic Segmentation on Sentinel-2 imagery
"""

__version__ = "0.1.1"
__author__ = "Mohsen Ghanbari"
__email__ = "mohsenghanbari@uvic.ca"
__doi__ = "10.57967/hf/6790"

# Suppress warnings
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from skema.lib import segment

__all__ = ["segment"]