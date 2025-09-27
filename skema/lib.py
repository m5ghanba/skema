import rasterio
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import glob
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trns
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from torchmetrics import JaccardIndex
from sklearn.model_selection import train_test_split

import os
import albumentations as A

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import random

# to save the results
import tifffile as tiff  # Use tifffile to handle TIFF files
import urllib.request
import getpass


print(torch.__version__)
print(torch.version.cuda)  # Should print the version of CUDA PyTorch is using
print("cuDNN version:", torch.backends.cudnn.version())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the device to GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)




def normalize_input_mean_std(image_hwc, mean_per_channel, std_per_channel, epsilon=1e-8):
    """Applies mean and std normalization to the input image (H, W, C) at once."""
    image_hwc = np.nan_to_num(image_hwc).astype(np.float32)  # Handle NaNs and ensure float type
    mean = np.array(mean_per_channel, dtype=np.float32)[np.newaxis, np.newaxis, :]
    std = np.array(std_per_channel, dtype=np.float32)[np.newaxis, np.newaxis, :]
    normalized_image = (image_hwc - mean) / (std + epsilon)
    return normalized_image

class SatelliteDataset(BaseDataset):
    CLASSES = ["water", "kelp", "land"]

    def __init__(self, image_paths, mask_paths, classes=None, augmentation=None, mean=None, std=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.mean = mean
        self.std = std
        self.calculated_mean = None
        self.calculated_std = None

        if classes is None:
            classes = self.CLASSES
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.index_calculators = {
            "ndvi": self.calculate_ndvi,
            "ndwi": self.calculate_ndwi,
            "gndvi": self.calculate_gndvi,
            "clgreen": self.calculate_chlorophyll_index_green,
            "ndvire": self.calculate_ndvi_re, #Normalized Difference of Red and Blue
            #"ndrb": self.calculate_ndrb, #Normalized Difference of Red and Blue
            #"mgvi": self.calculate_mgvi, #Modified Green Red Vegetation Index (MGVI)
            #"mpri": self.calculate_mpri, #Modified Photochemical Reflectance Index (MPRI)
            #"rgbvi": self.calculate_rgbvi, #Red Green Blue Vegetation Index (RGBVI)
            #"gli": self.calculate_gli, #Green Leaf Index (GLI)
            #"gi": self.calculate_gi, #Greenness Index (GI)
            #"br": self.calculate_blue_red, #Blue/Red
            #"exg": self.calculate_exg, #Excess of Green (ExG)
            #"vari": self.calculate_vari, #Visible Atmospherically Resistant Index (VARI)
            #"tvi": self.calculate_tvi, #Triangular Vegetation Index (TVI)
            #"rdvi": self.calculate_rdvi, #Renormalized Difference Vegetation Index (RDVI)
            #"ndreb": self.calculate_ndreb, #Normalized Difference Red-edge Blue (NDREB)
            #"evi": self.calculate_evi, #Enhanced Vegetation Index (EVI)
            #"cig": self.calculate_cig,  #Green Chlorophyll Index (CIG)
            #"blue_rededge": self.calculate_blue_rededge, #Blue/Red-edge
            #"bnir": self.calculate_blue_nir, #Blue/NIR
            #"rb": self.calculate_red_minus_blue, #R-B
            #"bndvi": self.calculate_bndvi, #Blue NDVI
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # --- 1. Read Image ---
        img_path = self.image_paths[index]
        with rasterio.open(img_path) as src_img:
            image = src_img.read([1, 2, 3, 4, 5, 11, 12]) # -> (C, H, W) = (6, H, W) 1-4:10m bands, 5-10: 20m bands resampled to 10m, 11:substrate 12: bathymetry

            # --- FIXED PREPROCESSING ---
            # # Modify bathymetry (channel index 5 in CHW format)
            # bathy_mask_gt = image[6, :, :] > 10
            # bathy_mask_lt = image[6, :, :] < -100
            # image[6, :, :][bathy_mask_gt | bathy_mask_lt] = -2000 # Combine conditions

            # # Modify substrate (channel index 4 in CHW format)
            # subs_mask = image[5, :, :] != 1
            # image[5, :, :][subs_mask] = 0
            # # --- END FIX ---

            image_hwc = np.transpose(image, (1, 2, 0)).astype(np.float32) # -> (H, W, 6)

        # --- 2. Calculate Indices ---
        indices_list = []
        for _, calculator in self.index_calculators.items():
            idx = calculator(image_hwc)
            indices_list.append(idx[..., np.newaxis])

        image_with_indices = np.concatenate([image_hwc] + indices_list, axis=-1)

        # --- 3. Read Mask ---
        mask_path = self.mask_paths[index]
        mask = self.read_and_process_mask(mask_path) # -> (H, W, num_classes)

        # --- 4. Apply Normalization ---
        if self.mean is not None and self.std is not None:
            image_with_indices = normalize_input_mean_std(image_with_indices, mean_per_channel=self.mean, std_per_channel=self.std)

        # --- 5. Apply Augmentation ---
        if self.augmentation:  # Proceeds to call whatever is stored in self.augmentation
            # Python expects self.augmentation to be a callable object (something that can be called using parentheses ()), which a function is. 
            # The image and mask variables are passed as arguments to this callable. In the case of albumentation, self.augmentation holds the
            # A.Compose object. When you call it with (image=image_with_indices, mask=mask), the A.Compose object's __call__ method  (which is what makes an
            # object callable) is executed. This method internally applies the defined horizontal and vertical flips (with their respective probabilities)
            # to the provided image and mask and returns a dictionary like {'image': augmented_image, 'mask': augmented_mask}.
            sample = self.augmentation(image=image_with_indices, mask=mask) 
            image_with_indices = sample['image']
            mask = sample['mask']

        # --- 6. Final Transpose ---
        image_final = np.transpose(image_with_indices, (2, 0, 1))
        mask_final = np.transpose(mask, (2, 0, 1))

        return image_final.astype(np.float32), mask_final.astype(np.float32)


    def read_and_process_mask(self, mask_path):
        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype(int) # -> (H, W)
            masks = [(mask == v) for v in self.class_values]
            return np.stack(masks, axis=-1).astype("float") #-> (H, W, num_classes)

    # --- Index Calculation Methods ---
    def calculate_ndvi(self, image_hwc):
        nir = image_hwc[..., 3]
        red = image_hwc[..., 2]
        return (nir - red) / (nir + red + 1e-10)

    def calculate_ndwi(self, image_hwc):
        green = image_hwc[..., 1]
        nir = image_hwc[..., 3]
        return (green - nir) / (green + nir + 1e-10)

    def calculate_gndvi(self, image_hwc):
        nir = image_hwc[..., 3]
        green = image_hwc[..., 1]
        return (nir - green) / (nir + green + 1e-10)

    def calculate_chlorophyll_index_green(self, image_hwc):
        nir = image_hwc[..., 3]
        green = image_hwc[..., 1]
        return (nir / (green + 1e-10)) - 1

    def calculate_ndvi_re(self, image_hwc):
        re = image_hwc[..., 4]
        red = image_hwc[..., 2]
        return (re - red) / (re + red + 1e-10)

    def calculate_evi(self, image_hwc):
        nir = image_hwc[..., 3]
        red = image_hwc[..., 2]
        blue = image_hwc[..., 0]
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10)

    def calculate_sr(self, image_hwc):
        nir = image_hwc[..., 3]
        red = image_hwc[..., 2]
        return nir / (red + 1e-10)

    def calculate_ndrb(self, image_hwc):
        return (image_hwc[..., 2] - image_hwc[..., 0]) / (image_hwc[..., 2] + image_hwc[..., 0] + 1e-10)

    def calculate_mgvi(self, image_hwc):
        return (image_hwc[..., 1]**2 - image_hwc[..., 2]**2) / (image_hwc[..., 1]**2 + image_hwc[..., 2]**2 + 1e-10)

    def calculate_mpri(self, image_hwc):
        return (image_hwc[..., 1] - image_hwc[..., 2]) / (image_hwc[..., 1] + image_hwc[..., 2] + 1e-10)

    def calculate_rgbvi(self, image_hwc):
        return (image_hwc[..., 1] - image_hwc[..., 0] * image_hwc[..., 2]) / (image_hwc[..., 1]**2 + image_hwc[..., 0] * image_hwc[..., 2] + 1e-10)

    def calculate_gli(self, image_hwc):
        return (2 * image_hwc[..., 1] - image_hwc[..., 2] - image_hwc[..., 0]) / (2 * image_hwc[..., 1] + image_hwc[..., 2] + image_hwc[..., 0] + 1e-10)

    def calculate_gi(self, image_hwc):
        return image_hwc[..., 1] / (image_hwc[..., 2] + 1e-10)

    def calculate_blue_red(self, image_hwc):
        return image_hwc[..., 0] / (image_hwc[..., 2] + 1e-10)

    def calculate_red_minus_blue(self, image_hwc):
        return image_hwc[..., 2] - image_hwc[..., 0]

    def calculate_exg(self, image_hwc):
        return 2 * image_hwc[..., 1] - image_hwc[..., 2] - image_hwc[..., 0]

    def calculate_vari(self, image_hwc):
        return (image_hwc[..., 1] - image_hwc[..., 2]) / (image_hwc[..., 1] + image_hwc[..., 2] - image_hwc[..., 0] + 1e-10)

    def calculate_tvi(self, image_hwc):
        return (120 * (image_hwc[..., 4] - image_hwc[..., 1]) - 200 * (image_hwc[..., 2] - image_hwc[..., 1])) / 2

    def calculate_rdvi(self, image_hwc):
        return (image_hwc[..., 3] - image_hwc[..., 2]) / np.sqrt(image_hwc[..., 3] + image_hwc[..., 2] + 1e-10)

    def calculate_ndreb(self, image_hwc):
        return (image_hwc[..., 4] - image_hwc[..., 0]) / (image_hwc[..., 4] + image_hwc[..., 0] + 1e-10)

    def calculate_cig(self, image_hwc):
        return (image_hwc[..., 3] / (image_hwc[..., 1] + 1e-10)) - 1

    def calculate_blue_rededge(self, image_hwc):
        return image_hwc[..., 0] / (image_hwc[..., 4] + 1e-10)

    def calculate_blue_nir(self, image_hwc):
        return image_hwc[..., 0] / (image_hwc[..., 3] + 1e-10)

    def calculate_bndvi(self, image_hwc):
        nir = image_hwc[..., 3]
        blue = image_hwc[..., 0]
        return (nir - blue) / (nir + blue + 1e-10)




OUT_CLASSES = 1

class segModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_weights=None,
            **kwargs,
        )
        # preprocessing parameteres for image (Cuurently no normalization, so the next few lines do not do anything--Normalizing data is most beneficial when input features have a high variance or differ significantly from what the model was trained on, which could lead to poor learning and performance.)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std //no normalization
        mask = self.model(image)
        return mask

    # How shared_step and shared_epoch_end Work Together
    # shared_step is called for each batch and returns the loss and segmentation statistics (tp, fp, fn, tn).
    # The results from multiple shared_step calls are collected into a list (e.g., self.training_step_outputs).
    # At the end of an epoch, shared_epoch_end aggregates all statistics and computes IoU metrics.
    def shared_step(self, batch, stage):
        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # def shared_epoch_end(self, outputs, stage):
    #     # aggregate step metics
    #     tp = torch.cat([x["tp"] for x in outputs])
    #     fp = torch.cat([x["fp"] for x in outputs])
    #     fn = torch.cat([x["fn"] for x in outputs])
    #     tn = torch.cat([x["tn"] for x in outputs])

    #     # per image IoU means that we first calculate IoU score for each image
    #     # and then compute mean over these scores
    #     per_image_iou = smp.metrics.iou_score(
    #         tp, fp, fn, tn, reduction="micro-imagewise"
    #     )

    #     # dataset IoU means that we aggregate intersection and union over whole dataset
    #     # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
    #     # in this particular case will not be much, however for dataset
    #     # with "empty" images (images without target class) a large gap could be observed.
    #     # Empty images influence a lot on per_image_iou and much less on dataset_iou. 
    #     # When we say that empty images influence per_image_iou a lot, it means they tend to increase it. 
    #     # This is because empty images typically have high IoU, and when computing per_image_iou (i.e., averaging IoU across all images),
    #     # empty images dominate the average because they tend to have high IoU values.
    #     dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    #     metrics = {
    #         f"{stage}_per_image_iou": per_image_iou,
    #         f"{stage}_dataset_iou": dataset_iou,
    #     }

    #     self.log_dict(metrics, prog_bar=True)

    # Mohsen Ghanbari Feb 2025 - Revised this fucntion to output other metrics as well
    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
    
        # Compute IoU:
        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou. 
        # When we say that empty images influence per_image_iou a lot, it means they tend to increase it. 
        # This is because empty images typically have high IoU, and when computing per_image_iou (i.e., averaging IoU across all images),
        # empty images dominate the average because they tend to have high IoU values.
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    
        # Compute additional metrics
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    
        # Log metrics
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_precision": precision,
            f"{stage}_recall": recall,
            f"{stage}_f1_score": f1_score,
            # f"{stage}_tp": tp.sum().item(),
            # f"{stage}_fp": fp.sum().item(),
            # f"{stage}_fn": fn.sum().item(),
            # f"{stage}_tn": tn.sum().item(),
        }
    
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return


# https://smp.readthedocs.io/en/latest/models.html#upernet     encoders: https://smp.readthedocs.io/en/latest/encoders.html
model = segModel("Unet", "tu-maxvit_tiny_tf_512", in_channels=12, out_classes=OUT_CLASSES)#decoder_channels=(512, 256, 128, 64), encoder_depth=4,  resnet18

# # Load the full Lightning checkpoint
# model = segModel(model_name, encoder_name, in_channels=8, out_classes=OUT_CLASSES)
# model = model.load_from_checkpoint(os.path.join(save_dir, checkpoint_filename))

# Load only the state dict

# save_dir = "C:/Users/mohsenghanbari/saved_models/DataNoError1percent10and20mbandsPlusSubstrateAndBathy/NewResults" #"C:/Users/mohsenghanbari/saved_models/AnnotatedData/uploaded to git"  #"C:/Users/mohsenghanbari/saved_models/DataNoError1percent10and20mbandsPlusSubstrateAndBathy/NewResults"
# state_dict_filename = "NormalizedInput_NoAugmentation_WithSubstrate_WithBathy_WithIndices_WithB5andNDVIRE_Unet_resnet18_20250416_191617.pth" #"model.pth" #"NormalizedInput_NoAugmentation_WithSubstrate_WithBathy_WithIndices_WithB5andNDVIRE_Unet_resnet18_20250416_191617.pth"
# model.load_state_dict(torch.load(os.path.join(save_dir, state_dict_filename)))



# instead load it from github 
import torch
import os
import urllib.request

def download_model(url, destination):
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, destination)
    print("Download complete.")
    return destination

MODEL_URL = "https://github.com/m5ghanba/labeled_pixel_collector/releases/download/v0.1.1-alpha/model.pth"
LOCAL_PATH = os.path.join(os.path.expanduser("~"), ".skema", "model.pth")

# Ensure directory exists
os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)

# Download if not exists
model_path = download_model(MODEL_URL, LOCAL_PATH)

# Load model
model.load_state_dict(torch.load(model_path, map_location="cpu"))




# # Set model to evaluation mode
# model.eval()


# supports having both 10m and 20m bands as well as substrate and bathymetry channels.
# also normalization is added here. Set mean and std to None if it was off during training. Otherwise, it is assumed mean_per_channel and 
# std_per_channel are provided. 
import os
import glob
import rasterio
import numpy as np
import torch
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from tqdm import tqdm


def normalize_tile_mean_std(tile, mean_per_channel, std_per_channel):
    """Applies mean and std normalization to each channel of the input tile (H, W, C) using vectorization."""
    tile = np.nan_to_num(tile).astype(np.float32)  # Handle NaNs and ensure float type
    mean = np.array(mean_per_channel, dtype=np.float32)
    std = np.array(std_per_channel, dtype=np.float32)
    # Reshape mean and std to match tile's shape for broadcasting
    mean = mean[np.newaxis, np.newaxis, :]
    std = std[np.newaxis, np.newaxis, :]
    normalized_tile = (tile - mean) / (std + 1e-8)
    return normalized_tile

def create_weight_map(tile_size, halo_size):
    """Create a weight map that gives less weight to edge pixels and more to center."""
    weight_map = np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Create a linear fade from edge to center
    for i in range(halo_size):
        fade_weight = (i + 1) / halo_size
        # Top and bottom edges
        weight_map[i, :] = fade_weight
        weight_map[tile_size-1-i, :] = fade_weight
        # Left and right edges
        weight_map[:, i] = np.minimum(weight_map[:, i], fade_weight)
        weight_map[:, tile_size-1-i] = np.minimum(weight_map[:, tile_size-1-i], fade_weight)
    
    return weight_map
#  July 2025 - Same as above but with these modifications: 
# 1.padding (zero-padding was failing in the edge tiles, so using mirror padding)
# 2.Handling the overlap differently now using a weighted average method: we give 
# weight 1 topredictions from a square sized halo_size by halo_size in the centre 
# of each tile and the rest of the tile get weight values faded from  1 to 0 going
# from the edge of the square in the centre to the edge of the tile. Specifically, 
# (a) Accumulation Process:Each tile adds its weighted prediction to the accumulator:
# predictions += tile_pred * tile_weights. Then, each tile adds its weights to the 
# weight accumulator: weight_accumulator += tile_weights 
# (b) Final Normalization: At the end, we divide by total weights: 
# predictions = predictions / weight_accumulator (Converts probabilities > 0.5 to 1
# probabilities ≤ 0.5 to 0)
# This gives us the weighted average of all predictions that contributed to each pixel
class DatasetInference(SatelliteDataset):
    def __init__(self, main_directory, model, dataset, tile_size=512, overlap=0.7, mean_per_channel=None, std_per_channel=None, halo_size=64, padding_mode='reflect'):
        self.main_directory = main_directory
        self.tile_size = tile_size
        self.overlap = overlap
        self.model = model.to(DEVICE)  # The trained model
        self.mean_per_channel = mean_per_channel
        self.std_per_channel = std_per_channel
        self.halo_size = halo_size
        self.padding_mode = padding_mode  # 'reflect', 'edge', 'constant', or 'symmetric'

        # Create weight map for weighted averaging
        self.weight_map = create_weight_map(tile_size, halo_size)

        # Get the four required file paths
        self.image_path1, self.image_path2, self.substrate_path, self.bathymetry_path = self.get_file_paths(main_directory)

        # Load the image and process it
        self.image, self.metadata = self.load_image()

        # Calculate indices using methods from the dataset class
        self.calculate_and_concat_indices(dataset)


    def get_file_paths(self, main_directory):
        """Retrieve the four file paths from the directory."""
        file_patterns = ["*_B2B3B4B8.tif", "*_B5B6B7B8A_B11B12.tif", "*_Substrate.tif", "*_Bathymetry.tif"]
        file_paths = []

        for pattern in file_patterns:
            matching_files = glob.glob(os.path.join(main_directory, pattern))
            if len(matching_files) != 1:
                raise ValueError(f"Expected one file for pattern {pattern}, found {len(matching_files)}.")
            file_paths.append(matching_files[0])

        return file_paths  # Returns four paths in order

    def load_image(self):
        """Load all image bands from the four source files (10m bands)."""

        # Load 10m bands (B2, B3, B4, B8)
        with rasterio.open(self.image_path1) as src1:
            image1 = src1.read([1, 2, 3, 4])  # Read bands at 10m resolution
            image1 = np.transpose(image1, (1, 2, 0)).astype(np.float32)  # Shape (H, W, 4)
            metadata = src1.meta  # Save metadata for later use

        # Load 20m bands (B5, B6, B7, B8A, B11, B12)
        with rasterio.open(self.image_path2) as src2:
            #rasterio.open().read() function accepts the indexes parameter to specify which bands to read from a
            # multi-band raster. This parameter allows you to select specific bands by providing their indices. In Rasterio, band indices start at 1
            # If you want bands 5, 6, 7, and 8a, for example, you should write src2.read(indexes=[1], out_shape=(1, image1.shape[0], image1.shape[1],
            # resampling=Resampling.nearest)
            image2 = src2.read(indexes=[1], out_shape=(
                1,  # change this and indexes based on the bands included in the training. Only band B5 (red edge), which is band index 1, was used.
                image1.shape[0],  # Match height of 10m bands
                image1.shape[1]    # Match width of 10m bands
            ), resampling=Resampling.nearest)  # Upsample using nearest neighbor
            image2 = np.transpose(image2, (1, 2, 0)).astype(np.float32)  # Shape (H, W, 1)

        # Load Substrate data
        with rasterio.open(self.substrate_path) as src3:
            substrate = src3.read(1).astype(np.float32)  # Shape (H, W)
            substrate = substrate[:, :, np.newaxis]  # Expand to (H, W, 1)

        # Load Bathymetry data
        with rasterio.open(self.bathymetry_path) as src4:
            bathymetry = src4.read(1).astype(np.float32)  # Shape (H, W)
            bathymetry = bathymetry[:, :, np.newaxis]  # Expand to (H, W, 1)

        # Merge all bands together
        self.image = np.concatenate((image1, image2, substrate, bathymetry), axis=-1)  # Shape (H, W, 7)

        return self.image, metadata

    def calculate_and_concat_indices(self, dataset):
        """Calculate indices using methods from the dataset class."""
        # Include all the indices that were used in the training.
        ndvi = dataset.calculate_ndvi(self.image)
        ndwi = dataset.calculate_ndwi(self.image)
        gndvi = dataset.calculate_gndvi(self.image)
        chlorophyll_index = dataset.calculate_chlorophyll_index_green(self.image)
        ndvire = dataset.calculate_ndvi_re(self.image) #Normalized Difference of Red and Blue
        # ndreb = dataset.calculate_ndreb(self.image)
        # blue_red = dataset.calculate_blue_red(self.image)
        # tvi = dataset.calculate_tvi(self.image)
        # rdvi = dataset.calculate_rdvi(self.image)
        # ndreb = dataset.calculate_ndreb(self.image)
        # cig = dataset.calculate_cig(self.image)
        # blue_rededge = dataset.calculate_blue_rededge(self.image)
        # blue_nir = dataset.calculate_blue_nir(self.image)
        # red_minus_blue = dataset.calculate_red_minus_blue(self.image)
        # bndvi = dataset.calculate_bndvi(self.image)


        #ndrb = dataset.calculate_ndrb(self.image), #Normalized Difference of Red and Blue
        #mgvi = dataset.calculate_mgvi(self.image), #Modified Green Red Vegetation Index (MGVI)
        #mpri = dataset.calculate_mpri(self.image), #Modified Photochemical Reflectance Index (MPRI)
        #rgbvi = dataset.calculate_rgbvi(self.image), #Red Green Blue Vegetation Index (RGBVI)
        #gli = dataset.calculate_gli(self.image), #Green Leaf Index (GLI)
        #gi = dataset.calculate_gi(self.image), #Greenness Index (GI)
        #br = dataset.calculate_blue_red(self.image), #Blue/Red
        #exg = dataset.calculate_exg(self.image), #Excess of Green (ExG)
        #vari = dataset.calculate_vari(self.image), #Visible Atmospherically Resistant Index (VARI)
        #evi = dataset.calculate_evi(self.image), #Enhanced Vegetation Index (EVI)


        # Concatenate indices to the original image along the band dimension
        self.image = np.concatenate((self.image, ndvi[..., np.newaxis], ndwi[..., np.newaxis],
                                         gndvi[..., np.newaxis], chlorophyll_index[..., np.newaxis], ndvire[..., np.newaxis]), axis=-1) # ,
                                         # ndvire[..., np.newaxis], ndreb[..., np.newaxis], blue_red[..., np.newaxis], tvi[..., np.newaxis],
                                         # rdvi[..., np.newaxis], ndreb[..., np.newaxis], cig[..., np.newaxis],
                                         # blue_rededge[..., np.newaxis], blue_nir[..., np.newaxis],
                                         # red_minus_blue[..., np.newaxis], bndvi[..., np.newaxis]
        self.image = self.image[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  # selection based on what the model is trained on (attention: zero-based)

        # import matplotlib.pyplot as plt
        # # Plot the extracted channel
        # plt.figure(figsize=(8, 6))
        # plt.imshow(self.image[:,:,5])  # 'viridis' is just an example colormap
        # # plt.colorbar()  # Optional: to display a color scale
        # plt.title('bathy Image before')
        # plt.show()

        # # ***********Remove this:
        # self.image[:,:,4][self.image[:,:,4] > 10] = -2000
        # self.image[:,:,4][self.image[:,:,4] < -100] = -2000


        # # ***********Remove this:
        # self.image[:,:,4][self.image[:,:,4] != 1] = 0


        # Assuming `self.image` has been set as per the code you provided

        # Extract the 6th channel (0-based index 5)
        # channel_5 = self.image[:,:,5]

        # # Plot the extracted channel
        # plt.figure(figsize=(8, 6))
        # plt.imshow(channel_5)  # 'viridis' is just an example colormap
        # # plt.colorbar()  # Optional: to display a color scale
        # plt.title('bathy Image after')
        # plt.show()

    def generate_tiles(self, image):
        """Generator that yields one tile and its coordinates at a time."""
        h, w, c = image.shape
        tile_size = self.tile_size
        overlap = self.overlap
        step_size = int(tile_size * (1 - overlap))
    
        # Main tiling loop
        for i in range(0, h - tile_size + 1, step_size):
            for j in range(0, w - tile_size + 1, step_size):
                tile = image[i:i+tile_size, j:j+tile_size]
                if self.mean_per_channel is not None and self.std_per_channel is not None:
                    tile = normalize_tile_mean_std(tile, self.mean_per_channel, self.std_per_channel)
                yield tile, (i, j)
    
        # Edge padding - handle incomplete tiles at edges
        edge_positions = []
        
        # Right edge tiles
        if w % step_size != 0 or w - tile_size < 0:
            for i in range(0, h - tile_size + 1, step_size):
                j_start = max(0, w - tile_size)
                if j_start not in [j for _, j in edge_positions if i == i]:  # Avoid duplicates
                    edge_positions.append((i, j_start))
        
        # Bottom edge tiles
        if h % step_size != 0 or h - tile_size < 0:
            for j in range(0, w - tile_size + 1, step_size):
                i_start = max(0, h - tile_size)
                if i_start not in [i for i, _ in edge_positions if j == j]:  # Avoid duplicates
                    edge_positions.append((i_start, j))
        
        # Bottom-right corner tile
        if (h % step_size != 0 or h - tile_size < 0) and (w % step_size != 0 or w - tile_size < 0):
            i_start = max(0, h - tile_size)
            j_start = max(0, w - tile_size)
            if (i_start, j_start) not in edge_positions:
                edge_positions.append((i_start, j_start))
        
        # Generate edge tiles
        for i, j in edge_positions:
            tile = image[i:min(i+tile_size, h), j:min(j+tile_size, w)]
            tile = self.pad_tile(tile)
            if self.mean_per_channel is not None and self.std_per_channel is not None:
                tile = normalize_tile_mean_std(tile, self.mean_per_channel, self.std_per_channel)
            yield tile, (i, j)

    def generate_tiles_not_weighted(self, image):
        """Generator that yields one tile and its coordinates at a time."""
        h, w, c = image.shape
        tile_size = self.tile_size
        overlap = self.overlap
        step_size = int(tile_size * (1 - overlap))
    
        # Main tiling loop
        for i in range(0, h - tile_size + 1, step_size):
            for j in range(0, w - tile_size + 1, step_size):
                tile = image[i:i+tile_size, j:j+tile_size]
                if self.mean_per_channel is not None and self.std_per_channel is not None:
                    tile = normalize_tile_mean_std(tile, self.mean_per_channel, self.std_per_channel)
                yield tile, (i, j)
    
        # Edge padding
        for i in range(h - tile_size, h, step_size):
            for j in range(w - tile_size, w, step_size):
                tile = self.pad_tile_not_weighted(image[i:min(i+tile_size, h), j:min(j+tile_size, w)])
                if self.mean_per_channel is not None and self.std_per_channel is not None:
                    tile = normalize_tile_mean_std(tile, self.mean_per_channel, self.std_per_channel)
                yield tile, (i, j)

    def pad_tile_not_weighted(self, tile):
        """Apply padding to the tile if it's smaller than the tile_size."""
        pad_h = max(0, self.tile_size - tile.shape[0])
        pad_w = max(0, self.tile_size - tile.shape[1])
        # return np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        return np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    def count_tiles_not_weighted(self, image_shape, tile_size, overlap):
        """Count how many tiles will be generated, exactly matching generate_tiles()."""
        h, w, _ = image_shape
        step_size = int(tile_size * (1 - overlap))
        count = 0
    
        # Main tiling loop
        for i in range(0, h - tile_size + 1, step_size):
            for j in range(0, w - tile_size + 1, step_size):
                count += 1
    
        # Edge padding (only bottom-right area, just like in generate_tiles)
        for i in range(h - tile_size, h, step_size):
            for j in range(w - tile_size, w, step_size):
                count += 1
    
        return count
    def pad_tile(self, tile):
        """Apply intelligent padding to the tile if it's smaller than the tile_size."""
        pad_h = max(0, self.tile_size - tile.shape[0])
        pad_w = max(0, self.tile_size - tile.shape[1])
        
        if pad_h == 0 and pad_w == 0:
            return tile
            
        # Use the specified padding mode
        if self.padding_mode == 'reflect':
            # Handle cases where tile is too small for reflect padding
            if tile.shape[0] == 1 and pad_h > 0:
                mode = 'edge'  # Fall back to edge for very small tiles
            elif tile.shape[1] == 1 and pad_w > 0:
                mode = 'edge'
            else:
                mode = 'reflect'
        else:
            mode = self.padding_mode
            
        return np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)

    def count_tiles(self, image_shape, tile_size, overlap):
        """Count how many tiles will be generated, exactly matching generate_tiles()."""
        h, w, _ = image_shape
        step_size = int(tile_size * (1 - overlap))
        count = 0
    
        # Main tiling loop
        for i in range(0, h - tile_size + 1, step_size):
            for j in range(0, w - tile_size + 1, step_size):
                count += 1
    
        # Edge tiles count
        edge_positions = []
        
        # Right edge tiles
        if w % step_size != 0 or w - tile_size < 0:
            for i in range(0, h - tile_size + 1, step_size):
                j_start = max(0, w - tile_size)
                edge_positions.append((i, j_start))
        
        # Bottom edge tiles
        if h % step_size != 0 or h - tile_size < 0:
            for j in range(0, w - tile_size + 1, step_size):
                i_start = max(0, h - tile_size)
                edge_positions.append((i_start, j))
        
        # Bottom-right corner tile
        if (h % step_size != 0 or h - tile_size < 0) and (w % step_size != 0 or w - tile_size < 0):
            i_start = max(0, h - tile_size)
            j_start = max(0, w - tile_size)
            edge_positions.append((i_start, j_start))
        
        # Remove duplicates
        edge_positions = list(set(edge_positions))
        count += len(edge_positions)
    
        return count

    def _process_batch(self, tiles, coords, predictions):
        """Not weighted - Run inference on a batch of tiles and write results into full image."""
        batch_tensor = torch.cat(tiles, dim=0).to(DEVICE)  # shape: (B, C, H, W)
        outputs = self.model(batch_tensor)  # shape: (B, 1, H, W) or (B, H, W)
    
        # Handle binary output (thresholding)
        if outputs.shape[1] == 1:
            outputs = (outputs.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
        else:
            outputs = outputs.cpu().numpy().astype(np.uint8)
    
        # Write predictions into full image
        for pred, (i, j) in zip(outputs, coords):
            effective_tile_height = min(self.tile_size, predictions.shape[0] - i)
            effective_tile_width = min(self.tile_size, predictions.shape[1] - j)
    
            predictions[i:i + effective_tile_height, j:j + effective_tile_width] = np.maximum(
                predictions[i:i + effective_tile_height, j:j + effective_tile_width],
                pred[:effective_tile_height, :effective_tile_width]
            )

    def _process_batch_weighted_averaging(self, tiles, coords, predictions, weight_accumulator):
        """Process batch using weighted averaging with halo method."""
        batch_tensor = torch.cat(tiles, dim=0).to(DEVICE)
        outputs = self.model(batch_tensor)
    
        # Handle binary output (thresholding)
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(1).sigmoid().cpu().numpy()  # Keep as probabilities for averaging
        else:
            outputs = torch.softmax(outputs, dim=1).cpu().numpy()  # Multi-class probabilities
    
        # Process each prediction in the batch
        for pred, (i, j) in zip(outputs, coords):
            # Calculate the region bounds in the full image
            end_i = min(i + self.tile_size, predictions.shape[0])
            end_j = min(j + self.tile_size, predictions.shape[1])
            
            # Calculate effective tile size (in case of edge tiles)
            effective_h = end_i - i
            effective_w = end_j - j
            
            # Get the corresponding portion of the prediction and weight map
            tile_pred = pred[:effective_h, :effective_w]
            tile_weights = self.weight_map[:effective_h, :effective_w]
            
            # Weighted accumulation
            predictions[i:end_i, j:end_j] += tile_pred * tile_weights
            weight_accumulator[i:end_i, j:end_j] += tile_weights

    def run_model_on_tiles(self, batch_size=8):
        """
        Run the model on tiles with improved edge handling.
        
        Args:
            batch_size (int): Number of tiles to process in each batch
        """
        self.model.eval()
        

        predictions = np.zeros_like(self.image[:, :, 0], dtype=np.float32)
        weight_accumulator = np.zeros_like(self.image[:, :, 0], dtype=np.float32)

    
        tile_generator = self.generate_tiles(self.image)
        total_tiles = self.count_tiles(self.image.shape, self.tile_size, self.overlap)
    
        batch_tiles = []
        batch_coords = []
    
        with torch.no_grad():
            for tile, (i, j) in tqdm(tile_generator, total=total_tiles, desc="Predicting on Tiles"):
                # Preprocess and add tile to batch
                tile_tensor = torch.tensor(tile).permute(2, 0, 1).unsqueeze(0).float()
                batch_tiles.append(tile_tensor)
                batch_coords.append((i, j))
    
                # Run batch when full
                if len(batch_tiles) == batch_size:

                    self._process_batch_weighted_averaging(batch_tiles, batch_coords, predictions, weight_accumulator)

                    batch_tiles.clear()
                    batch_coords.clear()
    
            # Handle remaining tiles
            if batch_tiles:
                self._process_batch_weighted_averaging(batch_tiles, batch_coords, predictions, weight_accumulator)
    
        # Finalize predictions

        # Avoid division by zero
        weight_accumulator = np.where(weight_accumulator == 0, 1, weight_accumulator)
        predictions = predictions / weight_accumulator
            
        # Convert back to binary/class predictions
        if predictions.ndim == 2:  # Binary case
            predictions = (predictions > 0.5).astype(np.uint8)
        else:  # Multi-class case
            predictions = np.argmax(predictions, axis=-1).astype(np.uint8)
    
        # Optional: visualize the final result
        plt.figure(figsize=(10, 8))
        plt.imshow(predictions, cmap='gray')
        plt.title("Final Prediction")
        plt.axis('off')
        plt.show()
    
        return predictions

    def run_model_on_tiles_not_weighted(self, batch_size=8):
        """Run the model on tiles in batches with GPU acceleration and low RAM usage."""
        self.model.eval()
        predictions = np.zeros_like(self.image[:, :, 0], dtype=np.uint8)
    
        tile_generator = self.generate_tiles_not_weighted(self.image)
        total_tiles = self.count_tiles_not_weighted(self.image.shape, self.tile_size, self.overlap)
    
        batch_tiles = []
        batch_coords = []
    
        with torch.no_grad():
            for tile, (i, j) in tqdm(tile_generator, total=total_tiles, desc="Predicting on Tiles"):
                # Preprocess and add tile to batch
                tile_tensor = torch.tensor(tile).permute(2, 0, 1).unsqueeze(0).float()
                batch_tiles.append(tile_tensor)
                batch_coords.append((i, j))
    
                # Run batch when full
                if len(batch_tiles) == batch_size:
                    self._process_batch(batch_tiles, batch_coords, predictions)
                    batch_tiles.clear()
                    batch_coords.clear()
    
            # Handle remaining tiles
            if batch_tiles:
                self._process_batch(batch_tiles, batch_coords, predictions)
    
        # Optional: visualize the final result
        plt.imshow(predictions, cmap='gray')
        plt.title("Final Predictions")
        plt.axis('off')
        plt.show()
    
        return predictions


    def __getitem__(self, idx):
        """Return the tile and its corresponding coordinates."""
        return self.tiles[idx], self.tile_coords[idx]

    def __len__(self):
        """Return the number of tiles."""
        return len(self.tiles)

    def save_output(self, predictions, output_path):
        """Save the reconstructed output as a GeoTIFF."""
        # Update predictions to uint8 if needed before saving
        predictions = predictions.astype(np.uint8)

        # Update the metadata, ensuring it matches the predictions
        updated_metadata = self.metadata.copy()
        updated_metadata.update({
            'driver': 'GTiff',
            'dtype': 'uint8',      # Make sure this matches predictions' dtype
            'count': 1,          # Single-band output
            'compress': 'lzw'      # Optional compression to reduce file size
        })

        # Save the file with rasterio, ensuring that spatial metadata is preserved
        with rasterio.open(output_path, 'w', **updated_metadata) as dst:
            dst.write(predictions, 1)  # Write data to the first band

# Set the main directory
# main_directory = r"C:\Alena\results\20220806T191919_20220806T192707_T09UXQ"
# output_filename = "output_cli.tif"
# mean_per_channel = [ 9.73005488e+02  7.08909146e+02  4.49016997e+02  7.64114558e+02
#   5.04806707e+02  1.55057075e+00 -3.07090868e+01 -4.03823853e-02
#   2.47833866e-01 -2.47833866e-01 -3.45288439e-02  1.30275308e-02]
# std_per_channel = [3.14151787e+02 2.99688583e+02 3.10539248e+02 9.25681716e+02
#  3.78586231e+02 1.27053181e+00 8.72741729e+01 3.88193209e-01
#  4.20784913e-01 4.20784913e-01 1.12317310e+00 1.60626435e-01]
# data = SatelliteDataset(image_paths=X_val_paths, mask_paths=y_val_paths, augmentation=None, classes=["kelp"])  # solely to access to the methods of the class for index calculation
# dataset = DatasetInference(main_directory=main_directory, model=model, dataset=data, mean_per_channel=mean_per_channel, std_per_channel=std_per_channel)
# 
# # Run model and save predictions
# predictions = dataset.run_model_on_tiles()
# 
# output_path = os.path.join(main_directory, output_filename)
# dataset.save_output(predictions, output_path)


def classify(input_dir, output_filename, mean_per_channel, std_per_channel):
    """
    Perform semantic segmentation inference on a Sentinel-2 scene using a preloaded model.
    
    Parameters:
    - input_dir: directory containing the expected input TIFF files
    - output_filename: output TIFF filename to save prediction
    - mean_per_channel, std_per_channel: normalization stats used during training
    """
    # Use existing model, dataset class, and logic
    data = SatelliteDataset(image_paths=None, mask_paths=None, augmentation=None, classes=["kelp"])
    dataset = DatasetInference(
        main_directory=input_dir,
        model=model,
        dataset=data,
        mean_per_channel=mean_per_channel,
        std_per_channel=std_per_channel,
        tile_size=512,
        overlap=0.5,  
        halo_size=64,
        padding_mode='reflect'  # Use reflect padding instead of zero padding
    )

    predictions = dataset.run_model_on_tiles(batch_size=8)  # run_model_on_tiles for the weighted option and run_model_on_tiles_not_weighted for the not-weighted one
    output_path = os.path.join(input_dir, output_filename)
    dataset.save_output(predictions, output_path)