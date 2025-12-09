# src/data/dataset.py (Full Fixed Version)

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from transformers import MaskFormerImageProcessor
# Assuming you have the config file correctly set up
from configs.maskformer_config import ID_TO_LABEL, IGNORE_INDEX, MODEL_CHECKPOINT
from typing import Dict, Any, List

class CityscapesLikeDataset(Dataset):
    """
    Handles Cityscapes-style data where images and masks are loaded by file name.
    """
    def __init__(self, root_dir: str, image_folder: str, mask_folder: str, processor: MaskFormerImageProcessor):
        self.image_dir = os.path.join(root_dir, image_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        self.processor = processor

        # List files. We'll use the image file list as the main index.
        # Ensure your file naming convention is consistent
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        # Construct the corresponding mask filename.
        mask_filename = img_filename.replace("leftImg8bit", "gtFine_labelIds")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        try:
            image = Image.open(img_path).convert("RGB")
            # Load the single-channel, label-ID segmentation mask
            segmentation_map = Image.open(mask_path)
        except Exception as e:
            print(f"Error loading files for {img_filename}: {e}. Skipping and returning None.")
            return None # Return None to be filtered by collate_fn

        # The processor handles resizing, normalization, and converting raw masks
        # into the required mask_labels and class_labels for MaskFormer loss.
        inputs = self.processor(
            image,
            segmentation_maps=segmentation_map,
            return_tensors="pt",
            instance_id_to_semantic_id=ID_TO_LABEL,
            ignore_index=IGNORE_INDEX
        )

        # ⭐️ FIX: Extract the single tensor from the batch dimension (dim 0)
        # The processor returns tensors wrapped in an unnecessary batch dimension of size 1
        # for single-sample input when return_tensors="pt" is used.
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                # For pixel_values, pixel_mask (fixed size), squeeze dim 0
                result[k] = v.squeeze(0)
            elif isinstance(v, List) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                # For mask_labels/class_labels (variable size, padded by processor),
                # extract the single element from the outer list [tensor] -> tensor
                result[k] = v[0]
            else:
                result[k] = v # Keep other non-tensor/list items as is

        return result


# --- Collation Function (Crucial for Batching) ---
def collate_fn(batch: List[Dict[str, Any]]):
    """
    Collation function designed for MaskFormer.
    It stacks fixed-size inputs (image and pixel mask) and returns variable-size
    inputs (mask_labels and class_labels) as a list of tensors.
    """
    # 1. Filter out any skipped samples (where __getitem__ returned None)
    batch = [item for item in batch if item is not None]

    # 2. Stack fixed-size tensors (pixel_values: [B, C, H, W], pixel_mask: [B, H, W])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])

    # 3. Return variable-size tensors as lists (MaskFormer handles the batching/padding internally)
    # mask_labels: [ [N1, H, W], [N2, H, W], ... ]
    # class_labels: [ [N1], [N2], ... ]
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels, # LIST of Tensors
        "class_labels": class_labels, # LIST of Tensors
    }
