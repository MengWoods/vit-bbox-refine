# src/data/dataset.py (Refined for Cityscapes-style data)

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from transformers import MaskFormerImageProcessor
from configs.maskformer_config import ID_TO_LABEL, IGNORE_INDEX, MODEL_CHECKPOINT
from typing import Dict, Any

class CityscapesLikeDataset(Dataset):
    """
    Handles Cityscapes-style data where images and masks are loaded by file name.
    """
    def __init__(self, root_dir: str, image_folder: str, mask_folder: str, processor: MaskFormerImageProcessor):
        self.image_dir = os.path.join(root_dir, image_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        self.processor = processor
        
        # List files. We'll use the image file list as the main index.
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        
        # ⚠️ CRITICAL STEP: Construct the corresponding mask filename.
        # Cityscapes image names look like: frankfurt_000000_000294_leftImg8bit.png
        # Mask names look like: frankfurt_000000_000294_gtFine_labelIds.png
        mask_filename = img_filename.replace("leftImg8bit", "gtFine_labelIds")
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
            # Load the single-channel, label-ID segmentation mask
            segmentation_map = Image.open(mask_path)
        except Exception as e:
            print(f"Error loading files for {img_filename}: {e}. Skipping.")
            # Simple handling: return a recursive call to load the next sample
            return self.__getitem__((idx + 1) % len(self)) 
        
        # The processor handles resizing, normalization, and converting raw masks 
        # into the required mask_labels and class_labels for MaskFormer loss.
        inputs = self.processor(
            image, 
            segmentation_maps=segmentation_map, 
            return_tensors="pt",
            instance_id_to_semantic_id=ID_TO_LABEL, 
            ignore_index=IGNORE_INDEX
        )
        
        # Squeeze the batch dimension (which is 1)
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        return inputs


# --- Collation Function (Remains the same, but include it here for completeness) ---
def collate_fn(batch):
    # Filter out any skipped samples (due to loading errors, if any)
    batch = [item for item in batch if item]
    
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }