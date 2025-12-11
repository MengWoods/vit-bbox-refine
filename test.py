import torch
from torch.utils.data import DataLoader
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
import os
from typing import Any, Dict, List, Tuple
# Assuming move_to_device is accessible or copied here
from train import move_to_device
from src.data.dataset import CityscapesLikeDataset, collate_fn

# --- Configuration (Adjust these) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "models/checkpoints/fold_5_epoch_10_best" # <-- Target checkpoint folder
BATCH_SIZE = 1
# --- Data Setup (Adjust these to your file structure) ---
DATA_ROOT = "src/data/"
TEST_IMG_FOLDER = "test_images"
TEST_MASK_FOLDER = "test_masks" # Ensure this folder holds your ground truth mask annotations

# Placeholder for your configurations (assuming from configs/maskformer_config.py)
from configs.maskformer_config import MODEL_CHECKPOINT
# ... (and other config variables needed for dataset/processor)



def run_testing(checkpoint_path: str, device: torch.device):
    # --- Model Loading ---
    print(f"Loading model from: {checkpoint_path}")
    model = MaskFormerForInstanceSegmentation.from_pretrained(checkpoint_path).to(device)
    processor = MaskFormerImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    model.eval() # CRITICAL: Disable dropout and batch normalization updates

    # --- Data Loading ---
    # Assuming CityscapesLikeDataset and collate_fn are accessible
    test_dataset = CityscapesLikeDataset(
        DATA_ROOT, TEST_IMG_FOLDER, TEST_MASK_FOLDER, processor
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # Do NOT shuffle for testing
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # --- Evaluation Loop ---
    # You would typically initialize a COCOEvaluator here
    # For simplicity, we just collect predictions and ground truths (GTs)
    all_predictions = []
    all_ground_truths = []

    print(f"Starting inference on {len(test_loader)} test batches...")

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            # 1. Move Data to Device (Crucial!)
            batch = move_to_device(batch, device)

            # 2. Forward Pass
            outputs = model(
                pixel_values=batch["pixel_values"],
                pixel_mask=batch["pixel_mask"]
            )

            # 3. Post-Processing
            # The processor converts raw logits into final instance masks/boxes

            # Get the original image sizes from the batch for correct resizing
            target_sizes = [(h, w) for h, w in zip(batch['original_heights'], batch['original_widths'])]

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            with torch.cuda.amp.autocast():
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    pixel_mask=batch["pixel_mask"]
                )

                # Post-processing is run inside autocast to use lower precision
                results = processor.post_process_instance_segmentation(
                    outputs=outputs,
                    target_sizes=target_sizes,
                )

            # 4. Collect Results
            # 'results' is a list of dictionaries (one per image in the batch).
            # Each dictionary contains keys like 'segmentation' and 'segments_info'.
            all_predictions.extend(results)

            # NOTE: The processor outputs 'mask_labels' and 'class_labels' as LISTS of Tensors
            all_ground_truths.extend(
                zip(batch['mask_labels'], batch['class_labels'])
            )

            if (step + 1) % 10 == 0:
                print(f"Processed {step + 1}/{len(test_loader)} batches...")

    # --- 5. Metric Calculation (Placeholder) ---
    print("Inference complete. Calculating final metrics...")

    # You would use a library like pycocotools or huggingface/evaluate here:
    # final_metrics = calculate_metrics(all_predictions, all_ground_truths, class_names)

    # For now, just print the size of the collected data
    print(f"Collected {len(all_predictions)} final predictions (one per image).")
    # print(f"Final AP@[0.5:0.95]: {final_metrics['AP']}")


# --- Execution ---
if __name__ == "__main__":
    run_testing(CHECKPOINT_PATH, DEVICE)
