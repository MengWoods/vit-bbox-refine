# train.py (K-Fold Cross-Validation Version)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.cuda.amp as amp # Using torch.cuda.amp for GradScaler and autocast
import os
import time
from sklearn.model_selection import KFold
import numpy as np
from typing import Dict, Any, Tuple

from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from configs.maskformer_config import (
    MODEL_CHECKPOINT, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
)
# Note: Assuming src/data/dataset.py (and collate_fn) is now the fixed version
from src.data.dataset import CityscapesLikeDataset, collate_fn

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "models/checkpoints"
DATA_ROOT = "src/data/"
TRAIN_IMG_FOLDER = "train_images"
TRAIN_MASK_FOLDER = "train_masks"
K_FOLDS = 5 # Define the number of folds

DEBUG_MODE = False      # Set to False for full training
DEBUG_SAMPLES = 100    # Use only 100 samples for the training subset

# --- Helper Functions ---

def init_model(processor: MaskFormerImageProcessor) -> Tuple[MaskFormerForInstanceSegmentation, optim.AdamW, amp.GradScaler]:
    """Initializes and returns a fresh, untrained model instance, optimizer, and gradient scaler."""
    # Load model with custom number of classes (for the fine-tuning head)
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        MODEL_CHECKPOINT,
        ignore_mismatched_sizes=True, # Allow mismatch since we change the class head
        num_labels=NUM_CLASSES
    ).to(DEVICE)

    # Set model to training mode, ensuring layers like dropout and BN behave correctly.
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Initialize the gradient scaler for Automatic Mixed Precision (AMP)
    scaler = amp.GradScaler()
    return model, optimizer, scaler

def evaluate_model(model: MaskFormerForInstanceSegmentation, data_loader: DataLoader, device: torch.device) -> float:
    """Calculates the average loss on the validation set."""
    model.eval() # Set model to evaluation mode
    total_val_loss = 0

    with torch.no_grad(): # Disable gradient calculations
        for batch in data_loader:
            # Move data to GPU (batch['mask_labels'] and batch['class_labels'] remain lists here)

            batch = move_to_device(batch, device)

            with amp.autocast(): # Use mixed precision
                outputs = model(**batch)
                loss = outputs.loss

            total_val_loss += loss.item()

    model.train() # Switch model back to training mode
    return total_val_loss / len(data_loader)

def move_to_device(data: Any, device: torch.device):
    """Recursively moves tensors within a dictionary or list to the specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        # Recursively move each tensor inside the list
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        # Recursively move tensors inside the dictionary
        return {k: move_to_device(v, device) for k, v in data.items()}
    else:
        return data

# --- Main Training Function ---

def train_model_kfold():
    """Trains a segmentation model using K-Fold cross-validation."""
    print(f"--- Starting {K_FOLDS}-Fold Cross-Validation on {DEVICE} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Setup Processor and Full Dataset
    processor = MaskFormerImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    full_dataset = CityscapesLikeDataset(DATA_ROOT, TRAIN_IMG_FOLDER, TRAIN_MASK_FOLDER, processor)

    print(f"Loaded total {len(full_dataset)} training samples.")

    # K-Fold Setup
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_losses = []

    # --- K-FOLD LOOP ---
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n#################################################################")
        print(f"############ FOLD {fold+1}/{K_FOLDS} - START ############################")
        print(f"#################################################################")

        # Convert train_ids to a list for slicing (if it's a numpy array)
        train_ids = list(train_ids)
        val_ids = list(val_ids)

        if DEBUG_MODE:
            # Slice the list to keep only the first N samples
            train_ids = train_ids[:DEBUG_SAMPLES]
            # Optionally reduce validation set too, for fast validation
            val_ids = val_ids[:min(len(val_ids), DEBUG_SAMPLES // 4)]

            print(f"ATTENTION: DEBUG MODE ACTIVE. Reducing train set to {len(train_ids)} samples.")

        # 2. Create Data Samplers and Loaders for the current fold
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(
            full_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            collate_fn=collate_fn, # ⭐️ Use the custom collate_fn
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            full_dataset,
            batch_size=BATCH_SIZE,
            sampler=val_sampler,
            collate_fn=collate_fn, # ⭐️ Use the custom collate_fn
            num_workers=4,
            pin_memory=True
        )
        print(f"Fold {fold+1}: Train size: {len(train_ids)}, Validation size: {len(val_ids)}")

        # 3. Initialize Model, Optimizer, and Scaler for the new fold
        model, optimizer, scaler = init_model(processor)
        best_val_loss = float('inf')

        # 4. EPOCH LOOP (Inside Fold)
        fold_start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            train_loss = 0
            model.train()

            # --- Training Step ---
            for step, batch in enumerate(train_loader):
                batch_start_time = time.time()

                # Move fixed-size Tensors to GPU. mask_labels and class_labels remain lists.
                # batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch = move_to_device(batch, DEVICE)

                optimizer.zero_grad()

                # ⭐️ CLEAN FIX: REMOVED ALL MANUAL STACKING/UNNESTING.
                # The data is now correctly passed as a list of tensors for variable-size inputs.
                with amp.autocast():

                    # 2. DEBUG: Print the final Tensor shapes before calling the model
                    # NOTE: This print logic now verifies the CORRECT LIST structure
                    if step == 0 and epoch == 0:
                        print("\n--- DEBUG: BATCH STRUCTURE BEFORE MODEL CALL ---")
                        print(f"Pixel Values Shape: {batch['pixel_values'].shape}")
                        print(f"Pixel Mask Shape:   {batch['pixel_mask'].shape}")

                        # CRITICAL CHECK: These should be LISTS of Tensors
                        print(f"Mask Labels Type:   {type(batch['mask_labels'])}")
                        print(f"Class Labels Type:  {type(batch['class_labels'])}")
                        print(f"Example Mask Shape: {batch['mask_labels'][0].shape}")
                        print("----------------------------------------------\n")

                    outputs = model(**batch)
                    loss = outputs.loss

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                # Detailed step-by-step logging
                if (step + 1) % 50 == 0:
                    print(f"  [Train] Epoch {epoch+1}/{NUM_EPOCHS}, Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Batch Time: {(time.time() - batch_start_time):.3f}s")

            avg_train_loss = train_loss / len(train_loader)

            # --- Validation Step ---
            val_loss = evaluate_model(model, val_loader, DEVICE)
            # Epoch Summary Log
            epoch_time = time.time() - epoch_start_time
            print(f"\n--- FOLD {fold+1}, Epoch {epoch+1} Summary ---")
            print(f"  Time Used: {epoch_time:.2f}s")
            print(f"  Avg Train Loss: {avg_train_loss:.4f}")
            print(f"  Avg Val Loss: {val_loss:.4f}")

            # Save Checkpoint based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = os.path.join(OUTPUT_DIR, f"fold_{fold+1}_epoch_{epoch+1}_best")
                model.save_pretrained(model_save_path)
                print(f"  -> Saved new best model checkpoint for Fold {fold+1} (Loss: {best_val_loss:.4f})")

        fold_time = time.time() - fold_start_time
        fold_losses.append(best_val_loss)
        print(f"\n############ FOLD {fold+1} COMPLETE #########################")
        print(f"Total Fold Time: {fold_time:.2f}s, Best Validation Loss: {best_val_loss:.4f}")

    # 5. Final Results Summary
    print("\n\n=============== K-FOLD CROSS-VALIDATION RESULTS ===============")
    for i, loss in enumerate(fold_losses):
        print(f"Fold {i+1} Best Validation Loss: {loss:.4f}")

    avg_final_loss = np.mean(fold_losses)
    std_final_loss = np.std(fold_losses)
    print(f"---------------------------------------------------------------")
    print(f"Average Best Validation Loss across {K_FOLDS} Folds: **{avg_final_loss:.4f}** (Std Dev: {std_final_loss:.4f})")
    print("===============================================================")


if __name__ == "__main__":
    train_model_kfold()
