# The main script to run the fine-tuning process.
# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp # Import for Automatic Mixed Precision
import os
import time

from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from configs.maskformer_config import (
    MODEL_CHECKPOINT, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
)
from src.data.dataset import CityscapesLikeDataset, collate_fn

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "models/checkpoints"
# Assume your data is organized as data/train_images and data/train_masks
DATA_ROOT = "src/data/"
TRAIN_IMG_FOLDER = "train_images"
TRAIN_MASK_FOLDER = "train_masks"


def train_model():
    print(f"--- Starting Fine-Tuning on {DEVICE} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Setup Processor, Model, and Optimizer
    processor = MaskFormerImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    
    # Load model with custom number of classes (for the fine-tuning head)
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        MODEL_CHECKPOINT, 
        ignore_mismatched_sizes=True, # Allow mismatch since we change the class head
        num_labels=NUM_CLASSES 
    ).to(DEVICE)
    model.train() # Set to training mode
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Scaler for Automatic Mixed Precision (AMP)
    scaler = amp.GradScaler() 
    
    # 2. Setup DataLoaders
    train_dataset = CityscapesLikeDataset(DATA_ROOT, TRAIN_IMG_FOLDER, TRAIN_MASK_FOLDER, processor)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4, # Use multiple workers for faster data loading
        pin_memory=True # Pin memory for faster CPU-to-GPU transfer
    )
    print(f"Loaded {len(train_dataset)} training samples.")
    
    # 3. Training Loop
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            # Move data to GPU
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # --- Automatic Mixed Precision (AMP) Block ---
            with amp.autocast():
                # Forward pass: model expects pixel_values, pixel_mask, mask_labels, and class_labels
                outputs = model(**batch)
                loss = outputs.loss

            # Backward pass and optimization using scaler for AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Step {step}/{len(train_dataloader)}: Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        epoch_time = time.time() - start_time
        print(f"\n--- Epoch {epoch+1} Complete ---")
        print(f"Time: {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        # 4. Save Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(os.path.join(OUTPUT_DIR, f"maskformer_epoch_{epoch+1}"))
            print(f"Saved new best model checkpoint to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()