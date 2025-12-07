import numpy as np
from PIL import Image
import os

# --- IMPORTANT: Change this path to one of your actual gtFine_labelIds.png files ---
mask_path = "./train_masks/aachen_000000_000019_gtFine_labelIds.png" 
# ----------------------------------------------------------------------------------

def check_mask_ids(path):
    if not os.path.exists(path):
        print(f"ERROR: Mask file not found at {path}")
        return

    # Load the single-channel PNG mask using PIL
    # We load it in grayscale mode ('L') to ensure it's treated as a single channel integer map
    mask_img = Image.open(path).convert('L')
    
    # Convert the PIL image to a NumPy array
    mask_array = np.array(mask_img, dtype=np.uint8) # Use uint8 as Cityscapes IDs are small
    
    # Find all unique pixel values (these are your Label IDs)
    unique_ids = np.unique(mask_array)
    
    print("--- Mask File Analysis ---")
    print(f"File: {os.path.basename(path)}")
    print(f"Shape: {mask_array.shape}")
    print(f"Data Type: {mask_array.dtype}")
    print("--------------------------")
    print(f"✅ Found {len(unique_ids)} unique Label IDs (Pixel Values):")
    print(unique_ids)

# Run the check
check_mask_ids(mask_path)