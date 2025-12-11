# configs/maskformer_config.py

# Map from the Cityscapes RAW IDs (found in the PNG file) to your model's
# sequential Target Class Index (0, 1, 2, 3, etc.).

# We will use 0 for all non-movable background items (road, building, trees, etc.)
# and sequential indices for our main traffic objects.

# configs/maskformer_config.py

ID_TO_LABEL = {
    # --- RAW IDs MAPPED TO BACKGROUND (0) ---
    # Void/Unlabeled (IDs 0-6, 9-10, 14-16, 29-30, -1)
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 9: 0, 10: 0, 14: 0, 
    15: 0, 16: 0, 29: 0, 30: 0, -1: 0, 
    
    # Non-Traffic Background (IDs 7, 8, 11-13, 17-23)
    7: 0, 8: 0, 11: 0, 12: 0, 13: 0, 17: 0, 18: 0, 19: 0, 20: 0, 
    21: 0, 22: 0, 23: 0,
    
    # --- RAW IDs MAPPED TO TARGET TRAFFIC USERS (1-8) ---
    24: 1,  # person
    25: 2,  # rider
    26: 3,  # car
    27: 4,  # truck
    28: 5,  # bus
    31: 6,  # train
    32: 7,  # motorcycle
    33: 8,  # bicycle
}

# The number of unique output classes is the highest index (8) + 1 (for index 0).
NUM_CLASSES = 9
MODEL_CHECKPOINT = "facebook/maskformer-swin-base-coco" # Keep the powerful pretrained backbone
IGNORE_INDEX = 255          
BATCH_SIZE = 2             # Start very low for a large model (RTX 5080)
LEARNING_RATE = 5e-5        
NUM_EPOCHS = 10