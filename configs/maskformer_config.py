# configs/maskformer_config.py

# --- Cityscapes Class Mapping (Common Subset) ---
# Cityscapes has 30 classes, but we only target the *instance* classes
# that are relevant for traffic detection (e.g., people and vehicles).
ID_TO_LABEL = {
    0: "unlabeled",
    1: "car",
    2: "truck",
    3: "bus",
    4: "on_rails",   # train/tram
    5: "motorcycle",
    6: "bicycle",
    7: "person",
    8: "rider",
}
# NOTE: This is a simplified list. Cityscapes uses many more classes (road, sidewalk, etc.).
# For instance detection, we focus on the movable objects.

NUM_CLASSES = len(ID_TO_LABEL) 
MODEL_CHECKPOINT = "facebook/maskformer-swin-base-coco" # Keep the powerful pretrained backbone
IGNORE_INDEX = 255          
BATCH_SIZE = 2             # Start very low for a large model (RTX 5080)
LEARNING_RATE = 5e-5        
NUM_EPOCHS = 10