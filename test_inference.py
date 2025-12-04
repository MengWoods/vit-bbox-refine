# test_inference.py

import torch
from PIL import Image, ImageDraw
import requests
import time
import os
from datetime import datetime 
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from torchvision.ops import masks_to_boxes

# --- Configuration ---
MODEL_NAME = "facebook/maskformer-swin-base-coco"
SAMPLE_IMAGE_URL = "https://images.unsplash.com/photo-1623311785782-b7deb5d25de4?q=80&w=780&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" 

# Generate a detailed timestamp (e.g., 20251204_153457)
DATETIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
# ⭐️ NEW OUTPUT FILE NAME includes the timestamp
OUTPUT_FILE = f"output_detections_{DATETIME_STR}.jpg" 

# Generate a date-based directory name (e.g., 'inference_results/2025-12-04')
DATE_STR = datetime.now().strftime("%Y-%m-%d")
OUTPUT_ROOT = "inference_results"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, DATE_STR) 

# --- Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# [mask_to_bbox_detection function remains the same]
def mask_to_bbox_detection(segments_info, segmentation_map, image_size):
    """
    Converts predicted segmentation masks into standard bounding box and class labels.
    """
    # ... (Function content remains the same) ...
    detections = []
    for segment in segments_info:
        segment_id = segment['id']
        class_id = segment['label_id']
        score = segment.get('score', 1.0) 
        instance_mask_tensor = (segmentation_map == segment_id).float()
        if instance_mask_tensor.sum() == 0:
            continue
        try:
            boxes = masks_to_boxes(instance_mask_tensor.unsqueeze(0))
            bbox = boxes.squeeze(0).tolist() 
        except ValueError:
            continue 

        detections.append({
            "box": [round(c, 2) for c in bbox],
            "class_id": class_id,
            "score": round(score, 4),
            "mask_area": instance_mask_tensor.sum().item()
        })
    return detections

def draw_detections_on_image(image: Image.Image, detections: list, id_to_label: dict, folder_path: str, filename: str):
    """Draws bounding boxes and labels onto the original image and saves it in the specified folder."""
    
    # Check and create folder structure recursively if it doesn't exist
    os.makedirs(folder_path, exist_ok=True) 
    
    draw = ImageDraw.Draw(image)
    
    for det in detections:
        box = det['box']
        class_id = det['class_id']
        score = det['score']
        class_name = id_to_label.get(class_id, "UNKNOWN")
        
        color = "red" if class_name in ["car", "truck", "bus"] else "blue"
        label = f"{class_name}: {score:.2f}"
        
        draw.rectangle(box, outline=color, width=3)
        
        text_x = box[0] 
        text_y = box[1] - 15 
        
        text_bbox = draw.textbbox((text_x, text_y), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((text_x, text_y), label, fill="white")

    save_path = os.path.join(folder_path, filename)
    image.save(save_path)
    print(f"\n🖼️ Visualized output saved to: **{save_path}**")


def run_inference_test():
    """Loads the model, runs inference on a sample image, and extracts bounding boxes."""
    
    # --- Step 1 to 5 (Loading, Preprocessing, Inference, BBox Conversion) ---
    print("LOG: Loading processor and model...")
    processor = MaskFormerImageProcessor.from_pretrained(MODEL_NAME)
    model = MaskFormerForInstanceSegmentation.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval() 

    print(f"LOG: Downloading sample image from {SAMPLE_IMAGE_URL}")
    image = Image.open(requests.get(SAMPLE_IMAGE_URL, stream=True).raw).convert("RGB")
    original_size = image.size 

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')): 
            outputs = model(**inputs)
    inference_time = (time.time() - start_time) * 1000

    target_size = original_size[::-1] 
    segmentation_output = processor.post_process_panoptic_segmentation(
        outputs, 
        target_sizes=[target_size]
    )[0] 
    
    detections = mask_to_bbox_detection(segmentation_output['segments_info'], segmentation_output['segmentation'], original_size)
    final_detections = [d for d in detections if d['score'] > 0.7]

    # --- Step 6: Display Results (Modified for brevity) ---
    id_to_label = model.config.id2label

    # --- Step 7: Visualize and Save ---
    draw_detections_on_image(image, final_detections, id_to_label, OUTPUT_DIR, OUTPUT_FILE)

    print("\n--- Test Complete ---")
    print(f"Total objects detected: {len(final_detections)}")

if __name__ == "__main__":
    run_inference_test()