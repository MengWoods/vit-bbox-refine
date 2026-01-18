# video_inference_visualization_toggle.py (With Resizable Live Preview)

import torch
from PIL import Image, ImageDraw, ImageColor
import cv2
import numpy as np
import time
import os
from datetime import datetime
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from torchvision.ops import masks_to_boxes

# --- Configuration ---
MODEL_NAME = "facebook/maskformer-swin-base-coco"

# ⭐️ NEW CONFIGURATION: LIVE PREVIEW ⭐️
SHOW_LIVE_PREVIEW = False # <-- Set to True to see the video live, False to process silently

# ⭐️ NEW CONFIGURATION: PREVIEW SCALE ⭐️
# Set the factor for live window scaling (e.g., 0.5 is half size, 1.0 is original size)
LIVE_PREVIEW_SCALE_FACTOR = 0.95

# ⭐️ VISUALIZATION MODE ⭐️
VISUALIZATION_MODE = 'mask'  # 'mask' or 'bbox'

# ⭐️ CLASS FILTERING ⭐️
CLASS_IDS = [0, 1, 2, 3, 5, 7, 9, 11]
# Ground segmentation relevant classes
CLASS_IDS = [
    87,   # floor-wood
    90,   # gravel
    96,   # platform
    97,   # playingfield
    98,   # railroad
    99,   # river
    100,  # road
    102,  # sand
    103,  # sea
    105,  # snow
    113,  # water-other
    122,  # floor-other-merged
    123,  # pavement-merged
    124,  # mountain-merged
    125,  # grass-merged
    126,  # dirt-merged
    130,  # rock-merged
    132,  # rug-merged
]

MIN_SCORE_THRESHOLD = 0.8

# --- VIDEO INPUT/OUTPUT CONFIG ---
INPUT_VIDEO_PATH = "/home/mh/Downloads/Dataset/trafic-recordings/20250810_102526-sello-martin.mp4"
DATETIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO_FILE = f"processed_video_{VISUALIZATION_MODE}_filtered_{DATETIME_STR}.mp4"

DATE_STR = datetime.now().strftime("%Y-%m-%d")
OUTPUT_ROOT = "results/video_inference_results"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, DATE_STR)

# --- Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_FILE)

print(f"Using device: {DEVICE}")

# --- Color Mapping for Masks ---
COLOR_MAP = {
    'person': 'red', 'bicycle': 'orange', 'car': 'blue',
    'motorcycle': 'yellow', 'bus': 'lime', 'truck': 'cyan',
    'traffic light': 'green', 'stop sign': 'magenta'
}
# Below for ground segmentation usage
COLOR_MAP = {
    "road": "yellow",
    "pavement-merged": "orange",
    "floor-wood": "peru",
    "floor-other-merged": "saddlebrown",
    "gravel": "dimgray",
    "railroad": "slategray",
    "platform": "lightgray",
    "playingfield": "lime",
    "grass-merged": "green",
    "dirt-merged": "sienna",
    "sand": "khaki",
    "snow": "white",
    "river": "dodgerblue",
    "sea": "navy",
    "water-other": "blue",
    "rock-merged": "darkslategray",
    "mountain-merged": "rosybrown",
    "rug-merged": "tan",
}

DEFAULT_MASK_COLOR = 'white'
MASK_ALPHA = 150 # Transparency level (0=Full Transparent, 255=Fully Opaque)

# --- Helper Functions (mask_to_bbox_detection and draw_mask_on_image remain unchanged) ---

def mask_to_bbox_detection(segments_info, segmentation_map, image_size):
    # ... (function body remains the same) ...
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
            "mask": instance_mask_tensor,
            "mask_area": instance_mask_tensor.sum().item()
        })
    return detections

def draw_mask_on_image(image: Image.Image, detections: list, id_to_label: dict) -> Image.Image:
    # ... (function body remains the same, ensures transparency) ...
    processed_image = image.convert("RGBA")

    for det in detections:
        class_id = det['class_id']
        class_name = id_to_label.get(class_id, "UNKNOWN")
        mask_tensor = det['mask']

        color_name = COLOR_MAP.get(class_name.lower(), DEFAULT_MASK_COLOR)
        rgb_tuple = ImageColor.getrgb(color_name)

        color_image = Image.new('RGBA', image.size, color=(*rgb_tuple, 255))

        mask_array = mask_tensor.cpu().numpy().astype(np.uint8)

        alpha_mask_array = mask_array * MASK_ALPHA
        alpha_mask_image = Image.fromarray(alpha_mask_array, mode='L')

        color_image.putalpha(alpha_mask_image)

        processed_image = Image.alpha_composite(processed_image, color_image)

    draw = ImageDraw.Draw(processed_image)
    for det in detections:
        box = det['box']
        class_id = det['class_id']
        score = det['score']
        class_name = id_to_label.get(class_id, "UNKNOWN")

        text_color = "black" if color_name in ['lime', 'yellow', 'cyan'] else "white"
        label = f"{class_name}: {score:.2f}"

        text_x = box[0]
        text_y = box[1] - 15

        try:
            text_width, text_height = draw.textsize(label)
            text_bbox = [text_x, text_y, text_x + text_width + 5, text_y + text_height]
        except AttributeError:
             text_bbox = [text_x, text_y, text_x + 100, text_y + 15]

        # draw.rectangle(text_bbox, fill=(*rgb_tuple, 255))
        draw.text((text_x, text_y), label, fill=text_color)

    return processed_image.convert("RGB")

def draw_detections_on_image(image: Image.Image, detections: list, id_to_label: dict, mode: str) -> Image.Image:
    """Draws detections based on the selected mode ('bbox' or 'mask')."""
    if mode == 'mask':
        return draw_mask_on_image(image, detections, id_to_label)
    elif mode == 'bbox':
        # Bounding box drawing logic
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
            try:
                text_width, text_height = draw.textsize(label)
                text_bbox = [text_x, text_y, text_x + text_width + 5, text_y + text_height]
            except AttributeError:
                text_bbox = [text_x, text_y, text_x + 100, text_y + 15]
            draw.rectangle(text_bbox, fill=color)
            draw.text((text_x, text_y), label, fill="white")
        return image
    return image


def process_video_inference():
    # 1. Load Model and Processor
    print("LOG: Loading processor and MaskFormer model...")
    processor = MaskFormerImageProcessor.from_pretrained(MODEL_NAME)
    model = MaskFormerForInstanceSegmentation.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    id_to_label = model.config.id2label

    # 2. Setup Video Input and Output
    print(f"LOG: Opening input video: {INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {INPUT_VIDEO_PATH}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize VideoWriter
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    total_inference_time_ms = 0
    WINDOW_NAME = 'Live Inference Preview (Press Q to Quit)'

    print(f"LOG: Processing video ({total_frames} frames) in '{VISUALIZATION_MODE}' mode at {fps:.2f} FPS. Output path: {OUTPUT_VIDEO_PATH}")
    print(f"LOG: Filtering to Class IDs: {CLASS_IDS}")

    # ⭐️ MODIFICATION 1: Create Resizable Window ⭐️
    if SHOW_LIVE_PREVIEW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print(f"LOG: Live preview enabled with scale factor: {LIVE_PREVIEW_SCALE_FACTOR}. Press 'q' key to stop.")

    # 3. Process Video Frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        original_size = pil_image.size

        # --- Inference Block ---
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                outputs = model(**inputs)
        inference_time_ms = (time.time() - start_time) * 1000
        total_inference_time_ms += inference_time_ms

        # Post-processing and BBox/Mask Conversion
        target_size = original_size[::-1]
        segmentation_output = processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[target_size]
        )[0]

        detections = mask_to_bbox_detection(segmentation_output['segments_info'], segmentation_output['segmentation'], original_size)

        # APPLY CLASS AND SCORE FILTERING
        filtered_detections = []
        for det in detections:
            if det['score'] >= MIN_SCORE_THRESHOLD and det['class_id'] in CLASS_IDS:
                filtered_detections.append(det)

        # Draw results (either mask or bbox)
        processed_pil_image = draw_detections_on_image(pil_image.copy(), filtered_detections, id_to_label, VISUALIZATION_MODE)

        # Convert back to OpenCV BGR format
        processed_rgb_frame = np.array(processed_pil_image)
        processed_bgr_frame = cv2.cvtColor(processed_rgb_frame, cv2.COLOR_RGB2BGR)

        out.write(processed_bgr_frame)

        # ⭐️ MODIFICATION 2: Resize and Display ⭐️
        if SHOW_LIVE_PREVIEW:
            # Resize the frame using the scale factor
            scaled_frame = cv2.resize(
                processed_bgr_frame,
                None,
                fx=LIVE_PREVIEW_SCALE_FACTOR,
                fy=LIVE_PREVIEW_SCALE_FACTOR,
                interpolation=cv2.INTER_LINEAR
            )

            cv2.imshow(WINDOW_NAME, scaled_frame)

            # Check for 'q' key press to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nLOG: Live preview manually stopped by user.")
                break


        print(f"Processed frame {frame_count} / {total_frames}. Detections: {len(filtered_detections)}. Time: {inference_time_ms:.2f}ms")


    # 4. Release resources
    cap.release()
    out.release()
    # Ensure all OpenCV windows are closed
    cv2.destroyAllWindows()

    # Final Summary
    avg_inference_time_ms = total_inference_time_ms / frame_count if frame_count > 0 else 0
    print("\n--- Video Processing Complete ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Average inference time per frame: {avg_inference_time_ms:.2f}ms")
    print(f"Video saved to: **{OUTPUT_VIDEO_PATH}**")


if __name__ == "__main__":
    process_video_inference()
