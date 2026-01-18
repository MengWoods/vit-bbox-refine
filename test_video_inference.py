# video_inference_visualization_toggle.py
import torch
from PIL import Image, ImageDraw, ImageColor, ImageFont
import cv2
import numpy as np
import time
import os
from datetime import datetime
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from torchvision.ops import masks_to_boxes

# ==========================================================
# --- MASTER CONFIGURATION SECTION ---
# ==========================================================

# 1. MODEL & TARGET SELECTION
MODEL_NAME = "facebook/maskformer-swin-base-coco"
TARGET_TYPE = 'GROUND'  # OPTIONS: 'ROAD_USER' or 'GROUND'
VISUALIZATION_MODE = 'mask' # OPTIONS: 'mask' or 'bbox'

# 2. VISUAL APPEARANCE
CAPTION_FONT_SIZE = 20
MASK_ALPHA = 90            # Lowered to 120 for better see-through effect
MIN_SCORE_THRESHOLD = 0.8
LIVE_PREVIEW_SCALE_FACTOR = 0.75

# 3. TOGGLES
SHOW_LIVE_PREVIEW = False

# 4. INPUT / OUTPUT PATHS
INPUT_VIDEO_PATH = "/home/mh/Downloads/Dataset_preprocessed/traffic_recordings_fin/aalto-martinkyla.mp4"
OUTPUT_ROOT = "results/video_inference_results"

# --- DYNAMIC CLASS MAPPING ---
if TARGET_TYPE == 'ROAD_USER':
    CLASS_IDS = [0, 1, 2, 3, 5, 7, 9, 11]
    COLOR_MAP = {
        'person': 'red', 'bicycle': 'orange', 'car': 'blue',
        'motorcycle': 'yellow', 'bus': 'lime', 'truck': 'cyan',
        'traffic light': 'green', 'stop sign': 'magenta'
    }
else: # GROUND
    CLASS_IDS = [87, 90, 96, 97, 98, 99, 100, 102, 103, 105, 113, 122, 123, 124, 125, 126, 130, 132]
    COLOR_MAP = {
        "road": "yellow", "pavement-merged": "orange", "floor-wood": "peru",
        "floor-other-merged": "saddlebrown", "gravel": "dimgray", "railroad": "slategray",
        "platform": "lightgray", "playingfield": "lime", "grass-merged": "green",
        "dirt-merged": "sienna", "sand": "khaki", "snow": "white", "river": "dodgerblue",
        "sea": "navy", "water-other": "blue", "rock-merged": "darkslategray",
        "mountain-merged": "rosybrown", "rug-merged": "tan",
    }

DATE_STR = datetime.now().strftime("%Y-%m-%d")
DATETIME_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, DATE_STR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, f"proc_{TARGET_TYPE}_{DATETIME_STR}.mp4")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================

def get_font(size):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except: return ImageFont.load_default()

def mask_to_bbox_detection(segments_info, segmentation_map):
    detections = []
    for segment in segments_info:
        segment_id = segment['id']
        instance_mask_tensor = (segmentation_map == segment_id).float()
        if instance_mask_tensor.sum() == 0: continue
        try:
            boxes = masks_to_boxes(instance_mask_tensor.unsqueeze(0))
            bbox = boxes.squeeze(0).tolist()
        except ValueError: continue
        detections.append({
            "box": [round(c, 2) for c in bbox],
            "class_id": segment['label_id'],
            "score": round(segment.get('score', 1.0), 4),
            "mask": instance_mask_tensor
        })
    return detections

def draw_results(image, detections, id_to_label, mode):
    # 1. Base image (The original video frame)
    base_rgba = image.convert("RGBA")

    # 2. Mask Layer (Where we draw the colored masks)
    mask_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # 3. Text Layer (Where we draw labels on top of everything)
    text_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw_text = ImageDraw.Draw(text_layer)

    font = get_font(CAPTION_FONT_SIZE)

    for det in detections:
        class_name = id_to_label.get(det['class_id'], "UNKNOWN").lower()
        color_name = COLOR_MAP.get(class_name, 'white')
        rgb = ImageColor.getrgb(color_name)
        label = f"{class_name}: {det['score']:.2f}"

        if mode == 'mask':
            # Create a temporary solid color layer for THIS detection
            # We apply MASK_ALPHA here
            color_fill = Image.new('RGBA', image.size, (*rgb, MASK_ALPHA))
            # Convert the boolean mask to a grayscale 'L' mask
            mask_binary = Image.fromarray((det['mask'].cpu().numpy() * 255).astype(np.uint8), mode='L')
            # Paste the color onto the MASK LAYER using the binary mask as the stencil
            mask_layer.paste(color_fill, (0,0), mask_binary)
        else:
            # For BBox, we draw directly on the text layer for convenience
            draw_text.rectangle(det['box'], outline=(*rgb, 255), width=3)

        # Draw Label (matching mask color) with dark background
        tx, ty = det['box'][0], det['box'][1] - CAPTION_FONT_SIZE - 5
        try:
            t_bbox = draw_text.textbbox((tx, ty), label, font=font)
            draw_text.rectangle(t_bbox, fill=(0, 0, 0, 160))
        except: pass
        draw_text.text((tx, ty), label, fill=color_name, font=font)

    # MAGIC STEP: alpha_composite BLENDS the layers instead of overwriting
    out = Image.alpha_composite(base_rgba, mask_layer)
    out = Image.alpha_composite(out, text_layer)

    return out.convert("RGB")

def process_video_inference():
    print(f"LOG: Loading model on {DEVICE}...")
    processor = MaskFormerImageProcessor.from_pretrained(MODEL_NAME)
    model = MaskFormerForInstanceSegmentation.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    id_to_label = model.config.id2label

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw, fh = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

    if SHOW_LIVE_PREVIEW:
        cv2.namedWindow('Processing...', cv2.WINDOW_NORMAL)

    frame_count, total_proc_time = 0, 0
    start_wall_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        frame_start = time.time()

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = {k: v.to(DEVICE) for k, v in processor(images=pil_img, return_tensors="pt").items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # label_ids_to_fuse=None keeps ground segments separate as requested
        segmentation = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[pil_img.size[::-1]], label_ids_to_fuse=None
        )[0]

        dets = mask_to_bbox_detection(segmentation['segments_info'], segmentation['segmentation'])
        filtered = [d for d in dets if d['score'] >= MIN_SCORE_THRESHOLD and d['class_id'] in CLASS_IDS]

        processed_pil = draw_results(pil_img, filtered, id_to_label, VISUALIZATION_MODE)
        final_frame = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
        out.write(final_frame)

        latency = (time.time() - frame_start) * 1000
        total_proc_time += latency

        if frame_count % 10 == 0 or frame_count == total_frames:
            avg_time = total_proc_time / frame_count
            eta = time.strftime('%H:%M:%S', time.gmtime((avg_time / 1000) * (total_frames - frame_count)))
            print(f"[{frame_count}/{total_frames}] Latency: {latency:4.0f}ms | Avg: {avg_time:4.0f}ms | ETA: {eta}")

        if SHOW_LIVE_PREVIEW:
            scaled = cv2.resize(final_frame, None, fx=LIVE_PREVIEW_SCALE_FACTOR, fy=LIVE_PREVIEW_SCALE_FACTOR)
            cv2.imshow('Processing...', scaled)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nFinished in {time.time() - start_wall_time:.2f}s. Saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    process_video_inference()
