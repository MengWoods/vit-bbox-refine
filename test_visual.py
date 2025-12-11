import torch
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from PIL import Image
import os
import gc # Garbage Collection for cleanup

# --- CONFIGURATION (EDIT THESE) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Paths
FINETUNED_MODEL_PATH = "models/checkpoints/fold_5_epoch_10_best"
BASE_MODEL_HUB_NAME = "facebook/maskformer-swin-base-coco" # Used for config
INPUT_IMAGE_PATH = "src/data/test_images/munich_000395_000019_leftImg8bit.png"

# Target Class Configuration
CUSTOM_ID_TO_NAME = {
    0: 'Background', 1: 'person', 2: 'rider', 3: 'car', 4: 'truck',
    5: 'bus', 6: 'train', 7: 'motorcycle', 8: 'bicycle'
}
NUM_CLASSES = len(CUSTOM_ID_TO_NAME) # 9

# --------------------------------------------------------------------------

from scipy.special import softmax # We need softmax for confidence scores

def run_manual_inference_and_process():
    # ... (Model loading logic remains identical and successful) ...
    # ... (Input image preparation remains identical) ...

    # 4. Inference
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = finetuned_model(**inputs)

    # 5. Manual Post-Processing and Filtering
    print("\n--- Manual Post-Processing and Slicing ---")

    # ⭐️ FIX: SLICE THE LOGITS TENSOR TO KEEP ONLY THE FIRST NUM_CLASSES (9) ⭐️
    # The output is [1, 100, 35]. We only want [1, 100, 9].

    # Get the raw logits and mask predictions
    raw_logits = outputs.class_queries_logits
    raw_masks = outputs.masks_queries_logits

    # Slice the logits to only include the meaningful 9 classes (index 0 to 8)
    # The post-processor expects the last dimension to be C+1 (classes + no-object).
    # Since your C=8, it should be 9. The model gives 35. Let's slice to 9.

    # We will slice to NUM_CLASSES (9) which includes 8 classes + 1 background/no-object.
    sliced_logits = raw_logits[:, :, :NUM_CLASSES]

    # ⭐️ The MaskFormer processor expects C+1 or C logits depending on the task/model.
    # If C=8, and the model was trained with no_object, it expects 9.
    # Since the original model output 35, let's try slicing to 35 - (35-9) = 9

    # Convert logits to probabilities (confidence scores)
    scores = softmax(sliced_logits.cpu().squeeze(0).numpy(), axis=-1)

    # Get the predicted class IDs and confidence scores for each query
    labels = scores.argmax(axis=-1)
    scores = scores.max(axis=-1)

    # Resize the raw masks to the original image size
    H, W = original_height, original_width
    masks = torch.nn.functional.interpolate(raw_masks, size=(H, W), mode="bilinear", align_corners=False)
    masks = masks.squeeze(0).sigmoid().cpu().numpy() > 0.5 # Threshold masks

    # Now build the segment info, filtering by our expected classes (1-8)
    results = {'segments_info': []}
    MIN_CONFIDENCE = 0.5 # Confidence threshold

    for mask_idx in range(len(labels)):
        label_id = labels[mask_idx]
        score = scores[mask_idx]

        # Check against our expected target IDs (1-8)
        # Note: We rely on the mask being non-empty, which is implicitly handled by the model.
        if label_id in TARGET_CLASS_IDS and score > MIN_CONFIDENCE:
            segment = {
                'id': mask_idx + 1,
                'label_id': int(label_id),
                'score': float(score),
                'mask': masks[mask_idx] # Keep the mask for visualization
            }
            results['segments_info'].append(segment)

    # Recreate the segmentation map (this is complex, simplified for debug)
    # We rely on the visualization function using the segments_info and the original outputs.
    # Since the original processor failed, we manually create the results dict:

    # ⭐️ The visualization function needs 'segmentation' tensor. We will skip creating it
    # for this minimal test and just rely on 'segments_info'.
    # We will use the segments_info we manually created to print final results.

    segments_info = results.get('segments_info', [])

    # 7. Print Final Segment Results (Filtering successful if this is not empty)
    print("\n--- Final Segment Results (Manual Filter) ---")

    if not segments_info:
        print("Manual filtering still resulted in NO segments above confidence threshold.")
        print("This suggests the model is truly predicting low confidence for classes 1-8.")

    else:
        # Print a few example segments
        # ... (Same print logic as before) ...
        print(f"Total Segments Found: {len(segments_info)}")
        print("Example Segments (First 5):")
        for i, segment in enumerate(segments_info[:5]):
            print(f"  Segment {i+1}: ID={segment['id']}, Label_ID={segment['label_id']}, Score={segment['score']:.4f}")

        predicted_ids = sorted(list({segment['label_id'] for segment in segments_info}))
        print(f"\nUnique Label IDs Predicted: {predicted_ids}")

def run_minimal_inference_and_print():
    print(f"--- Starting Minimal Inference Test on: {FINETUNED_MODEL_PATH} ---")
    print(f"Using device: {DEVICE}")

    # 1. Load Common Processor
    try:
        processor = MaskFormerImageProcessor.from_pretrained(BASE_MODEL_HUB_NAME)
    except Exception as e:
        print(f"❌ Error loading Image Processor: {e}")
        return

    # 2. Load Fine-Tuned Model (Applying the critical size mismatch fix)
    try:
        # 2a. Load the base configuration
        config = MaskFormerForInstanceSegmentation.config_class.from_pretrained(BASE_MODEL_HUB_NAME)

        # 2b. OVERRIDE num_labels to 34. This parameter forces the head size to 134 (100 queries + 34 classes),
        # matching the checkpoint's saved weights size.
        config.num_labels = 34
        print("Model configuration override applied: num_labels set to 34.")

        # 2c. Load the model using the modified configuration, ignoring mismatches
        finetuned_model = MaskFormerForInstanceSegmentation.from_pretrained(
            FINETUNED_MODEL_PATH,
            config=config,
            ignore_mismatched_sizes=True
        ).to(DEVICE)
        finetuned_model.eval()
        print("✅ Fine-Tuned Model loaded successfully (size mismatch bypassed).")

    except Exception as e:
        print(f"❌ CRITICAL FAILURE during Model Load: {e}")
        return

    # 3. Prepare Input Image
    try:
        original_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
        original_width, original_height = original_image.size
        print(f"Image loaded: {INPUT_IMAGE_PATH} ({original_width}x{original_height})")

        inputs = processor(original_image, return_tensors="pt").to(DEVICE)
        target_sizes = [(original_height, original_width)]
    except Exception as e:
        print(f"❌ Error preparing input image: {e}")
        return

    # 4. Inference
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = finetuned_model(**inputs)

    # 5. Post-Processing and Raw Output Inspection
    print("\n--- Inspecting Model Output ---")

    # Print raw output structure
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"Raw Output - {key}: Shape {value.shape}, Device {value.device}")
        elif isinstance(value, list):
            print(f"Raw Output - {key}: List of length {len(value)}, First element shape: {value[0].shape}")
        else:
            print(f"Raw Output - {key}: Type {type(value)}")

    # 6. Post-Processing (Generate Segments Info)
    print("\n--- Running Post-Processing ---")

    try:
        results = processor.post_process_instance_segmentation(
            outputs=outputs,
            target_sizes=target_sizes,
        )[0]
    except Exception as e:
        print(f"❌ Error during Post-Processing: {e}")
        # Print raw outputs again for better debug if post-processing fails
        return

    # 7. Print Final Segment Results
    print("\n--- Final Segment Results ---")
    segments_info = results.get('segments_info', [])

    if not segments_info:
        print("Post-processor returned NO segments (empty list for 'segments_info').")
        print("This confirms the reason for the lack of color in the visualization.")

    else:
        # Print a few example segments
        print(f"Total Segments Found: {len(segments_info)}")
        print("Example Segments (First 5):")
        for i, segment in enumerate(segments_info[:5]):
            print(f"  Segment {i+1}: ID={segment['id']}, Label_ID={segment['label_id']}, Score={segment['score']:.4f}")

        # Print unique label IDs found
        predicted_ids = sorted(list({segment['label_id'] for segment in segments_info}))
        target_ids_list = sorted(list(CUSTOM_ID_TO_NAME.keys()))

        print(f"\nUnique Label IDs Predicted: {predicted_ids}")
        print(f"Expected Target IDs (Custom Map): {target_ids_list}")


    # 8. Cleanup
    del finetuned_model
    gc.collect()
    torch.cuda.empty_cache()
    print("\n--- Test Complete ---")


if __name__ == "__main__":
    run_minimal_inference_and_print()
