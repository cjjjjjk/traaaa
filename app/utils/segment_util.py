from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
import cv2
import numpy as np
import os

# Define model path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "segformer-cityscapes")

try:
    processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
    print(f"[INFO] Loaded SegFormer model from {MODEL_PATH}")

except Exception as e:
    print(f"[ERROR] Failed to load SegFormer model from {MODEL_PATH}. Error: {e}")
    processor = None
    model = None


def detect_road_area(frame: np.ndarray):
    """
    Detect road area in the frame using SegFormer.
    Returns the number of pixels classified as road (label 0).
    """
    if model is None or processor is None:
        print("Model ", model)
        print("Processor ", processor)
        print("[ERROR] Model is not loaded. Cannot detect road area.")
        return 0

    try:
        # Preprocess image
        # SegFormer expects RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        inputs = processor(images=image_rgb, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post-process results
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        
        # Upsample logits to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image_rgb.shape[:2], # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        # Get prediction (argmax)
        pred_seg = upsampled_logits.argmax(dim=1)[0] # shape (height, width)
        
        # Move to CPU and numpy
        pred_seg_np = pred_seg.cpu().numpy()
        
        # Label 0 is road
        road_pixels = np.sum(pred_seg_np == 0)
        total_pixels = pred_seg_np.size
        ratio = road_pixels / total_pixels if total_pixels > 0 else 0
        
        # print(f"[INFO] Road detection: {road_pixels} pixels (Ratio: {ratio:.4f})")
        
        return road_pixels

    except Exception as e:
        print(f"[ERROR] in detect_road_area: {e}")
        return 0
