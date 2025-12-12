import sys
from pathlib import Path

# Add project root to sys.path to allow imports from shared
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Disable XLA and JIT to avoid compatibility issues
tf.config.optimizer.set_jit(False)

from shared.custom_layers import (
    ClippedResidualAdd,
    ResizeByScale,
    ResizeToMatch,
    charbonnier_loss,
    psnr_metric,
)
from shared.pipeline import (
    load_rgb_image_full,
    rgb_to_luma_bt601,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SR model predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .keras model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the HR image.")
    parser.add_argument("--scale", type=float, required=True, help="Scale factor (e.g., 0.2).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output PNG.")
    parser.add_argument("--patch_size", type=int, default=100, help="Size of the patch to extract (in HR pixels).")
    parser.add_argument("--patch_x", type=int, default=None, help="Top-left X coordinate of patch (optional).")
    parser.add_argument("--patch_y", type=int, default=None, help="Top-left Y coordinate of patch (optional).")
    return parser.parse_args()

def compute_metrics(hr, pred):
    """Compute PSNR and SSIM on Y channel."""
    hr_y = rgb_to_luma_bt601(tf.convert_to_tensor(hr)).numpy()
    pred_y = rgb_to_luma_bt601(tf.convert_to_tensor(pred)).numpy()
    
    psnr = tf.image.psnr(hr_y, pred_y, max_val=1.0).numpy()
    if np.isinf(psnr):
        psnr = 100.0
    ssim = tf.image.ssim(hr_y, pred_y, max_val=1.0).numpy()
    return float(psnr), float(ssim)

from train_adaptive_unet import build_super_resolution_unet
import re

def main():
    args = parse_args()
    
    # 1. Load Model
    print(f"Loading model from {args.model_path}...")
    try:
        # Load saved model to get weights
        saved_model = tf.keras.models.load_model(args.model_path, custom_objects={
            'ClippedResidualAdd': ClippedResidualAdd,
            'ResizeByScale': ResizeByScale,
            'ResizeToMatch': ResizeToMatch,
            'charbonnier_loss': charbonnier_loss,
            'psnr': psnr_metric,
        }, compile=False)
        
        # Infer depth from model name
        model_name = saved_model.name
        depth_match = re.search(r"depth(\d+)", model_name)
        depth = int(depth_match.group(1)) if depth_match else 1
        
        print(f"Rebuilding model with flexible input shape (depth={depth})...")
        
        # Rebuild with flexible input
        model, _ = build_super_resolution_unet(
            scale=args.scale,
            depth_override=depth,
            input_size=None,
        )
        
        # Transfer weights
        model.set_weights(saved_model.get_weights())
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 2. Load HR Image
    print(f"Loading image from {args.image_path}...")
    if not Path(args.image_path).exists():
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
        
    hr_image = load_rgb_image_full(args.image_path) # (H, W, 3), float32 [0, 1]
    h, w = hr_image.shape[:2]
    
    # 3. Generate LR Image (Degrade)
    # We replicate the degradation logic: Downsample -> Upsample (Bicubic)
    # But wait, the model expects LR input (small size) if it's an upsampler?
    # Or does the model take LR and output HR?
    # The `train_adaptive_unet.py` uses `degrade_to_lr_tf` which returns UP-SAMPLED LR (same size as HR)
    # if the model is "Adaptive Depth U-Net" which usually operates on "same size" inputs (U-Net style).
    # Let's check `train_adaptive_unet.py` again.
    # It uses `degrade_to_lr_tf` which does: resize(down) -> resize(up, bicubic).
    # So the input to the model is the same spatial size as HR, but blurry.
    
    h_lr = max(1, int(round(h * args.scale)))
    w_lr = max(1, int(round(w * args.scale)))
    
    # Create LR (Low Res) - Downsample
    lr_small = cv2.resize(hr_image, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
    
    # Upsample back to HR size (Bicubic Baseline / Model Input)
    lr_bicubic = cv2.resize(lr_small, (w, h), interpolation=cv2.INTER_CUBIC)
    lr_bicubic = np.clip(lr_bicubic, 0.0, 1.0).astype(np.float32)
    
    # 4. Predict SR
    print("Running prediction...")
    input_batch = np.expand_dims(lr_bicubic, axis=0)
    sr_image = model.predict(input_batch, verbose=0)[0]
    sr_image = np.clip(sr_image, 0.0, 1.0)
    
    # 5. Select Patch
    patch_size = args.patch_size
    if args.patch_x is not None and args.patch_y is not None:
        x, y = args.patch_x, args.patch_y
    else:
        # Center crop default
        x = (w - patch_size) // 2
        y = (h - patch_size) // 2
        
    # Ensure bounds
    x = max(0, min(x, w - patch_size))
    y = max(0, min(y, h - patch_size))
    
    # Extract Patches
    hr_patch = hr_image[y:y+patch_size, x:x+patch_size, :]
    bicubic_patch = lr_bicubic[y:y+patch_size, x:x+patch_size, :]
    sr_patch = sr_image[y:y+patch_size, x:x+patch_size, :]
    
    # 6. Compute Metrics for Patches
    psnr_bic, ssim_bic = compute_metrics(hr_patch, bicubic_patch)
    psnr_sr, ssim_sr = compute_metrics(hr_patch, sr_patch)
    
    print(f"Bicubic Patch: PSNR={psnr_bic:.2f}, SSIM={ssim_bic:.4f}")
    print(f"SR Patch:      PSNR={psnr_sr:.2f}, SSIM={ssim_sr:.4f}")
    
    # 7. Plotting
    print("Generating plot...")
    fig = plt.figure(figsize=(20, 10))
    
    # Layout: 
    # Row 1: Original Image with Bounding Box
    # Row 2: GT Patch, Bicubic Patch, SR Patch
    
    # Original Image
    ax_orig = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax_orig.imshow(hr_image)
    ax_orig.set_title("Original Image (HR)", fontsize=16)
    ax_orig.axis('off')
    
    # Draw Rectangle
    rect = Rectangle((x, y), patch_size, patch_size, linewidth=2, edgecolor='yellow', facecolor='none')
    ax_orig.add_patch(rect)
    
    # Patches
    # Ground Truth
    ax_gt = plt.subplot2grid((2, 3), (1, 0))
    ax_gt.imshow(hr_patch)
    ax_gt.set_title("Ground Truth Patch", fontsize=14)
    ax_gt.axis('off')
    
    # Bicubic
    ax_bic = plt.subplot2grid((2, 3), (1, 1))
    ax_bic.imshow(bicubic_patch)
    ax_bic.set_title(f"Bicubic\nPSNR: {psnr_bic:.2f} dB / SSIM: {ssim_bic:.4f}", fontsize=14)
    ax_bic.axis('off')
    
    # SR
    ax_sr = plt.subplot2grid((2, 3), (1, 2))
    ax_sr.imshow(sr_patch)
    ax_sr.set_title(f"SR (Model)\nPSNR: {psnr_sr:.2f} dB / SSIM: {ssim_sr:.4f}", fontsize=14)
    ax_sr.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to {args.output_path}")

if __name__ == "__main__":
    main()
