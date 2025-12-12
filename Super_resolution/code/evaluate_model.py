"""
Evaluate a trained Super-Resolution model on standard test datasets (Set14, Urban100).
"""
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from shared
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import glob
import os
import numpy as np
import tensorflow as tf
# Disable XLA compilation to avoid ScaleAndTranslate errors with ResizeByScale
tf.config.optimizer.set_jit(False)
import cv2
from shared.custom_layers import (
    ClippedResidualAdd,
    ResizeByScale,
    ResizeToMatch,
    estimate_bottleneck_size,
    custom_depth_from_scale,
    charbonnier_loss,
    psnr_metric,
)
from train_adaptive_unet import build_super_resolution_unet
from shared.pipeline import (
    load_rgb_image_full,
    degrade_image,
    rgb_to_luma_bt601,
)

# Disable JIT for Resize ops
tf.config.optimizer.set_jit(False)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SR model on test datasets.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .keras model.")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Root directory containing test datasets (Set14, Urban100).")
    parser.add_argument("--scale", type=float, required=True, help="Scale factor used for training/evaluation.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the evaluation summary CSV.")
    parser.add_argument("--shave", type=int, default=None, help="Pixels to shave off borders (default: scale-dependent).")
    return parser.parse_args()

def evaluate_dataset(model, dataset_name, dataset_path, scale, shave):
    """
    Evaluates the model on a single dataset.
    Returns a list of dictionaries containing metrics for each image.
    """
    print(f"Evaluating on {dataset_name} from {dataset_path}...")
    
    # Recursively find all images
    # We specifically look for HR images to serve as ground truth.
    # Common patterns: *HR.png, *HR.bmp, or just images if no HR suffix is used (but here we know structure)
    # The user has structure like Set14/image_SRF_2/img_001_SRF_2_HR.png
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                # Filter for HR images if "HR" is in the filename, or accept all if we can't distinguish
                # Given the user's structure, we should prefer *HR.png
                if "HR" in file:
                    image_files.append(os.path.join(root, file))
    
    # Fallback: if no "HR" files found, maybe they are just named normally?
    if not image_files:
         for root, _, files in os.walk(dataset_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                     image_files.append(os.path.join(root, file))

    # Remove duplicates if any (unlikely with os.walk but good practice)
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"[warn] No images found in {dataset_path}")
        return []

    results = []
    
    for img_path in image_files:
        try:
            # Load HR image
            hr_image = load_rgb_image_full(img_path) # (H, W, 3)
            h, w = hr_image.shape[:2]
            
            # Degrade to LR (custom logic to preserve aspect ratio and original size)
            # We want to simulate the degradation process: Downsample -> Upsample
            # But crucially, we must upsample back to EXACTLY (H, W)
            h_lr = max(1, int(round(h * scale)))
            w_lr = max(1, int(round(w * scale)))
            
            lr_small = cv2.resize(hr_image, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
            lr_image = cv2.resize(lr_small, (w, h), interpolation=cv2.INTER_CUBIC)
            lr_image = lr_image.astype(np.float32)
            lr_image = np.clip(lr_image, 0.0, 1.0)
            
            # Prepare for prediction (add batch dim)
            lr_batch = np.expand_dims(lr_image, axis=0)
            
            # Predict
            pred_rgb = model.predict(lr_batch, verbose=0)
            pred_rgb = np.clip(pred_rgb[0], 0.0, 1.0)
            
            # Convert to Y channel (Luma)
            hr_y = rgb_to_luma_bt601(tf.cast(hr_image, tf.float32)).numpy()
            pred_y = rgb_to_luma_bt601(tf.cast(pred_rgb, tf.float32)).numpy()
            
            # rgb_to_luma_bt601 might return 4D (1, H, W, 1) due to broadcasting
            if hr_y.ndim == 4:
                hr_y = hr_y[0]
            if pred_y.ndim == 4:
                pred_y = pred_y[0]
            
            # Shave borders
            if shave > 0:
                hr_y = hr_y[shave:-shave, shave:-shave, :]
                pred_y = pred_y[shave:-shave, shave:-shave, :]
                
            # Compute Metrics
            psnr = tf.image.psnr(hr_y, pred_y, max_val=1.0).numpy()
            if np.isinf(psnr):
                psnr = 100.0
            ssim = tf.image.ssim(hr_y, pred_y, max_val=1.0).numpy()
            mse = np.mean((hr_y - pred_y) ** 2)
            
            results.append({
                'Dataset': dataset_name,
                'Filename': Path(img_path).name,
                'PSNR': float(psnr) if np.ndim(psnr) == 0 else psnr.item(),
                'SSIM': float(ssim) if np.ndim(ssim) == 0 else ssim.item(),
                'MSE': float(mse)
            })
            
        except Exception as e:
            print(f"[error] Failed to evaluate {img_path}: {e}")
            
    return results

def main():
    args = parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"[error] Model not found: {args.model_path}")
        sys.exit(1)
        
    print(f"Loading model from {args.model_path}...")
    try:
        # 1. Load the saved model to get weights and config
        # We use custom_objects to ensure all layers/losses are found
        saved_model = tf.keras.models.load_model(args.model_path, custom_objects={
            'ClippedResidualAdd': ClippedResidualAdd,
            'ResizeByScale': ResizeByScale,
            'ResizeToMatch': ResizeToMatch,
            'charbonnier_loss': charbonnier_loss,
            'psnr': psnr_metric,
        })
        
        # 2. Rebuild the model with flexible input shape (None, None, 3)
        # We need to extract parameters from the saved model or args
        # Since we don't have easy access to the exact arguments used for building (unless saved in config),
        # we can try to infer or just use the weights transfer.
        # Ideally, we call build_super_resolution_unet with input_size=None.
        
        # Let's try to get depth from the model name or config if possible, 
        # or just rely on the fact that weights must match.
        # The build function needs: scale, base_channels, residual_head_channels, depth_override, input_size, max_depth
        
        # We can try to infer depth from the layer count or name?
        # Name format: U-Net_SR_scale{scale:.2f}_depth{depth}
        model_name = saved_model.name
        import re
        depth_match = re.search(r"depth(\d+)", model_name)
        depth = int(depth_match.group(1)) if depth_match else 1 # Default or fallback
        
        print(f"Rebuilding model with flexible input shape (depth={depth})...")
        
        # We assume standard channel sizes as defaults in train script: base=64, head=64
        # If these were changed, weight loading will fail. 
        # For now, we assume defaults or what was used in the user's run (which seemed to use defaults).
        
        flexible_model, _ = build_super_resolution_unet(
            scale=args.scale,
            depth_override=depth,
            input_size=None, # This enables flexible shape
        )
        
        # 3. Transfer weights
        flexible_model.set_weights(saved_model.get_weights())
        
        # Explicitly compile with jit_compile=False to ensure XLA is disabled
        flexible_model.compile(jit_compile=False)
        
        model = flexible_model
        print("Model rebuilt successfully with flexible input shape.")

    except Exception as e:
        print(f"[error] Failed to load or rebuild model: {e}")
        sys.exit(1)

    # Determine shave size if not provided
    if args.shave is None:
        inv_scale = 1.0 / args.scale if args.scale > 0 else 0.0
        scale_factor = int(round(inv_scale)) if inv_scale > 0 else 0
        shave = 2 * scale_factor if scale_factor > 0 else 0
    else:
        shave = args.shave
    
    print(f"Evaluation Shave: {shave} pixels")

    # Define datasets to evaluate
    # We look for 'Set14' and 'Urban100' (or 'X2 Urban100' as seen in user image)
    # We will search for directories that contain these strings
    
    target_datasets = ['Set14', 'Urban100']
    found_datasets = []
    
    root = Path(args.test_data_dir)
    if not root.exists():
         print(f"[error] Test data root not found: {root}")
         sys.exit(1)

    # Simple discovery: look for subdirectories
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    
    for target in target_datasets:
        match = None
        for d in subdirs:
            if target.lower() in d.name.lower():
                match = d
                break
        if match:
            found_datasets.append((target, match))
        else:
            print(f"[warn] Dataset '{target}' not found in {args.test_data_dir}")

    if not found_datasets:
        print("[error] No target datasets found.")
        sys.exit(1)

    all_results = []
    
    for name, path in found_datasets:
        dataset_results = evaluate_dataset(model, name, path, args.scale, shave)
        all_results.extend(dataset_results)
        
        # Calculate average for this dataset
        if dataset_results:
            avg_psnr = np.mean([r['PSNR'] for r in dataset_results])
            avg_ssim = np.mean([r['SSIM'] for r in dataset_results])
            print(f"  -> {name} Average: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")

    # Save to CSV
    if all_results:
        print(f"Saving results to {args.output_csv}...")
        keys = ['Dataset', 'Filename', 'PSNR', 'SSIM', 'MSE']
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print("Done.")
    else:
        print("[warn] No results generated.")

if __name__ == "__main__":
    main()
