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
import cv2
from shared.custom_layers import (
    ClippedResidualAdd,
    ResizeByScale,
    ResizeToMatch,
    estimate_bottleneck_size,
    custom_depth_from_scale,
)
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
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_path, ext)))
    
    if not image_files:
        print(f"[warn] No images found in {dataset_path}")
        return []

    results = []
    
    for img_path in sorted(image_files):
        try:
            # Load HR image
            hr_image = load_rgb_image_full(img_path) # (H, W, 3)
            h, w = hr_image.shape[:2]
            
            # Degrade to LR
            lr_image = degrade_image(hr_image, scale, output_size=-1) # (h_lr, w_lr, 3)
            
            # Prepare for prediction (add batch dim)
            lr_batch = np.expand_dims(lr_image, axis=0)
            
            # Predict
            pred_rgb = model.predict(lr_batch, verbose=0)
            pred_rgb = np.clip(pred_rgb[0], 0.0, 1.0)
            
            # Convert to Y channel (Luma)
            hr_y = rgb_to_luma_bt601(tf.cast(hr_image, tf.float32)).numpy()
            pred_y = rgb_to_luma_bt601(tf.cast(pred_rgb, tf.float32)).numpy()
            
            # Shave borders
            if shave > 0:
                hr_y = hr_y[shave:-shave, shave:-shave, :]
                pred_y = pred_y[shave:-shave, shave:-shave, :]
                
            # Compute Metrics
            psnr = tf.image.psnr(hr_y, pred_y, max_val=1.0).numpy()
            ssim = tf.image.ssim(hr_y, pred_y, max_val=1.0).numpy()
            mse = np.mean((hr_y - pred_y) ** 2)
            
            results.append({
                'Dataset': dataset_name,
                'Filename': Path(img_path).name,
                'PSNR': float(psnr),
                'SSIM': float(ssim),
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
        model = tf.keras.models.load_model(args.model_path, custom_objects={
            'ClippedResidualAdd': ClippedResidualAdd,
            'ResizeByScale': ResizeByScale,
            'ResizeToMatch': ResizeToMatch
        })
    except Exception as e:
        print(f"[error] Failed to load model: {e}")
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
