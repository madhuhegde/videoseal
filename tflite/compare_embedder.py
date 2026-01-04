#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare PyTorch and TFLite VideoSeal Embedders.

This script compares:
1. Output quality (PSNR, SSIM)
2. Bit accuracy (via detector)
3. Inference speed
4. Memory usage
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("Warning: scikit-image not installed. SSIM will not be computed.")
    ssim = None

import videoseal
from videoseal.tflite import load_embedder, load_detector


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images."""
    if ssim is None:
        return 0.0
    return ssim(img1, img2, channel_axis=2, data_range=255)


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and TFLite embedders")
    parser.add_argument(
        "--model",
        type=str,
        default="videoseal_1.0",
        help="Model name (default: videoseal_1.0)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image (default: generate random)"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of inference runs for speed test (default: 10)"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("VideoSeal Embedder Comparison: PyTorch vs TFLite")
    print("="*80)
    
    # 1. Load models
    print("\n1. Loading models...")
    print("   Loading PyTorch model...")
    pytorch_model = videoseal.load(args.model)
    pytorch_model.eval()
    
    print("   Loading TFLite embedder...")
    tflite_embedder = load_embedder(quantization='float32')
    
    print("   Loading TFLite detector (for verification)...")
    tflite_detector = load_detector(quantization='int8')
    
    # 2. Prepare test image
    print("\n2. Preparing test image...")
    if args.image:
        img = Image.open(args.image).convert('RGB')
        img = img.resize((256, 256), Image.LANCZOS)
    else:
        # Generate test pattern
        img = Image.new('RGB', (256, 256), color=(128, 128, 128))
        pixels = np.array(img)
        for i in range(0, 256, 32):
            pixels[i:i+16, :] = [200, 100, 150]
        img = Image.fromarray(pixels)
    
    img_np = np.array(img)
    print(f"   Image shape: {img_np.shape}")
    
    # 3. Generate message
    print("\n3. Generating random 256-bit message...")
    message = np.random.randint(0, 2, 256)
    print(f"   Message (first 32 bits): {message[:32]}")
    
    # 4. PyTorch embedding
    print("\n4. PyTorch embedding...")
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    msg_tensor = torch.from_numpy(message).unsqueeze(0).float()
    
    with torch.no_grad():
        start = time.time()
        img_w_pytorch = pytorch_model.embed(img_tensor, msg_tensor)
        pytorch_time = time.time() - start
    
    img_w_pytorch_np = (img_w_pytorch[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    print(f"   Time: {pytorch_time*1000:.2f} ms")
    
    # 5. TFLite embedding
    print("\n5. TFLite embedding...")
    start = time.time()
    img_w_tflite = tflite_embedder.embed(img, message, return_pil=False)
    tflite_time = time.time() - start
    print(f"   Time: {tflite_time*1000:.2f} ms")
    
    # 6. Compare outputs
    print("\n6. Comparing outputs...")
    
    # PSNR between original and watermarked
    psnr_pytorch = compute_psnr(img_np, img_w_pytorch_np)
    psnr_tflite = compute_psnr(img_np, img_w_tflite)
    print(f"   PyTorch PSNR: {psnr_pytorch:.2f} dB")
    print(f"   TFLite PSNR:  {psnr_tflite:.2f} dB")
    
    # SSIM
    if ssim is not None:
        ssim_pytorch = compute_ssim(img_np, img_w_pytorch_np)
        ssim_tflite = compute_ssim(img_np, img_w_tflite)
        print(f"   PyTorch SSIM: {ssim_pytorch:.4f}")
        print(f"   TFLite SSIM:  {ssim_tflite:.4f}")
    
    # PSNR between PyTorch and TFLite outputs
    psnr_diff = compute_psnr(img_w_pytorch_np, img_w_tflite)
    print(f"   PSNR (PyTorch vs TFLite): {psnr_diff:.2f} dB")
    
    # Pixel-wise difference
    max_diff = np.abs(img_w_pytorch_np.astype(np.float32) - img_w_tflite.astype(np.float32)).max()
    mean_diff = np.abs(img_w_pytorch_np.astype(np.float32) - img_w_tflite.astype(np.float32)).mean()
    print(f"   Max pixel difference: {max_diff:.2f}")
    print(f"   Mean pixel difference: {mean_diff:.2f}")
    
    # 7. Verify with detector
    print("\n7. Verifying with TFLite detector...")
    
    result_pytorch = tflite_detector.detect(Image.fromarray(img_w_pytorch_np))
    result_tflite = tflite_detector.detect(Image.fromarray(img_w_tflite))
    
    acc_pytorch = (message == result_pytorch['message']).mean() * 100
    acc_tflite = (message == result_tflite['message']).mean() * 100
    
    print(f"   PyTorch → Detector:")
    print(f"     Confidence: {result_pytorch['confidence']:.3f}")
    print(f"     Bit accuracy: {acc_pytorch:.2f}%")
    
    print(f"   TFLite → Detector:")
    print(f"     Confidence: {result_tflite['confidence']:.3f}")
    print(f"     Bit accuracy: {acc_tflite:.2f}%")
    
    # 8. Speed benchmark
    print(f"\n8. Speed benchmark ({args.num_runs} runs)...")
    
    # PyTorch
    pytorch_times = []
    for _ in range(args.num_runs):
        start = time.time()
        with torch.no_grad():
            _ = pytorch_model.embed(img_tensor, msg_tensor)
        pytorch_times.append(time.time() - start)
    
    # TFLite
    tflite_times = []
    for _ in range(args.num_runs):
        start = time.time()
        _ = tflite_embedder.embed(img, message, return_pil=False)
        tflite_times.append(time.time() - start)
    
    print(f"   PyTorch: {np.mean(pytorch_times)*1000:.2f} ± {np.std(pytorch_times)*1000:.2f} ms")
    print(f"   TFLite:  {np.mean(tflite_times)*1000:.2f} ± {np.std(tflite_times)*1000:.2f} ms")
    print(f"   Speedup: {np.mean(pytorch_times)/np.mean(tflite_times):.2f}×")
    
    # 9. Save outputs
    print("\n9. Saving outputs...")
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    Image.fromarray(img_np).save(output_dir / "original.jpg")
    Image.fromarray(img_w_pytorch_np).save(output_dir / "watermarked_pytorch.jpg")
    Image.fromarray(img_w_tflite).save(output_dir / "watermarked_tflite.jpg")
    
    # Difference image (amplified)
    diff = np.abs(img_w_pytorch_np.astype(np.float32) - img_w_tflite.astype(np.float32))
    diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)
    Image.fromarray(diff_amplified).save(output_dir / "difference_amplified.jpg")
    
    print(f"   Saved to: {output_dir}")
    
    # 10. Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Image size: 256×256")
    print(f"Message size: 256 bits")
    print()
    print("Quality Metrics:")
    print(f"  PyTorch PSNR: {psnr_pytorch:.2f} dB")
    print(f"  TFLite PSNR:  {psnr_tflite:.2f} dB")
    print(f"  Output difference: {psnr_diff:.2f} dB PSNR")
    print()
    print("Detection Accuracy:")
    print(f"  PyTorch: {acc_pytorch:.2f}%")
    print(f"  TFLite:  {acc_tflite:.2f}%")
    print()
    print("Speed:")
    print(f"  PyTorch: {np.mean(pytorch_times)*1000:.2f} ms")
    print(f"  TFLite:  {np.mean(tflite_times)*1000:.2f} ms")
    print(f"  Speedup: {np.mean(pytorch_times)/np.mean(tflite_times):.2f}×")
    print()
    print("Model Size:")
    print(f"  TFLite: {tflite_embedder.model_path.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    # Check if results are acceptable
    if psnr_diff > 40 and acc_tflite > 95:
        print("✅ TFLite embedder is working correctly!")
    else:
        print("⚠️  TFLite embedder may have issues:")
        if psnr_diff < 40:
            print(f"   - Low output similarity (PSNR: {psnr_diff:.2f} dB)")
        if acc_tflite < 95:
            print(f"   - Low bit accuracy ({acc_tflite:.2f}%)")
    
    print("="*80)


if __name__ == "__main__":
    main()

