#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare ChunkySeal PyTorch and TFLite embedder performance.

This script benchmarks:
1. Embedding quality (PSNR)
2. Detection accuracy (bit accuracy)
3. Inference speed
4. Model size
"""

import sys
import time
from pathlib import Path

# Add videoseal to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from PIL import Image

import videoseal
from videoseal.chunky_tflite import load_embedder as load_tflite_embedder
from videoseal.chunky_tflite import load_detector as load_tflite_detector


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two images."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def main():
    print("="*70)
    print("ChunkySeal PyTorch vs TFLite Embedder Comparison")
    print("="*70)
    print()
    
    # Load models
    print("Loading models...")
    
    # PyTorch
    print("  Loading PyTorch model...")
    model_pytorch = videoseal.load('chunkyseal')
    model_pytorch.eval()
    
    # TFLite
    print("  Loading TFLite embedder...")
    embedder_tflite = load_tflite_embedder()
    print()
    
    print("  Loading TFLite detector...")
    detector_tflite = load_tflite_detector()
    print()
    
    # Create test image
    print("Creating test image...")
    img_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img_np)
    
    # Create random message
    message = np.random.randint(0, 2, 1024)
    print(f"Message (first 32 bits): {message[:32]}")
    print()
    
    # PyTorch embedding
    print("="*70)
    print("PyTorch Embedding")
    print("="*70)
    
    with torch.no_grad():
        # Prepare inputs
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        msg_tensor = torch.from_numpy(message).unsqueeze(0).float()
        
        # Warm-up
        _ = model_pytorch.embed(img_tensor, msg_tensor)
        
        # Benchmark
        start = time.time()
        result = model_pytorch.embed(img_tensor, msg_tensor)
        pytorch_time = time.time() - start
        
        img_w_pytorch = result['imgs_w'][0].permute(1, 2, 0).cpu().numpy()
        img_w_pytorch = (img_w_pytorch * 255).clip(0, 255).astype(np.uint8)
    
    # Calculate PSNR
    psnr_pytorch = calculate_psnr(img_np, img_w_pytorch)
    
    print(f"  Inference time: {pytorch_time*1000:.2f} ms")
    print(f"  PSNR: {psnr_pytorch:.2f} dB")
    print()
    
    # Detect with PyTorch
    with torch.no_grad():
        img_w_tensor = torch.from_numpy(img_w_pytorch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        result = model_pytorch.detect(img_w_tensor)
        detected_msg_pytorch = (result['preds'][0] > 0).cpu().numpy().astype(int)
    
    bit_acc_pytorch = np.mean(message == detected_msg_pytorch) * 100
    print(f"  Bit accuracy (PyTorch detect): {bit_acc_pytorch:.2f}%")
    print()
    
    # TFLite embedding
    print("="*70)
    print("TFLite Embedding")
    print("="*70)
    
    # Warm-up
    _ = embedder_tflite.embed(img_pil, message, return_pil=False)
    
    # Benchmark
    start = time.time()
    img_w_tflite = embedder_tflite.embed(img_pil, message, return_pil=False)
    tflite_time = time.time() - start
    
    # Calculate PSNR
    psnr_tflite = calculate_psnr(img_np, img_w_tflite)
    
    print(f"  Inference time: {tflite_time*1000:.2f} ms")
    print(f"  PSNR: {psnr_tflite:.2f} dB")
    print()
    
    # Detect with TFLite
    result = detector_tflite.detect(Image.fromarray(img_w_tflite))
    detected_msg_tflite = result['message']
    confidence_tflite = result['confidence']
    
    bit_acc_tflite = np.mean(message == detected_msg_tflite) * 100
    print(f"  Bit accuracy (TFLite detect): {bit_acc_tflite:.2f}%")
    print(f"  Confidence: {confidence_tflite:.3f}")
    print()
    
    # Comparison
    print("="*70)
    print("Comparison Summary")
    print("="*70)
    print()
    
    print(f"{'Metric':<30} {'PyTorch':<20} {'TFLite':<20} {'Difference':<20}")
    print("-"*90)
    print(f"{'Inference Time (ms)':<30} {pytorch_time*1000:<20.2f} {tflite_time*1000:<20.2f} {(tflite_time/pytorch_time):<20.2f}x")
    print(f"{'PSNR (dB)':<30} {psnr_pytorch:<20.2f} {psnr_tflite:<20.2f} {psnr_tflite-psnr_pytorch:<20.2f}")
    print(f"{'Bit Accuracy (%)':<30} {bit_acc_pytorch:<20.2f} {bit_acc_tflite:<20.2f} {bit_acc_tflite-bit_acc_pytorch:<20.2f}")
    print()
    
    # Model size comparison
    print("="*70)
    print("Model Size Comparison")
    print("="*70)
    print()
    
    embedder_size = Path(embedder_tflite.model_path).stat().st_size / (1024**3)
    detector_size = Path(detector_tflite.model_path).stat().st_size / (1024**3)
    total_size = embedder_size + detector_size
    
    print(f"TFLite Embedder: {embedder_size:.2f} GB")
    print(f"TFLite Detector: {detector_size:.2f} GB")
    print(f"Total TFLite: {total_size:.2f} GB")
    print()
    
    # Key observations
    print("="*70)
    print("Key Observations")
    print("="*70)
    print()
    
    if abs(psnr_pytorch - psnr_tflite) < 1.0:
        print("✅ PSNR difference < 1 dB: Excellent quality preservation")
    elif abs(psnr_pytorch - psnr_tflite) < 3.0:
        print("✅ PSNR difference < 3 dB: Good quality preservation")
    else:
        print(f"⚠️  PSNR difference {abs(psnr_pytorch - psnr_tflite):.2f} dB: Quality degradation detected")
    
    if abs(bit_acc_pytorch - bit_acc_tflite) < 5.0:
        print("✅ Bit accuracy difference < 5%: Excellent detection accuracy")
    elif abs(bit_acc_pytorch - bit_acc_tflite) < 10.0:
        print("✅ Bit accuracy difference < 10%: Good detection accuracy")
    else:
        print(f"⚠️  Bit accuracy difference {abs(bit_acc_pytorch - bit_acc_tflite):.2f}%: Accuracy degradation detected")
    
    if tflite_time < pytorch_time:
        print(f"✅ TFLite is {pytorch_time/tflite_time:.2f}x faster than PyTorch")
    else:
        print(f"⚠️  TFLite is {tflite_time/pytorch_time:.2f}x slower than PyTorch")
    
    print()
    print("="*70)
    print("✅ Comparison completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

