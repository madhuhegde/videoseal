#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usage of ChunkySeal TFLite Embedder.

This script demonstrates how to:
1. Load the ChunkySeal TFLite embedder
2. Embed a 1024-bit watermark into an image
3. Verify the watermark with the detector
4. Compare with PyTorch implementation
"""

import sys
from pathlib import Path

# Add videoseal to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from chunky_tflite import load_embedder, load_detector


def example1_basic_embedding():
    """Example 1: Basic watermark embedding."""
    print("="*70)
    print("Example 1: Basic Watermark Embedding")
    print("="*70)
    print()
    
    # Load embedder
    embedder = load_embedder()
    print()
    
    # Create a test image
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    
    # Create a random 1024-bit message
    message = np.random.randint(0, 2, 1024)
    print(f"Message (first 32 bits): {message[:32]}")
    print()
    
    # Embed watermark
    print("Embedding watermark...")
    img_w = embedder.embed(img, message)
    
    print(f"✓ Watermark embedded successfully")
    print(f"  Input size: {img.size}")
    print(f"  Output size: {img_w.size}")
    print()


def example2_embed_and_detect():
    """Example 2: Embed and detect watermark."""
    print("="*70)
    print("Example 2: Embed and Detect Watermark")
    print("="*70)
    print()
    
    # Load embedder and detector
    embedder = load_embedder()
    print()
    detector = load_detector()
    print()
    
    # Create a test image
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    
    # Create a random 1024-bit message
    message = np.random.randint(0, 2, 1024)
    print(f"Original message (first 32 bits): {message[:32]}")
    print()
    
    # Embed watermark
    print("Embedding watermark...")
    img_w = embedder.embed(img, message)
    print("✓ Watermark embedded")
    print()
    
    # Detect watermark
    print("Detecting watermark...")
    result = detector.detect(img_w)
    
    detected_msg = result['message']
    confidence = result['confidence']
    
    print(f"✓ Watermark detected")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Detected message (first 32 bits): {detected_msg[:32]}")
    print()
    
    # Calculate bit accuracy
    bit_accuracy = np.mean(message == detected_msg) * 100
    print(f"Bit accuracy: {bit_accuracy:.2f}%")
    print()


def example3_batch_embedding():
    """Example 3: Batch embedding."""
    print("="*70)
    print("Example 3: Batch Embedding")
    print("="*70)
    print()
    
    # Load embedder
    embedder = load_embedder()
    print()
    
    # Create multiple test images
    num_images = 5
    images = []
    for i in range(num_images):
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        images.append(Image.fromarray(img))
    
    # Create random messages
    messages = np.random.randint(0, 2, (num_images, 1024))
    
    print(f"Embedding {num_images} images...")
    imgs_w = embedder.embed_batch(images, messages)
    
    print(f"✓ All images watermarked successfully")
    print(f"  Number of images: {len(imgs_w)}")
    print()


def example4_real_image():
    """Example 4: Real image embedding (if available)."""
    print("="*70)
    print("Example 4: Real Image Embedding")
    print("="*70)
    print()
    
    # Check if test image exists
    test_img_path = Path.home() / "work" / "videoseal" / "videoseal" / "assets" / "test.jpg"
    
    if not test_img_path.exists():
        print(f"⚠️  Test image not found: {test_img_path}")
        print("   Using synthetic image instead")
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img)
    else:
        print(f"Loading test image: {test_img_path}")
        img = Image.open(test_img_path).convert('RGB')
    
    print(f"  Image size: {img.size}")
    print()
    
    # Load embedder and detector
    embedder = load_embedder()
    print()
    detector = load_detector()
    print()
    
    # Create a message
    message = np.random.randint(0, 2, 1024)
    print(f"Message (first 32 bits): {message[:32]}")
    print()
    
    # Embed watermark
    print("Embedding watermark...")
    img_w = embedder.embed(img, message)
    print("✓ Watermark embedded")
    print()
    
    # Detect watermark
    print("Detecting watermark...")
    result = detector.detect(img_w)
    
    detected_msg = result['message']
    confidence = result['confidence']
    
    print(f"✓ Watermark detected")
    print(f"  Confidence: {confidence:.3f}")
    print()
    
    # Calculate bit accuracy
    bit_accuracy = np.mean(message == detected_msg) * 100
    print(f"Bit accuracy: {bit_accuracy:.2f}%")
    print()
    
    # Calculate PSNR
    img_np = np.array(img).astype(np.float32)
    img_w_np = np.array(img_w).astype(np.float32)
    mse = np.mean((img_np - img_w_np) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
        print(f"PSNR: {psnr:.2f} dB")
    else:
        print("PSNR: Infinite (images are identical)")
    print()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ChunkySeal TFLite Embedder Examples")
    print("="*70)
    print()
    
    try:
        example1_basic_embedding()
        example2_embed_and_detect()
        example3_batch_embedding()
        example4_real_image()
        
        print("="*70)
        print("✅ All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

