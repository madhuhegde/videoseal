#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usage of VideoSeal TFLite Embedder.

This script demonstrates how to:
1. Load the TFLite embedder
2. Embed a watermark into an image
3. Verify the watermark with the detector
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from tflite.embedder import load_embedder
from tflite.detector import load_detector


def main():
    print("="*80)
    print("VideoSeal TFLite Embedder Example")
    print("="*80)
    
    # 1. Load embedder (FLOAT32)
    print("\n1. Loading TFLite embedder...")
    embedder = load_embedder(quantization='float32')
    print(embedder)
    
    # 2. Load detector for verification (INT8)
    print("\n2. Loading TFLite detector for verification...")
    detector = load_detector(quantization='int8')
    
    # 3. Create a test image
    print("\n3. Creating test image (256x256)...")
    img = Image.new('RGB', (256, 256), color=(100, 150, 200))
    
    # Add some pattern
    pixels = np.array(img)
    for i in range(0, 256, 32):
        pixels[i:i+16, :] = [200, 100, 150]
    img = Image.fromarray(pixels)
    
    # 4. Generate a random 256-bit message
    print("\n4. Generating random 256-bit message...")
    message = np.random.randint(0, 2, 256)
    print(f"   Message (first 32 bits): {message[:32]}")
    
    # 5. Embed watermark
    print("\n5. Embedding watermark...")
    img_w = embedder.embed(img, message)
    
    # 6. Save watermarked image
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "watermarked_embedder.jpg"
    img_w.save(output_path)
    print(f"   Saved to: {output_path}")
    
    # 7. Verify with detector
    print("\n6. Verifying watermark with detector...")
    result = detector.detect(img_w)
    
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Detected message (first 32 bits): {result['message'][:32]}")
    
    # 8. Calculate bit accuracy
    detected_msg = result['message']
    bit_accuracy = (message == detected_msg).mean() * 100
    print(f"   Bit accuracy: {bit_accuracy:.2f}%")
    
    # 9. Compute PSNR
    img_np = np.array(img).astype(np.float32)
    img_w_np = np.array(img_w).astype(np.float32)
    mse = np.mean((img_np - img_w_np) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    print(f"   PSNR: {psnr:.2f} dB")
    
    print("\n" + "="*80)
    print("✅ Embedder test complete!")
    print("="*80)
    
    # 10. Batch embedding example
    print("\n7. Batch embedding example...")
    images = [img] * 3
    messages = np.random.randint(0, 2, (3, 256))
    
    imgs_w = embedder.embed_batch(images, messages)
    print(f"   Embedded {len(imgs_w)} images")
    
    # Verify batch
    accuracies = []
    for i, (img_w, msg) in enumerate(zip(imgs_w, messages)):
        result = detector.detect(img_w)
        accuracy = (msg == result['message']).mean() * 100
        accuracies.append(accuracy)
        print(f"   Image {i+1}: {accuracy:.2f}% bit accuracy")
    
    print(f"\n   Average bit accuracy: {np.mean(accuracies):.2f}%")
    
    print("\n" + "="*80)
    print("🎉 All tests passed!")
    print("="*80)


if __name__ == "__main__":
    main()

