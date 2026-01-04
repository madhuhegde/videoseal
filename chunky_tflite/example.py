#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple example of using ChunkySeal TFLite detector.
"""

from pathlib import Path
from PIL import Image
from detector import load_detector


def main():
    print("="*70)
    print("ChunkySeal TFLite Detector - Simple Example")
    print("="*70)
    
    # Example 1: Load FLOAT32 detector
    print("\n1. Loading FLOAT32 TFLite detector...")
    detector = load_detector(model_name='chunkyseal')
    print(f"   {detector}")
    
    # Example 2: Load INT8 detector (if available)
    print("\n2. Loading INT8 TFLite detector...")
    try:
        detector_int8 = load_detector(model_name='chunkyseal', quantization='int8')
        print(f"   {detector_int8}")
        print("   ✓ INT8 model loaded successfully")
        # Use INT8 detector for the rest of the example
        detector = detector_int8
    except FileNotFoundError:
        print("   ⚠ INT8 model not found, using FLOAT32")
    
    # Display model info
    model_info = detector.get_model_info()
    print(f"\n   Model Information:")
    print(f"     Quantization: {model_info['quantization']}")
    print(f"     Model size: {model_info['model_size_mb']:.2f} MB")
    print(f"     Message capacity: {model_info['message_length']} bits (4× VideoSeal)")
    
    # Load test image
    image_path = Path(__file__).parent.parent / "assets" / "imgs" / "1.jpg"
    if not image_path.exists():
        print(f"\n✗ Test image not found: {image_path}")
        print("  Please provide an image path")
        return
    
    print(f"\n3. Loading image: {image_path.name}")
    img = Image.open(image_path)
    print(f"   Image size: {img.size}")
    
    # Detect watermark
    print("\n4. Detecting watermark...")
    result = detector.detect(img)
    
    print(f"   Detection confidence: {result['confidence']:.4f}")
    print(f"   Message (first 32 bits): {result['message'][:32]}")
    print(f"   Total '1' bits: {result['message'].sum()}/1024")
    
    # Verify watermark
    print("\n5. Verifying watermark...")
    is_watermarked, confidence, _ = detector.verify_watermark(
        img,
        confidence_threshold=0.5
    )
    
    if is_watermarked:
        print(f"   ✓ Watermark detected (confidence: {confidence:.4f})")
    else:
        print(f"   ✗ No watermark detected (confidence: {confidence:.4f})")
    
    # Extract message in different formats
    print("\n6. Extracting message in different formats...")
    message_hex = detector.extract_message(img, format='hex')
    message_int = detector.extract_message(img, format='int')
    message_bits = detector.extract_message(img, format='bits')
    
    print(f"   Hex:  {message_hex[:34]}...")  # Show first 32 hex chars
    print(f"   Bits: {message_bits[:64]}...")  # Show first 64 bits
    print(f"   Int:  {str(message_int)[:50]}...")  # Show first 50 digits
    
    # Show capacity comparison
    print("\n7. Capacity comparison:")
    print(f"   ChunkySeal: 1024 bits = 128 bytes")
    print(f"   VideoSeal:   256 bits =  32 bytes")
    print(f"   Capacity increase: 4×")
    
    print("\n" + "="*70)
    print("✓ Example complete!")
    print("="*70)


if __name__ == '__main__':
    main()

