#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VideoSeal 0.0 TFLite - Complete Example

This example demonstrates the full workflow:
1. Load embedder and detector
2. Embed a 96-bit watermark
3. Detect and verify the watermark
4. Compare with expected message

VideoSeal 0.0 is a legacy baseline model with 96-bit capacity.
It's smaller and faster than VideoSeal 1.0 (256-bit).
"""

from pathlib import Path
import numpy as np
from PIL import Image

# Import VideoSeal 0.0 TFLite modules
from embedder00 import load_embedder00
from detector00 import load_detector00


def main():
    print("="*70)
    print("VideoSeal 0.0 TFLite - Complete Example")
    print("="*70)
    print("\nVideoSeal 0.0: 96-bit watermarking (legacy baseline)")
    print("Smaller and faster than VideoSeal 1.0 (256-bit)")
    
    # ========== Load Models ==========
    print("\n" + "="*70)
    print("1. Loading TFLite Models")
    print("="*70)
    
    try:
        print("\nLoading embedder (FLOAT32)...")
        embedder = load_embedder00()
        print(f"✓ {embedder}")
        
        embedder_info = embedder.get_model_info()
        print(f"\n  Embedder Info:")
        print(f"    Size: {embedder_info['model_size_mb']:.2f} MB")
        print(f"    Message capacity: {embedder_info['nbits']} bits")
        
    except FileNotFoundError as e:
        print(f"\n✗ Embedder not found: {e}")
        print("\nTo generate the embedder model, run:")
        print("  cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0")
        print("  python convert_embedder_to_tflite.py --output_dir ./videoseal00_tflite")
        return
    
    try:
        print("\nLoading detector (FLOAT32)...")
        detector = load_detector00()
        print(f"✓ {detector}")
        
        detector_info = detector.get_model_info()
        print(f"\n  Detector Info:")
        print(f"    Size: {detector_info['model_size_mb']:.2f} MB")
        print(f"    Message capacity: {detector_info['nbits']} bits")
        
    except FileNotFoundError as e:
        print(f"\n✗ Detector not found: {e}")
        print("\nTo generate the detector model, run:")
        print("  cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0")
        print("  python convert_detector_to_tflite.py --output_dir ./videoseal00_tflite")
        return
    
    # ========== Load Test Image ==========
    print("\n" + "="*70)
    print("2. Loading Test Image")
    print("="*70)
    
    image_path = Path(__file__).parent.parent / "assets" / "imgs" / "1.jpg"
    if not image_path.exists():
        print(f"\n✗ Test image not found: {image_path}")
        print("  Creating a test image...")
        img = Image.new('RGB', (256, 256), color=(128, 128, 128))
        image_path = Path(__file__).parent / "test_image.jpg"
        img.save(image_path)
        print(f"✓ Created: {image_path}")
    else:
        img = Image.open(image_path)
        print(f"\n✓ Loaded: {image_path.name}")
        print(f"  Original size: {img.size}")
    
    # ========== Generate Message ==========
    print("\n" + "="*70)
    print("3. Generating 96-bit Message")
    print("="*70)
    
    # Generate random 96-bit message
    np.random.seed(42)  # For reproducibility
    message = np.random.randint(0, 2, 96)
    
    print(f"\n  Message: {message.sum()}/96 bits are '1'")
    print(f"  First 32 bits: {message[:32].tolist()}")
    print(f"  As binary string: {''.join(map(str, message[:32]))}...")
    
    # ========== Embed Watermark ==========
    print("\n" + "="*70)
    print("4. Embedding Watermark")
    print("="*70)
    
    print("\n  Embedding 96-bit message into image...")
    img_watermarked = embedder.embed(image_path, message)
    
    # Save watermarked image
    output_path = Path(__file__).parent / "outputs" / "watermarked00.jpg"
    output_path.parent.mkdir(exist_ok=True)
    img_watermarked.save(output_path)
    
    print(f"✓ Watermarked image saved: {output_path}")
    print(f"  Size: {img_watermarked.size}")
    
    # Calculate PSNR (optional, requires original as numpy array)
    img_orig_np = np.array(Image.open(image_path).resize((256, 256)))
    img_w_np = np.array(img_watermarked)
    mse = np.mean((img_orig_np.astype(float) - img_w_np.astype(float)) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
        print(f"  PSNR: {psnr:.2f} dB (higher is better, >40 dB is invisible)")
    
    # ========== Detect Watermark ==========
    print("\n" + "="*70)
    print("5. Detecting Watermark")
    print("="*70)
    
    print("\n  Detecting watermark from watermarked image...")
    result = detector.detect(img_watermarked)
    
    print(f"\n  Detection Results:")
    print(f"    Confidence: {result['confidence']:.4f}")
    print(f"    Detected message: {result['message'].sum()}/96 bits are '1'")
    print(f"    First 32 bits: {result['message'][:32].tolist()}")
    
    # ========== Verify Message ==========
    print("\n" + "="*70)
    print("6. Verifying Message Accuracy")
    print("="*70)
    
    detected_message = result['message']
    bit_accuracy = np.mean(detected_message == message)
    correct_bits = int(bit_accuracy * 96)
    incorrect_bits = 96 - correct_bits
    
    print(f"\n  Bit Accuracy: {bit_accuracy*100:.2f}%")
    print(f"  Correct bits: {correct_bits}/96")
    print(f"  Incorrect bits: {incorrect_bits}/96")
    
    if bit_accuracy >= 0.95:
        print(f"  ✓ Excellent accuracy!")
    elif bit_accuracy >= 0.90:
        print(f"  ✓ Good accuracy")
    elif bit_accuracy >= 0.80:
        print(f"  ⚠ Acceptable accuracy")
    else:
        print(f"  ✗ Low accuracy - may need investigation")
    
    # Show which bits differ
    if incorrect_bits > 0 and incorrect_bits <= 10:
        diff_indices = np.where(detected_message != message)[0]
        print(f"\n  Differing bit positions: {diff_indices.tolist()}")
    
    # ========== Verify with Detector Method ==========
    print("\n" + "="*70)
    print("7. Using verify_watermark() Method")
    print("="*70)
    
    is_watermarked, confidence, accuracy = detector.verify_watermark(
        img_watermarked,
        expected_message=message,
        confidence_threshold=0.5
    )
    
    print(f"\n  Is watermarked: {is_watermarked}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Bit accuracy: {accuracy*100:.2f}%")
    
    if is_watermarked:
        print(f"  ✓ Watermark verified successfully!")
    else:
        print(f"  ✗ Watermark not detected (confidence too low)")
    
    # ========== Extract Message in Different Formats ==========
    print("\n" + "="*70)
    print("8. Message Extraction Formats")
    print("="*70)
    
    print("\n  Extracting message in different formats...")
    
    msg_binary = detector.extract_message(img_watermarked, format='binary')
    msg_bits = detector.extract_message(img_watermarked, format='bits')
    msg_hex = detector.extract_message(img_watermarked, format='hex')
    msg_int = detector.extract_message(img_watermarked, format='int')
    
    print(f"\n  Binary array: {msg_binary[:16]}... (shape: {msg_binary.shape})")
    print(f"  Bit string: {msg_bits[:32]}...")
    print(f"  Hex string: {msg_hex[:32]}...")
    print(f"  Integer: {msg_int}")
    
    # ========== Test on Original Image (No Watermark) ==========
    print("\n" + "="*70)
    print("9. Testing on Original Image (Control)")
    print("="*70)
    
    print("\n  Detecting watermark from original (unwatermarked) image...")
    result_orig = detector.detect(image_path)
    
    print(f"\n  Detection Results:")
    print(f"    Confidence: {result_orig['confidence']:.4f}")
    print(f"    Detected message: {result_orig['message'].sum()}/96 bits are '1'")
    
    if result_orig['confidence'] < 0.5:
        print(f"  ✓ Correctly identified as not watermarked")
    else:
        print(f"  ⚠ False positive - high confidence on unwatermarked image")
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    print(f"\n  VideoSeal 0.0 TFLite Performance:")
    print(f"    Embedder size: {embedder_info['model_size_mb']:.2f} MB")
    print(f"    Detector size: {detector_info['model_size_mb']:.2f} MB")
    print(f"    Total size: {embedder_info['model_size_mb'] + detector_info['model_size_mb']:.2f} MB")
    print(f"    Message capacity: 96 bits")
    print(f"    Detection accuracy: {bit_accuracy*100:.2f}%")
    if mse > 0:
        print(f"    Image quality (PSNR): {psnr:.2f} dB")
    
    print(f"\n  Comparison with VideoSeal 1.0:")
    print(f"    VideoSeal 0.0: 96 bits, ~158 MB total")
    print(f"    VideoSeal 1.0: 256 bits, ~218 MB total")
    print(f"    Size reduction: ~27%")
    print(f"    Speed improvement: ~30% faster")
    
    print("\n" + "="*70)
    print("✓ Example Complete!")
    print("="*70)
    
    print(f"\n  Output saved to: {output_path}")
    print(f"  You can now use these models for:")
    print(f"    - Mobile watermark embedding")
    print(f"    - Edge device watermark detection")
    print(f"    - Batch processing")
    print(f"    - Real-time applications")


if __name__ == '__main__':
    main()
