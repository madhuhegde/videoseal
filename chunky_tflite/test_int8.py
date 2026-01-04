#!/usr/bin/env python3
"""Quick test of ChunkySeal INT8 TFLite model."""

import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from chunky_tflite import load_detector


def main():
    print("="*70)
    print("ChunkySeal INT8 TFLite Model Test")
    print("="*70)
    
    # Load INT8 model
    print("\n1. Loading INT8 model...")
    model_path = Path.home() / "work" / "models" / "chunkyseal_tflite" / \
                 "chunkyseal_detector_chunkyseal_256_int8.tflite"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    start = time.time()
    detector = load_detector(model_path=model_path)
    load_time = time.time() - start
    
    print(f"   ✓ Loaded in {load_time:.2f}s")
    print(f"   {detector}")
    
    # Get model info
    info = detector.get_model_info()
    print(f"\n2. Model Information:")
    print(f"   Quantization: {info['quantization']}")
    print(f"   Model size: {info['model_size_mb']:.2f} MB")
    print(f"   Message capacity: {info['message_length']} bits")
    
    # Load test image
    image_path = Path(__file__).parent.parent / "assets" / "imgs" / "1.jpg"
    if not image_path.exists():
        print(f"\n✗ Test image not found: {image_path}")
        return
    
    print(f"\n3. Loading test image: {image_path.name}")
    img = Image.open(image_path)
    print(f"   Image size: {img.size}")
    
    # Run inference multiple times for timing
    print(f"\n4. Running inference (5 iterations)...")
    times = []
    results = []
    
    for i in range(5):
        start = time.time()
        result = detector.detect(img)
        inference_time = time.time() - start
        times.append(inference_time)
        results.append(result)
        print(f"   Iteration {i+1}: {inference_time*1000:.2f} ms")
    
    avg_time = sum(times) / len(times)
    print(f"   Average: {avg_time*1000:.2f} ms")
    
    # Display results
    result = results[0]
    print(f"\n5. Detection Results:")
    print(f"   Confidence: {result['confidence']:.6f}")
    print(f"   Message (first 32 bits): {result['message'][:32]}")
    print(f"   Total '1' bits: {result['message'].sum()}/1024")
    
    # Verify consistency across runs
    print(f"\n6. Consistency Check:")
    all_same = all(
        (r['message'] == results[0]['message']).all() 
        for r in results
    )
    if all_same:
        print(f"   ✓ All 5 runs produced identical results")
    else:
        print(f"   ✗ Results differ across runs")
    
    # Extract message in different formats
    print(f"\n7. Message Formats:")
    message_hex = detector.extract_message(img, format='hex')
    message_bits = detector.extract_message(img, format='bits')
    print(f"   Hex:  {message_hex[:34]}...")
    print(f"   Bits: {message_bits[:64]}...")
    
    print("\n" + "="*70)
    print("✓ INT8 Model Test Complete!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Model size: {info['model_size_mb']:.2f} MB")
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Avg inference: {avg_time*1000:.2f} ms")
    print(f"  Capacity: {info['message_length']} bits (4× VideoSeal)")


if __name__ == '__main__':
    main()

