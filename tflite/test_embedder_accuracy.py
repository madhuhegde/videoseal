#!/usr/bin/env python3
"""
Test VideoSeal TFLite Embedder Accuracy with Real Images

This script tests the TFLite embedder with various image types and
compares against PyTorch reference implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image
import time

import videoseal
from tflite.embedder import load_embedder
from tflite.detector import load_detector


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def create_test_images():
    """Create various test images."""
    images = {}
    
    # 1. Solid color
    images['solid_gray'] = Image.new('RGB', (256, 256), color=(128, 128, 128))
    
    # 2. Gradient
    gradient = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        gradient[i, :] = [i, i, i]
    images['gradient'] = Image.fromarray(gradient)
    
    # 3. Checkerboard
    checker = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i // 32 + j // 32) % 2 == 0:
                checker[i:i+32, j:j+32] = [255, 255, 255]
    images['checkerboard'] = Image.fromarray(checker)
    
    # 4. Random noise
    noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    images['noise'] = Image.fromarray(noise)
    
    # 5. Natural-like texture
    texture = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            texture[i, j] = [
                int(128 + 50 * np.sin(i / 10) * np.cos(j / 10)),
                int(128 + 50 * np.sin(i / 15) * np.cos(j / 15)),
                int(128 + 50 * np.sin(i / 20) * np.cos(j / 20))
            ]
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    images['texture'] = Image.fromarray(texture)
    
    return images


def test_embedder(pytorch_model, tflite_embedder, tflite_detector, images):
    """Test embedder with various images."""
    
    print("\n" + "="*80)
    print("VideoSeal TFLite Embedder Accuracy Test")
    print("="*80)
    
    results = []
    
    for name, img in images.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        # Generate random message
        message = np.random.randint(0, 2, 256)
        
        # Convert to numpy
        img_np = np.array(img)
        
        # PyTorch embedding
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        msg_tensor = torch.from_numpy(message).unsqueeze(0).float()
        
        start = time.time()
        with torch.no_grad():
            result_dict = pytorch_model.embed(img_tensor, msg_tensor)
            img_w_pytorch = result_dict['imgs_w'] if isinstance(result_dict, dict) else result_dict
        pytorch_time = time.time() - start
        
        img_w_pytorch_np = (img_w_pytorch.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # TFLite embedding
        start = time.time()
        img_w_tflite = tflite_embedder.embed(img, message, return_pil=False)
        tflite_time = time.time() - start
        
        # Compare outputs
        psnr_original_pytorch = compute_psnr(img_np, img_w_pytorch_np)
        psnr_original_tflite = compute_psnr(img_np, img_w_tflite)
        psnr_pytorch_tflite = compute_psnr(img_w_pytorch_np, img_w_tflite)
        
        print(f"\n1. Embedding Quality:")
        print(f"   PyTorch PSNR (original → watermarked): {psnr_original_pytorch:.2f} dB")
        print(f"   TFLite PSNR (original → watermarked):  {psnr_original_tflite:.2f} dB")
        print(f"   PSNR (PyTorch ↔ TFLite outputs):      {psnr_pytorch_tflite:.2f} dB")
        
        # Pixel difference
        diff = np.abs(img_w_pytorch_np.astype(np.float32) - img_w_tflite.astype(np.float32))
        print(f"\n2. Pixel Differences:")
        print(f"   Max difference:  {diff.max():.2f}")
        print(f"   Mean difference: {diff.mean():.2f}")
        print(f"   Std difference:  {diff.std():.2f}")
        
        # Detection accuracy
        result_pytorch = tflite_detector.detect(Image.fromarray(img_w_pytorch_np))
        result_tflite = tflite_detector.detect(Image.fromarray(img_w_tflite))
        
        acc_pytorch = (message == result_pytorch['message']).mean() * 100
        acc_tflite = (message == result_tflite['message']).mean() * 100
        
        print(f"\n3. Detection Results:")
        print(f"   PyTorch embedder → TFLite detector:")
        print(f"     Confidence: {result_pytorch['confidence']:.3f}")
        print(f"     Bit accuracy: {acc_pytorch:.2f}%")
        
        print(f"   TFLite embedder → TFLite detector:")
        print(f"     Confidence: {result_tflite['confidence']:.3f}")
        print(f"     Bit accuracy: {acc_tflite:.2f}%")
        
        # Speed
        print(f"\n4. Performance:")
        print(f"   PyTorch: {pytorch_time*1000:.2f} ms")
        print(f"   TFLite:  {tflite_time*1000:.2f} ms")
        print(f"   Speedup: {pytorch_time/tflite_time:.2f}×")
        
        # Verdict
        print(f"\n5. Verdict:")
        if psnr_pytorch_tflite > 40:
            print(f"   ✅ Outputs are very similar (PSNR > 40 dB)")
        elif psnr_pytorch_tflite > 30:
            print(f"   ⚠️  Outputs have some differences (PSNR 30-40 dB)")
        else:
            print(f"   ❌ Outputs differ significantly (PSNR < 30 dB)")
        
        if acc_tflite > 95:
            print(f"   ✅ Excellent detection accuracy (> 95%)")
        elif acc_tflite > 80:
            print(f"   ⚠️  Good detection accuracy (80-95%)")
        else:
            print(f"   ❌ Low detection accuracy (< 80%)")
        
        # Store results
        results.append({
            'name': name,
            'psnr_pytorch_tflite': psnr_pytorch_tflite,
            'acc_pytorch': acc_pytorch,
            'acc_tflite': acc_tflite,
            'pytorch_time': pytorch_time,
            'tflite_time': tflite_time
        })
    
    return results


def print_summary(results):
    """Print summary of all tests."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n| Image | PSNR (PT↔TFL) | PT Acc | TFL Acc | PT Time | TFL Time |")
    print("|-------|---------------|--------|---------|---------|----------|")
    
    for r in results:
        print(f"| {r['name']:13s} | {r['psnr_pytorch_tflite']:6.2f} dB | "
              f"{r['acc_pytorch']:5.1f}% | {r['acc_tflite']:6.1f}% | "
              f"{r['pytorch_time']*1000:5.1f}ms | {r['tflite_time']*1000:6.1f}ms |")
    
    # Averages
    avg_psnr = np.mean([r['psnr_pytorch_tflite'] for r in results])
    avg_acc_pytorch = np.mean([r['acc_pytorch'] for r in results])
    avg_acc_tflite = np.mean([r['acc_tflite'] for r in results])
    avg_pytorch_time = np.mean([r['pytorch_time'] for r in results])
    avg_tflite_time = np.mean([r['tflite_time'] for r in results])
    
    print(f"| {'AVERAGE':13s} | {avg_psnr:6.2f} dB | "
          f"{avg_acc_pytorch:5.1f}% | {avg_acc_tflite:6.1f}% | "
          f"{avg_pytorch_time*1000:5.1f}ms | {avg_tflite_time*1000:6.1f}ms |")
    
    print("\n" + "="*80)
    print("OVERALL VERDICT")
    print("="*80)
    
    if avg_psnr > 40 and avg_acc_tflite > 95:
        print("✅ TFLite embedder is EXCELLENT!")
        print("   - Very high output similarity")
        print("   - Excellent detection accuracy")
    elif avg_psnr > 30 and avg_acc_tflite > 80:
        print("⚠️  TFLite embedder is GOOD")
        print("   - Good output similarity")
        print("   - Good detection accuracy")
    else:
        print("❌ TFLite embedder needs improvement")
        print("   - Low output similarity or accuracy")
    
    print(f"\nAverage speedup: {avg_pytorch_time/avg_tflite_time:.2f}×")
    
    print("\n" + "="*80)


def main():
    print("Loading models...")
    
    # Load PyTorch model
    pytorch_model = videoseal.load('videoseal_1.0')
    pytorch_model.eval()
    print("✓ PyTorch model loaded")
    
    # Load TFLite models
    tflite_embedder = load_embedder(quantization='float32')
    print("✓ TFLite embedder loaded")
    
    tflite_detector = load_detector(quantization='int8')
    print("✓ TFLite detector loaded")
    
    # Create test images
    print("\nCreating test images...")
    images = create_test_images()
    print(f"✓ Created {len(images)} test images")
    
    # Run tests
    results = test_embedder(pytorch_model, tflite_embedder, tflite_detector, images)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()

