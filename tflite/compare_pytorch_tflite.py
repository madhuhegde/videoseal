#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare VideoSeal PyTorch and TFLite detector performance.

This script demonstrates watermark detection using both PyTorch and TFLite
implementations and compares their results.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import videoseal
from tflite.detector import VideoSealDetectorTFLite


def create_watermarked_image(
    image_path: str,
    output_path: str,
    model_name: str = "videoseal"
) -> np.ndarray:
    """
    Create a watermarked image using PyTorch VideoSeal.
    
    Args:
        image_path: Path to input image
        output_path: Path to save watermarked image
        model_name: VideoSeal model variant
    
    Returns:
        Embedded message (256-bit binary array)
    """
    print("\n" + "="*70)
    print("Creating Watermarked Image (PyTorch)")
    print("="*70)
    
    # Load model
    print(f"Loading VideoSeal model: {model_name}")
    model = videoseal.load(model_name)
    model.eval()
    
    # Load image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).float()
    
    print(f"Image shape: {img_tensor.shape}")
    
    # Embed watermark
    print("Embedding watermark...")
    with torch.no_grad():
        outputs = model.embed(img_tensor, is_video=False)
    
    imgs_w = outputs["imgs_w"]
    msgs = outputs["msgs"]
    
    # Save watermarked image
    to_pil = T.ToPILImage()
    img_w_pil = to_pil(imgs_w[0])
    img_w_pil.save(output_path)
    
    # Calculate PSNR
    mse = torch.mean((imgs_w - img_tensor) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    
    print(f"✓ Watermarked image saved: {output_path}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Embedded message (first 32 bits): {msgs[0, :32].numpy()}")
    
    return msgs[0].numpy()


def detect_pytorch(
    image_path: str,
    model_name: str = "videoseal"
) -> dict:
    """
    Detect watermark using PyTorch VideoSeal.
    
    Args:
        image_path: Path to watermarked image
        model_name: VideoSeal model variant
    
    Returns:
        Detection results dictionary
    """
    print("\n" + "="*70)
    print("Detecting Watermark (PyTorch)")
    print("="*70)
    
    # Load model
    print(f"Loading VideoSeal model: {model_name}")
    model = videoseal.load(model_name)
    model.eval()
    
    # Load image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).float()
    
    # Detect watermark
    print("Detecting watermark...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.detect(img_tensor, is_video=False)
    inference_time = time.time() - start_time
    
    preds = outputs["preds"]
    
    # Extract results
    confidence = preds[0, 0].mean().item()
    message = (preds[0, 1:] > 0).float().numpy()
    message_logits = preds[0, 1:].numpy()
    
    print(f"✓ Detection complete (time: {inference_time*1000:.2f} ms)")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Detected message (first 32 bits): {message[:32]}")
    
    return {
        'confidence': confidence,
        'message': message,
        'message_logits': message_logits,
        'inference_time': inference_time
    }


def detect_tflite(
    image_path: str,
    model_path: str
) -> dict:
    """
    Detect watermark using TFLite VideoSeal.
    
    Args:
        image_path: Path to watermarked image
        model_path: Path to TFLite model
    
    Returns:
        Detection results dictionary
    """
    print("\n" + "="*70)
    print("Detecting Watermark (TFLite)")
    print("="*70)
    
    # Load detector
    print(f"Loading TFLite model: {Path(model_path).name}")
    detector = VideoSealDetectorTFLite(model_path)
    
    # Display model info
    model_info = detector.get_model_info()
    print(f"\nModel Information:")
    print(f"  Quantization: {model_info['quantization']}")
    print(f"  Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Load image
    print(f"\nLoading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    
    # Detect watermark
    print("Detecting watermark...")
    start_time = time.time()
    result = detector.detect(img)
    inference_time = time.time() - start_time
    
    print(f"✓ Detection complete (time: {inference_time*1000:.2f} ms)")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Detected message (first 32 bits): {result['message'][:32]}")
    
    result['inference_time'] = inference_time
    result['quantization'] = model_info['quantization']
    result['model_size_mb'] = model_info['model_size_mb']
    return result


def compare_results(
    pytorch_result: dict,
    tflite_result: dict,
    original_message: np.ndarray
) -> None:
    """
    Compare PyTorch and TFLite detection results.
    
    Args:
        pytorch_result: PyTorch detection results
        tflite_result: TFLite detection results
        original_message: Original embedded message
    """
    print("\n" + "="*70)
    print("Comparison Results")
    print("="*70)
    
    # Confidence comparison
    print("\n1. Detection Confidence:")
    print(f"  PyTorch:  {pytorch_result['confidence']:.6f}")
    print(f"  TFLite:   {tflite_result['confidence']:.6f}")
    print(f"  Difference: {abs(pytorch_result['confidence'] - tflite_result['confidence']):.6f}")
    
    # Message comparison
    print("\n2. Message Accuracy:")
    pytorch_accuracy = np.mean(
        pytorch_result['message'] == original_message
    ) * 100
    tflite_accuracy = np.mean(
        tflite_result['message'] == original_message
    ) * 100
    
    print(f"  PyTorch:  {pytorch_accuracy:.2f}% ({int(pytorch_accuracy * 2.56)}/256 bits)")
    print(f"  TFLite:   {tflite_accuracy:.2f}% ({int(tflite_accuracy * 2.56)}/256 bits)")
    
    # Agreement between PyTorch and TFLite
    agreement = np.mean(
        pytorch_result['message'] == tflite_result['message']
    ) * 100
    print(f"  Agreement: {agreement:.2f}%")
    
    # Logits comparison
    print("\n3. Message Logits (Raw Predictions):")
    logits_diff = np.abs(
        pytorch_result['message_logits'] - tflite_result['message_logits']
    )
    print(f"  Mean Absolute Error: {np.mean(logits_diff):.6f}")
    print(f"  Max Absolute Error:  {np.max(logits_diff):.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(logits_diff**2)):.6f}")
    
    # Inference time comparison
    print("\n4. Inference Time:")
    print(f"  PyTorch:  {pytorch_result['inference_time']*1000:.2f} ms")
    print(f"  TFLite:   {tflite_result['inference_time']*1000:.2f} ms")
    speedup = pytorch_result['inference_time'] / tflite_result['inference_time']
    print(f"  Speedup:  {speedup:.2f}x {'(TFLite faster)' if speedup > 1 else '(PyTorch faster)'}")
    
    # Model size and quantization info
    if 'quantization' in tflite_result:
        print("\n5. TFLite Model Info:")
        print(f"  Quantization: {tflite_result['quantization']}")
        print(f"  Model size: {tflite_result['model_size_mb']:.2f} MB")
        
        # Estimated size reduction for INT8/FP16
        if tflite_result['quantization'] == 'INT8':
            estimated_float32_size = tflite_result['model_size_mb'] / 0.25  # ~75% reduction
            print(f"  Estimated FLOAT32 size: ~{estimated_float32_size:.2f} MB")
            print(f"  Size reduction: ~75%")
        elif tflite_result['quantization'] == 'FP16':
            estimated_float32_size = tflite_result['model_size_mb'] / 0.5  # ~50% reduction
            print(f"  Estimated FLOAT32 size: ~{estimated_float32_size:.2f} MB")
            print(f"  Size reduction: ~50%")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if pytorch_accuracy > 95 and tflite_accuracy > 95:
        print("✓ Both implementations successfully detected the watermark")
    elif pytorch_accuracy > 95:
        print("⚠ PyTorch detected watermark, but TFLite had lower accuracy")
    elif tflite_accuracy > 95:
        print("⚠ TFLite detected watermark, but PyTorch had lower accuracy")
    else:
        print("✗ Both implementations had difficulty detecting the watermark")
    
    if agreement > 99:
        print("✓ PyTorch and TFLite results are highly consistent")
    elif agreement > 95:
        print("✓ PyTorch and TFLite results are mostly consistent")
    else:
        print("⚠ PyTorch and TFLite results differ significantly")
    
    if np.mean(logits_diff) < 0.01:
        print("✓ Logits are very similar (high numerical accuracy)")
    elif np.mean(logits_diff) < 0.1:
        print("✓ Logits are similar (good numerical accuracy)")
    else:
        print("⚠ Logits differ (numerical differences present)")


def main():
    parser = argparse.ArgumentParser(
        description='Compare VideoSeal PyTorch and TFLite detector performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default test image
  python compare_pytorch_tflite.py
  
  # Use custom image
  python compare_pytorch_tflite.py --input my_image.jpg
  
  # Use different model variant
  python compare_pytorch_tflite.py --model_name pixelseal
  
  # Use existing watermarked image (skip embedding)
  python compare_pytorch_tflite.py --watermarked_image watermarked.jpg --skip_embed
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='../assets/imgs/1.jpg',
        help='Input image path (default: ../assets/imgs/1.jpg)'
    )
    
    parser.add_argument(
        '--watermarked_image',
        type=str,
        default=None,
        help='Pre-watermarked image path (if provided, skips embedding)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory (default: ./outputs)'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='videoseal',
        choices=['videoseal', 'pixelseal', 'chunkyseal'],
        help='VideoSeal model variant (default: videoseal)'
    )
    
    parser.add_argument(
        '--tflite_model',
        type=str,
        default=None,
        help='Path to TFLite model (default: auto-detect)'
    )
    
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['int8', 'fp16'],
        default=None,
        help='Use quantized model (int8 or fp16). Default: FLOAT32'
    )
    
    parser.add_argument(
        '--skip_embed',
        action='store_true',
        help='Skip embedding step (use with --watermarked_image)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-detect TFLite model path
    if args.tflite_model is None:
        models_dir = Path.home() / "work" / "models" / "videoseal_tflite"
        quant_suffix = f"_{args.quantization}" if args.quantization else ""
        args.tflite_model = models_dir / f"videoseal_detector_{args.model_name}_256{quant_suffix}.tflite"
    
    if not Path(args.tflite_model).exists():
        print(f"Error: TFLite model not found: {args.tflite_model}")
        print("\nPlease convert the model first:")
        print("  cd ~/work/videoseal/videoseal")
        print("  python ~/work/ai_edge_torch/.../convert_detector_to_tflite.py")
        return 1
    
    # Step 1: Create watermarked image (or use existing)
    if args.skip_embed and args.watermarked_image:
        watermarked_path = args.watermarked_image
        original_message = None
        print(f"\nUsing pre-watermarked image: {watermarked_path}")
    else:
        watermarked_path = os.path.join(args.output_dir, "watermarked.jpg")
        original_message = create_watermarked_image(
            args.input,
            watermarked_path,
            args.model_name
        )
    
    # Step 2: Detect with PyTorch
    pytorch_result = detect_pytorch(watermarked_path, args.model_name)
    
    # Step 3: Detect with TFLite
    tflite_result = detect_tflite(watermarked_path, args.tflite_model)
    
    # Step 4: Compare results
    if original_message is not None:
        compare_results(pytorch_result, tflite_result, original_message)
    else:
        print("\n" + "="*70)
        print("Comparison Results (No Original Message)")
        print("="*70)
        print("\nAgreement between PyTorch and TFLite:")
        agreement = np.mean(
            pytorch_result['message'] == tflite_result['message']
        ) * 100
        print(f"  Message agreement: {agreement:.2f}%")
        
        logits_diff = np.abs(
            pytorch_result['message_logits'] - tflite_result['message_logits']
        )
        print(f"  Logits MAE: {np.mean(logits_diff):.6f}")
        print(f"  Logits Max Diff: {np.max(logits_diff):.6f}")
    
    print("\n" + "="*70)
    print("✓ Comparison complete!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())

