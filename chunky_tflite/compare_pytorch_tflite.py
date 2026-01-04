#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare ChunkySeal PyTorch and TFLite detector performance.

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
from chunky_tflite.detector import ChunkySealDetectorTFLite


def create_watermarked_image(
    image_path: str,
    output_path: str,
    model_name: str = "chunkyseal"
) -> np.ndarray:
    """
    Create a watermarked image using PyTorch ChunkySeal.
    
    Args:
        image_path: Path to input image
        output_path: Path to save watermarked image
        model_name: ChunkySeal model variant
    
    Returns:
        Embedded message (1024-bit binary array)
    """
    print("\n" + "="*70)
    print("Creating Watermarked Image (PyTorch)")
    print("="*70)
    
    # Load model
    print(f"Loading ChunkySeal model: {model_name}")
    model = videoseal.load(model_name)
    model.eval()
    
    # Load image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).float()
    
    print(f"Image shape: {img_tensor.shape}")
    
    # Embed watermark
    print("Embedding watermark (1024 bits)...")
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
    print(f"  Message capacity: {msgs.shape[1]} bits (4× VideoSeal)")
    
    return msgs[0].numpy()


def detect_pytorch(
    image_path: str,
    model_name: str = "chunkyseal"
) -> dict:
    """
    Detect watermark using PyTorch ChunkySeal.
    
    Args:
        image_path: Path to watermarked image
        model_name: ChunkySeal model variant
    
    Returns:
        Detection results dictionary
    """
    print("\n" + "="*70)
    print("Detecting Watermark (PyTorch)")
    print("="*70)
    
    # Load model
    print(f"Loading ChunkySeal model: {model_name}")
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
    print(f"  Message length: {len(message)} bits")
    
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
    Detect watermark using TFLite ChunkySeal.
    
    Args:
        image_path: Path to watermarked image
        model_path: Path to TFLite model
    
    Returns:
        Detection results dictionary
    """
    print("\n" + "="*70)
    print("Detecting Watermark (TFLite)")
    print("="*70)
    
    # Load TFLite model
    print(f"Loading TFLite model: {model_path}")
    detector = ChunkySealDetectorTFLite(model_path)
    
    # Load image
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    
    # Detect watermark
    print("Detecting watermark...")
    start_time = time.time()
    result = detector.detect(img)
    inference_time = time.time() - start_time
    
    print(f"✓ Detection complete (time: {inference_time*1000:.2f} ms)")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Detected message (first 32 bits): {result['message'][:32]}")
    print(f"  Message length: {len(result['message'])} bits")
    
    return {
        'confidence': result['confidence'],
        'message': result['message'],
        'message_logits': result['message_logits'],
        'inference_time': inference_time
    }


def compare_results(
    pytorch_result: dict,
    tflite_result: dict,
    original_message: np.ndarray
):
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
    
    # Compare confidence
    conf_diff = abs(pytorch_result['confidence'] - tflite_result['confidence'])
    print(f"\nConfidence:")
    print(f"  PyTorch: {pytorch_result['confidence']:.6f}")
    print(f"  TFLite:  {tflite_result['confidence']:.6f}")
    print(f"  Difference: {conf_diff:.6f}")
    
    # Compare messages
    pytorch_msg = pytorch_result['message'].astype(int)
    tflite_msg = tflite_result['message'].astype(int)
    
    # Bit accuracy vs original message
    pytorch_accuracy = np.mean(pytorch_msg == original_message) * 100
    tflite_accuracy = np.mean(tflite_msg == original_message) * 100
    
    print(f"\nBit Accuracy (vs original message):")
    print(f"  PyTorch: {pytorch_accuracy:.2f}%")
    print(f"  TFLite:  {tflite_accuracy:.2f}%")
    
    # Agreement between PyTorch and TFLite
    agreement = np.mean(pytorch_msg == tflite_msg) * 100
    print(f"\nPyTorch-TFLite Agreement:")
    print(f"  {agreement:.2f}% ({int(agreement * len(pytorch_msg) / 100)}/{len(pytorch_msg)} bits)")
    
    # Inference time comparison
    speedup = pytorch_result['inference_time'] / tflite_result['inference_time']
    print(f"\nInference Time:")
    print(f"  PyTorch: {pytorch_result['inference_time']*1000:.2f} ms")
    print(f"  TFLite:  {tflite_result['inference_time']*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}×")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"✓ ChunkySeal 1024-bit watermark successfully detected")
    print(f"✓ TFLite bit accuracy: {tflite_accuracy:.2f}%")
    print(f"✓ PyTorch-TFLite agreement: {agreement:.2f}%")
    print(f"✓ TFLite speedup: {speedup:.2f}×")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ChunkySeal PyTorch and TFLite detector performance"
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to input image (default: use assets/imgs/1.jpg)'
    )
    parser.add_argument(
        '--tflite_model',
        type=str,
        default=None,
        help='Path to TFLite model (auto-detect if not provided)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='chunkyseal',
        help='ChunkySeal model variant (default: chunkyseal)'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['float32', 'int8', 'fp16'],
        default=None,
        help='TFLite quantization type (default: auto-detect or float32)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/watermarked_chunky.jpg',
        help='Path to save watermarked image'
    )
    
    args = parser.parse_args()
    
    # Set default image path
    if args.image is None:
        script_dir = Path(__file__).parent.parent
        args.image = str(script_dir / "assets" / "imgs" / "1.jpg")
    
    # Auto-detect TFLite model path
    if args.tflite_model is None:
        # Try to find the model in common locations
        possible_paths = [
            Path.home() / "work" / "models" / "chunkyseal_tflite",
            Path.home() / "work" / "ai_edge_torch" / "ai-edge-torch" / 
                "ai_edge_torch" / "generative" / "examples" / "chunkyseal" / "chunkyseal_tflite",
            Path("./chunkyseal_tflite"),
        ]
        
        quant_suffix = f"_{args.quantization}" if args.quantization else ""
        model_filename = f"chunkyseal_detector_{args.model_name}_256{quant_suffix}.tflite"
        
        for base_path in possible_paths:
            model_path = base_path / model_filename
            if model_path.exists():
                args.tflite_model = str(model_path)
                break
        
        if args.tflite_model is None:
            print(f"Error: Could not find TFLite model: {model_filename}")
            print(f"Searched in:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPlease specify --tflite_model explicitly")
            return
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ChunkySeal PyTorch vs TFLite Comparison")
    print("="*70)
    print(f"Input image: {args.image}")
    print(f"TFLite model: {args.tflite_model}")
    print(f"Model variant: {args.model_name}")
    print(f"Output: {args.output}")
    
    # Step 1: Create watermarked image
    original_message = create_watermarked_image(
        args.image,
        args.output,
        args.model_name
    )
    
    # Step 2: Detect with PyTorch
    pytorch_result = detect_pytorch(args.output, args.model_name)
    
    # Step 3: Detect with TFLite
    tflite_result = detect_tflite(args.output, args.tflite_model)
    
    # Step 4: Compare results
    compare_results(pytorch_result, tflite_result, original_message)


if __name__ == '__main__':
    main()

