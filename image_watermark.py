#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Image Watermarking Script using VideoSeal
Provides command-line interface for embedding and detecting watermarks in images.
"""

import argparse
import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image

# Add the videoseal package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import videoseal


def embed_watermark(input_path: str, output_path: str, model, device: str = "cpu"):
    """
    Embed a watermark into an image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save watermarked image
        model: Loaded VideoSeal model
        device: Device to run on ('cpu' or 'cuda')
    """
    print(f"Loading image from: {input_path}")
    
    # Load image
    img = Image.open(input_path).convert("RGB")
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).float().to(device)
    
    print(f"Image shape: {img_tensor.shape}")
    print("Embedding watermark...")
    
    # Embed watermark
    with torch.no_grad():
        outputs = model.embed(img_tensor, is_video=False)
    
    # Extract results
    imgs_w = outputs["imgs_w"]  # Watermarked image
    msgs = outputs["msgs"]  # Embedded message (256-bit binary vector)
    
    # Convert back to PIL and save
    to_pil = T.ToPILImage()
    img_w_pil = to_pil(imgs_w[0].cpu())
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    img_w_pil.save(output_path)
    
    print(f"✓ Watermarked image saved to: {output_path}")
    print(f"✓ Embedded message (256 bits): {msgs[0].cpu().numpy()[:32]}... (showing first 32 bits)")
    print(f"  Message shape: {msgs.shape}")
    
    return msgs


def detect_watermark(input_path: str, model, device: str = "cpu"):
    """
    Detect and extract watermark from an image.
    
    Args:
        input_path: Path to watermarked image
        model: Loaded VideoSeal model
        device: Device to run on ('cpu' or 'cuda')
    """
    print(f"Loading watermarked image from: {input_path}")
    
    # Load image
    img = Image.open(input_path).convert("RGB")
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).float().to(device)
    
    print(f"Image shape: {img_tensor.shape}")
    print("Detecting watermark...")
    
    # Detect watermark
    with torch.no_grad():
        outputs = model.detect(img_tensor, is_video=False)
    
    # Extract predictions
    preds = outputs["preds"]  # Shape: [batch, 1+nbits, ...]
    
    # Extract binary message (skip first channel which is detection mask)
    detected_msg = (preds[0, 1:] > 0).float()
    
    # Calculate detection confidence (average of absolute prediction values)
    confidence = preds[0, 1:].abs().mean().item()
    
    print(f"✓ Watermark detected!")
    print(f"✓ Detected message (256 bits): {detected_msg.cpu().numpy()[:32]}... (showing first 32 bits)")
    print(f"  Message shape: {detected_msg.shape}")
    print(f"  Detection confidence: {confidence:.4f}")
    
    # Detection mask (first channel)
    detection_mask = preds[0, 0].mean().item()
    print(f"  Detection mask score: {detection_mask:.4f}")
    
    return detected_msg, confidence


def main():
    parser = argparse.ArgumentParser(
        description="VideoSeal Image Watermarking - Embed and detect watermarks in images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed watermark into an image
  python image_watermark.py --embed --input image.jpg --output watermarked.jpg
  
  # Detect watermark from an image
  python image_watermark.py --detect --input watermarked.jpg
  
  # Use GPU for faster processing
  python image_watermark.py --embed --input image.jpg --output watermarked.jpg --device cuda
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--embed", action="store_true", 
                           help="Embed watermark into an image")
    mode_group.add_argument("--detect", action="store_true",
                           help="Detect watermark from an image")
    
    # Common arguments
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--output", type=str,
                       help="Path to save watermarked image (required for --embed)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run on (default: cpu)")
    parser.add_argument("--model", type=str, default="videoseal",
                       help="Model to use (default: videoseal)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.embed and not args.output:
        parser.error("--output is required when using --embed")
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    print(f"Loading VideoSeal model: {args.model}")
    
    # Load model
    try:
        model = videoseal.load(args.model)
        model.eval()
        model.to(device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Execute mode
    try:
        if args.embed:
            embed_watermark(args.input, args.output, model, device)
        elif args.detect:
            detect_watermark(args.input, model, device)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

