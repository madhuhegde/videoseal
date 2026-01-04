#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ChunkySeal TFLite - High-Capacity Watermark Detection on Edge Devices

ChunkySeal is a high-capacity watermarking model that can embed and detect
1024-bit watermarks (4× the capacity of VideoSeal v1.0). This package provides
TFLite-based implementations for efficient on-device watermark detection.

Key Features:
- 1024-bit watermark capacity (vs 256-bit in VideoSeal)
- ConvNeXt-Chunky detector architecture (~200M params)
- Efficient TFLite inference for mobile and edge devices
- Support for FLOAT32, INT8, and FP16 quantization
- RGB processing (vs YUV in VideoSeal)

Example:
    >>> from videoseal.chunky_tflite import load_detector
    >>> detector = load_detector("chunkyseal_detector_chunkyseal_256.tflite")
    >>> results = detector.detect("watermarked.jpg")
    >>> print(f"Confidence: {results['confidence']:.3f}")
    >>> print(f"Message bits: {results['message'][:32]}")
"""

from .detector import ChunkySealDetectorTFLite, load_detector

__all__ = [
    'ChunkySealDetectorTFLite',
    'load_detector',
]

