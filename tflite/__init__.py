# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TFLite inference module for VideoSeal watermark embedding and detection.

Supports:
- VideoSeal 1.0: 256-bit watermarking (detector.py, embedder.py)
- VideoSeal 0.0: 96-bit watermarking (detector00.py, embedder00.py)
"""

from .detector import VideoSealDetectorTFLite, load_detector
from .embedder import VideoSealEmbedderTFLite, load_embedder
from .detector00 import VideoSeal00DetectorTFLite, load_detector00
from .embedder00 import VideoSeal00EmbedderTFLite, load_embedder00

__all__ = [
    # VideoSeal 1.0 (256-bit)
    'VideoSealDetectorTFLite',
    'VideoSealEmbedderTFLite',
    'load_detector',
    'load_embedder',
    # VideoSeal 0.0 (96-bit)
    'VideoSeal00DetectorTFLite',
    'VideoSeal00EmbedderTFLite',
    'load_detector00',
    'load_embedder00',
]

