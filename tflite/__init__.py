# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TFLite inference module for VideoSeal watermark embedding and detection.
"""

from .detector import VideoSealDetectorTFLite, load_detector
from .embedder import VideoSealEmbedderTFLite, load_embedder

__all__ = [
    'VideoSealDetectorTFLite',
    'VideoSealEmbedderTFLite',
    'load_detector',
    'load_embedder',
]

