#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VideoSeal 0.0 TFLite Detector for watermark detection and extraction.

VideoSeal 0.0 is a legacy baseline model with 96-bit message capacity.
This is a smaller, faster alternative to VideoSeal 1.0 (256-bit).

**Status**: ✅ Production Ready (FLOAT32)

Features:
- 96-bit message extraction (vs 256-bit in VideoSeal 1.0)
- Model size: 94.66 MB FLOAT32 (vs 127.57 MB)
- Detection accuracy: 96.88% (validated)
- Faster inference

Example:
    >>> from videoseal.tflite import load_detector00
    >>> detector = load_detector00()
    >>> result = detector.detect("watermarked.jpg")
    >>> print(f"Confidence: {result['confidence']:.3f}")
    >>> print(f"Message: {result['message'][:32]}")
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "TensorFlow is required for TFLite inference. "
        "Install it with: pip install tensorflow"
    )


class VideoSeal00DetectorTFLite:
    """
    TFLite-based VideoSeal 0.0 Detector for watermark detection.
    
    VideoSeal 0.0 uses 96-bit messages (vs 256-bit in VideoSeal 1.0).
    This makes it smaller and faster while maintaining good accuracy.
    
    Args:
        model_path: Path to the TFLite model file
        image_size: Expected input image size (default: 256)
    
    Example:
        >>> detector = VideoSeal00DetectorTFLite("videoseal00_detector_256.tflite")
        >>> img = Image.open("watermarked.jpg")
        >>> results = detector.detect(img)
        >>> print(f"Confidence: {results['confidence']:.3f}")
        >>> print(f"Message: {results['message'][:32]}")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        image_size: int = 256
    ):
        """Initialize the TFLite detector.
        
        Args:
            model_path: Path to the TFLite model file
            image_size: Expected input image size
        """
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.nbits = 96  # VideoSeal 0.0 uses 96-bit messages
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Detect quantization type from filename
        self.quantization = self._detect_quantization()
        
        # Warn about INT8 issues
        if self.quantization == 'INT8':
            print("⚠️  Warning: INT8 detector may have compatibility issues")
            print("   See INT8_BATCH_MATMUL_ISSUE.md for details")
            print("   Recommended: Use FLOAT32 detector")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Validate input shape
        expected_shape = (1, image_size, image_size, 3)  # NHWC format
        actual_shape = tuple(self.input_details[0]['shape'])
        if actual_shape != expected_shape:
            raise ValueError(
                f"Model expects input shape {expected_shape} (NHWC format), "
                f"but got {actual_shape}"
            )
        
        # Validate output shape (1 confidence + 96 message bits)
        expected_output_shape = (1, 97)
        actual_output_shape = tuple(self.output_details[0]['shape'])
        if actual_output_shape != expected_output_shape:
            raise ValueError(
                f"Model expects output shape {expected_output_shape}, "
                f"but got {actual_output_shape}"
            )
        
        # Get model size
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        
        print(f"Loaded VideoSeal 0.0 TFLite detector: {self.model_path.name}")
        print(f"  Quantization: {self.quantization}")
        print(f"  Model size: {model_size_mb:.2f} MB")
        print(f"  Message capacity: {self.nbits} bits")
        print(f"  Input shape: {actual_shape}")
        print(f"  Output shape: {actual_output_shape}")
    
    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> np.ndarray:
        """
        Preprocess image for TFLite inference.
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
        
        Returns:
            Preprocessed image array of shape (1, H, W, 3) in [0, 1] range (NHWC format)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize to expected size
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR
            )
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Image is already in HWC format (NHWC after adding batch dimension)
        # No need to transpose
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _detect_quantization(self) -> str:
        """
        Detect quantization type from model filename.
        
        Returns:
            Quantization type: 'INT8', 'FP16', or 'FLOAT32'
        """
        filename = self.model_path.name.lower()
        if '_int8' in filename or 'int8' in filename:
            return 'INT8'
        elif '_fp16' in filename or 'fp16' in filename:
            return 'FP16'
        else:
            return 'FLOAT32'
    
    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        
        return {
            'model_name': self.model_path.name,
            'model_path': str(self.model_path),
            'model_version': 'VideoSeal 0.0',
            'quantization': self.quantization,
            'model_size_mb': model_size_mb,
            'image_size': self.image_size,
            'nbits': self.nbits,
            'input_shape': tuple(self.input_details[0]['shape']),
            'output_shape': tuple(self.output_details[0]['shape']),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_dtype': str(self.output_details[0]['dtype'])
        }
    
    def detect(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        threshold: float = 0.0
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Detect watermark in an image.
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
            threshold: Threshold for binary message extraction (default: 0.0)
        
        Returns:
            Dictionary containing:
                - confidence: Detection confidence score
                - message: Binary message (96 bits)
                - message_logits: Raw logits for message bits
                - predictions: Full prediction array
        """
        # Preprocess image
        img_array = self.preprocess_image(image)
        
        # Run inference
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            img_array
        )
        self.interpreter.invoke()
        
        # Get predictions
        predictions = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        # Extract results
        confidence = float(predictions[0, 0])
        message_logits = predictions[0, 1:]
        message = (message_logits > threshold).astype(np.int32)
        
        return {
            'confidence': confidence,
            'message': message,
            'message_logits': message_logits,
            'predictions': predictions[0]
        }
    
    def detect_batch(
        self,
        images: list,
        threshold: float = 0.0
    ) -> list:
        """
        Detect watermarks in multiple images.
        
        Args:
            images: List of images (PIL Images, numpy arrays, or paths)
            threshold: Threshold for binary message extraction
        
        Returns:
            List of detection results (one per image)
        """
        results = []
        for image in images:
            result = self.detect(image, threshold)
            results.append(result)
        return results
    
    def extract_message(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        threshold: float = 0.0,
        format: str = 'binary'
    ) -> Union[np.ndarray, str, int]:
        """
        Extract watermark message from an image.
        
        Args:
            image: Input image
            threshold: Threshold for binary message extraction
            format: Output format ('binary', 'hex', 'int', 'bits')
        
        Returns:
            Extracted message in specified format
        """
        result = self.detect(image, threshold)
        message = result['message']
        
        if format == 'binary':
            return message
        elif format == 'hex':
            # Convert binary array to hex string
            message_int = int(''.join(map(str, message)), 2)
            return hex(message_int)
        elif format == 'int':
            # Convert binary array to integer
            return int(''.join(map(str, message)), 2)
        elif format == 'bits':
            # Return as bit string
            return ''.join(map(str, message))
        else:
            raise ValueError(
                f"Unknown format: {format}. "
                f"Choose from: 'binary', 'hex', 'int', 'bits'"
            )
    
    def verify_watermark(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        expected_message: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.5
    ) -> Tuple[bool, float, Optional[float]]:
        """
        Verify if an image contains a watermark.
        
        Args:
            image: Input image
            expected_message: Expected message to compare against (optional)
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            Tuple of (is_watermarked, confidence, bit_accuracy)
            - is_watermarked: True if watermark detected with sufficient confidence
            - confidence: Detection confidence score
            - bit_accuracy: Accuracy if expected_message provided, else None
        """
        result = self.detect(image)
        confidence = result['confidence']
        is_watermarked = confidence > confidence_threshold
        
        bit_accuracy = None
        if expected_message is not None:
            detected_message = result['message']
            bit_accuracy = np.mean(
                detected_message == expected_message
            ).item()
        
        return is_watermarked, confidence, bit_accuracy
    
    def __repr__(self) -> str:
        return (
            f"VideoSeal00DetectorTFLite("
            f"model={self.model_path.name}, "
            f"quantization={self.quantization}, "
            f"image_size={self.image_size}, "
            f"nbits={self.nbits})"
        )


def load_detector00(
    model_path: Optional[Union[str, Path]] = None,
    image_size: int = 256,
    quantization: Optional[str] = None,
    models_dir: Optional[Union[str, Path]] = None
) -> VideoSeal00DetectorTFLite:
    """
    Load a VideoSeal 0.0 TFLite detector.
    
    VideoSeal 0.0 is a legacy baseline model with 96-bit message capacity.
    It's smaller and faster than VideoSeal 1.0 (256-bit).
    
    Args:
        model_path: Direct path to model file (overrides other args)
        image_size: Image size the model was trained on (default: 256)
        quantization: Quantization type (None for FLOAT32, 'int8', 'fp16')
                     Note: INT8 has known issues (BATCH_MATMUL), use FLOAT32
        models_dir: Directory containing TFLite models
                   (default: ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/
                             generative/examples/videoseal0.0/videoseal00_tflite/)
    
    Returns:
        Loaded VideoSeal00DetectorTFLite instance
    
    Example:
        >>> # Load FLOAT32 model from default location
        >>> detector = load_detector00()
        
        >>> # Load from custom path
        >>> detector = load_detector00(
        ...     model_path='/path/to/videoseal00_detector_256.tflite'
        ... )
        
        >>> # Detect watermark
        >>> result = detector.detect("watermarked.jpg")
        >>> print(f"Confidence: {result['confidence']:.3f}")
        >>> print(f"Message: {result['message'][:32]}")
    """
    if model_path is None:
        # Auto-detect model path
        if models_dir is None:
            # Default to ai-edge-torch conversion output
            models_dir = (
                Path.home() / "work" / "ai_edge_torch" / "ai-edge-torch" /
                "ai_edge_torch" / "generative" / "examples" / "videoseal0.0" /
                "videoseal00_tflite"
            )
        else:
            models_dir = Path(models_dir)
        
        # Warn about INT8
        if quantization == 'int8':
            print("⚠️  Warning: INT8 detector has known issues (BATCH_MATMUL)")
            print("   Attempting to load, but may fail at runtime")
            print("   Recommended: Use FLOAT32 detector (94.66 MB, 96.88% accuracy)")
            print("   See: INT8_BATCH_MATMUL_ISSUE.md for details")
        
        # Build filename
        quant_suffix = f"_{quantization}" if quantization else ""
        model_filename = f"videoseal00_detector_{image_size}{quant_suffix}.tflite"
        model_path = models_dir / model_filename
        
        # If not found, try without quantization suffix
        if not model_path.exists() and quantization:
            print(f"Warning: {quantization.upper()} model not found, trying FLOAT32...")
            model_filename = f"videoseal00_detector_{image_size}.tflite"
            model_path = models_dir / model_filename
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n\n"
            f"Expected location: {models_dir}\n"
            f"Expected filename: videoseal00_detector_256.tflite\n\n"
            f"To generate the model, run:\n"
            f"  cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0\n"
            f"  python convert_detector_to_tflite.py --output_dir ./videoseal00_tflite"
        )
    
    return VideoSeal00DetectorTFLite(model_path, image_size)
