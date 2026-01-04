#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ChunkySeal TFLite Detector for high-capacity watermark detection.

This module provides a TFLite-based implementation of the ChunkySeal detector
for efficient on-device watermark detection with 1024-bit capacity.
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


class ChunkySealDetectorTFLite:
    """
    TFLite-based ChunkySeal Detector for high-capacity watermark detection.
    
    ChunkySeal provides 1024-bit watermark capacity (4× VideoSeal v1.0) using
    a ConvNeXt-Chunky architecture with proportionally scaled dimensions.
    
    This detector can:
    - Detect if an image contains a watermark
    - Extract the embedded 1024-bit message
    - Run efficiently on mobile and edge devices
    - Support FLOAT32, INT8, and FP16 quantization
    
    Args:
        model_path: Path to the TFLite model file
        image_size: Expected input image size (default: 256)
    
    Example:
        >>> detector = ChunkySealDetectorTFLite("chunkyseal_detector_chunkyseal_256.tflite")
        >>> img = Image.open("watermarked.jpg")
        >>> results = detector.detect(img)
        >>> print(f"Confidence: {results['confidence']:.3f}")
        >>> print(f"Message: {results['message'][:32]}")  # First 32 of 1024 bits
    """
    
    MESSAGE_LENGTH = 1024  # ChunkySeal uses 1024-bit messages
    
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
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Detect quantization type from filename
        self.quantization = self._detect_quantization()
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Validate input shape
        expected_shape = (1, 3, image_size, image_size)
        actual_shape = tuple(self.input_details[0]['shape'])
        if actual_shape != expected_shape:
            raise ValueError(
                f"Model expects input shape {expected_shape}, "
                f"but got {actual_shape}"
            )
        
        # Validate output shape (should be 1 + 1024 = 1025 channels)
        expected_output_shape = (1, 1025)
        actual_output_shape = tuple(self.output_details[0]['shape'])
        if actual_output_shape != expected_output_shape:
            raise ValueError(
                f"Model expects output shape {expected_output_shape}, "
                f"but got {actual_output_shape}"
            )
        
        # Get model size
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        
        print(f"Loaded ChunkySeal TFLite model: {self.model_path.name}")
        print(f"  Quantization: {self.quantization}")
        print(f"  Model size: {model_size_mb:.2f} MB")
        print(f"  Input shape: {actual_shape}")
        print(f"  Output shape: {actual_output_shape}")
        print(f"  Message capacity: {self.MESSAGE_LENGTH} bits (4× VideoSeal)")
    
    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> np.ndarray:
        """
        Preprocess image for TFLite inference.
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
        
        Returns:
            Preprocessed image array of shape (1, 3, H, W) in [0, 1] range
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
        
        # Convert from HWC to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))
        
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
        if '_int8' in filename:
            return 'INT8'
        elif '_fp16' in filename:
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
            'quantization': self.quantization,
            'model_size_mb': model_size_mb,
            'image_size': self.image_size,
            'message_length': self.MESSAGE_LENGTH,
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
                - message: Binary message (1024 bits)
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
            f"ChunkySealDetectorTFLite("
            f"model={self.model_path.name}, "
            f"quantization={self.quantization}, "
            f"image_size={self.image_size}, "
            f"capacity={self.MESSAGE_LENGTH}bits)"
        )


def load_detector(
    model_path: Optional[Union[str, Path]] = None,
    model_name: str = "chunkyseal",
    image_size: int = 256,
    quantization: Optional[str] = None,
    models_dir: Optional[Union[str, Path]] = None
) -> ChunkySealDetectorTFLite:
    """
    Load a ChunkySeal TFLite detector.
    
    Args:
        model_path: Direct path to model file (overrides other args)
        model_name: Model variant name (default: 'chunkyseal')
        image_size: Image size the model was trained on (default: 256)
        quantization: Quantization type ('int8', 'fp16', or None for FLOAT32)
        models_dir: Directory containing TFLite models
    
    Returns:
        Loaded ChunkySealDetectorTFLite instance
    
    Example:
        >>> # Load FLOAT32 model from default location
        >>> detector = load_detector()
        
        >>> # Load INT8 quantized model
        >>> detector = load_detector(quantization='int8')
        
        >>> # Load from custom path
        >>> detector = load_detector(model_path='/path/to/model.tflite')
    """
    if model_path is None:
        # Auto-detect model path
        if models_dir is None:
            # Try multiple common locations
            possible_dirs = [
                Path.home() / "work" / "models" / "chunkyseal_tflite",
                Path.home() / "work" / "ai_edge_torch" / "ai-edge-torch" / 
                    "ai_edge_torch" / "generative" / "examples" / "chunkyseal" / "chunkyseal_tflite",
                Path("./chunkyseal_tflite"),
            ]
            
            # Find first existing directory
            models_dir = None
            for dir_path in possible_dirs:
                if dir_path.exists():
                    models_dir = dir_path
                    break
            
            if models_dir is None:
                models_dir = possible_dirs[0]  # Use first as default
        else:
            models_dir = Path(models_dir)
        
        # Build filename with optional quantization suffix
        quant_suffix = f"_{quantization}" if quantization else ""
        model_filename = f"chunkyseal_detector_{model_name}_{image_size}{quant_suffix}.tflite"
        model_path = models_dir / model_filename
        
        # If quantized model not found, try FLOAT32
        if not model_path.exists() and quantization:
            print(f"Warning: {quantization.upper()} model not found, trying FLOAT32...")
            model_filename = f"chunkyseal_detector_{model_name}_{image_size}.tflite"
            model_path = models_dir / model_filename
    else:
        model_path = Path(model_path)
    
    return ChunkySealDetectorTFLite(model_path, image_size)

