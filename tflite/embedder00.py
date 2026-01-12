#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VideoSeal 0.0 TFLite Embedder for watermark embedding.

VideoSeal 0.0 is a legacy baseline model with 96-bit message capacity.
This is a smaller, faster alternative to VideoSeal 1.0 (256-bit).

**Status**: ✅ Production Ready (FLOAT32)

Features:
- 96-bit message capacity (vs 256-bit in VideoSeal 1.0)
- Smaller model size: 63.81 MB (vs 90.42 MB)
- Faster inference
- PSNR: ~46 dB (invisible watermark)

Example:
    >>> from videoseal.tflite import load_embedder00
    >>> embedder = load_embedder00()
    >>> message = np.random.randint(0, 2, 96)
    >>> img_w = embedder.embed("original.jpg", message)
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


class VideoSeal00EmbedderTFLite:
    """
    TFLite-based VideoSeal 0.0 Embedder for watermark embedding.
    
    VideoSeal 0.0 uses 96-bit messages (vs 256-bit in VideoSeal 1.0).
    This makes it smaller and faster while maintaining good quality.
    
    Args:
        model_path: Path to the TFLite model file
        image_size: Expected input image size (default: 256)
    
    Example:
        >>> embedder = VideoSeal00EmbedderTFLite("videoseal00_embedder_256.tflite")
        >>> img = Image.open("original.jpg")
        >>> message = np.random.randint(0, 2, 96)
        >>> img_w = embedder.embed(img, message)
        >>> img_w.save("watermarked.jpg")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        image_size: int = 256
    ):
        """Initialize the TFLite embedder.
        
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
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Validate input shapes
        # Input 0: Image [1, 256, 256, 3] (NHWC format)
        # Input 1: Message [1, 96]
        expected_img_shape = (1, image_size, image_size, 3)
        expected_msg_shape = (1, self.nbits)
        
        actual_img_shape = tuple(self.input_details[0]['shape'])
        actual_msg_shape = tuple(self.input_details[1]['shape'])
        
        if actual_img_shape != expected_img_shape:
            raise ValueError(
                f"Model expects image shape {expected_img_shape}, "
                f"but got {actual_img_shape}"
            )
        
        if actual_msg_shape != expected_msg_shape:
            raise ValueError(
                f"Model expects message shape {expected_msg_shape}, "
                f"but got {actual_msg_shape}"
            )
        
        # Get model size
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        
        print(f"Loaded VideoSeal 0.0 TFLite embedder: {self.model_path.name}")
        print(f"  Quantization: {self.quantization}")
        print(f"  Model size: {model_size_mb:.2f} MB")
        print(f"  Message capacity: {self.nbits} bits")
        print(f"  Image shape: {actual_img_shape}")
        print(f"  Message shape: {actual_msg_shape}")
        print(f"  Output shape: {tuple(self.output_details[0]['shape'])}")
    
    def _detect_quantization(self) -> str:
        """Detect quantization type from filename."""
        filename = self.model_path.name.lower()
        if 'int8' in filename:
            return 'INT8'
        elif 'fp16' in filename:
            return 'FP16'
        else:
            return 'FLOAT32'
    
    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> np.ndarray:
        """
        Preprocess an image for the embedder.
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
        
        Returns:
            Preprocessed image as numpy array [1, H, W, 3] in range [0, 1] (NHWC format)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Convert to numpy if PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # RGBA
            image = image[..., :3]
        
        # Resize if needed
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            pil_img = Image.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize((self.image_size, self.image_size), Image.LANCZOS)
            image = np.array(pil_img)
        
        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Image is already in HWC format (NHWC after adding batch dimension)
        # No need to transpose
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def preprocess_message(
        self,
        message: Union[np.ndarray, list, str]
    ) -> np.ndarray:
        """
        Preprocess a message for the embedder.
        
        Args:
            message: Binary message (numpy array, list, or binary string)
        
        Returns:
            Preprocessed message as numpy array [1, 96] with float32 values
        """
        # Convert binary string to array
        if isinstance(message, str):
            message = np.array([int(b) for b in message])
        
        # Convert list to array
        if isinstance(message, list):
            message = np.array(message)
        
        # Validate shape
        if message.shape[0] != self.nbits:
            raise ValueError(
                f"Message must have {self.nbits} bits, got {message.shape[0]}"
            )
        
        # Ensure float32
        message = message.astype(np.float32)
        
        # Add batch dimension if needed
        if message.ndim == 1:
            message = np.expand_dims(message, axis=0)
        
        return message
    
    def postprocess_image(self, output: np.ndarray) -> np.ndarray:
        """
        Postprocess the embedder output.
        
        Args:
            output: Model output [1, H, W, 3] in range [0, 1] (NHWC format)
        
        Returns:
            Image as numpy array [H, W, 3] in range [0, 255]
        """
        # Remove batch dimension
        output = output[0]
        
        # Output is already in HWC format, no need to transpose
        
        # Clip to [0, 1] and convert to uint8
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)
        
        return output
    
    def embed(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        message: Union[np.ndarray, list, str],
        return_pil: bool = True
    ) -> Union[Image.Image, np.ndarray]:
        """
        Embed a watermark message into an image.
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
            message: Binary message (96 bits)
            return_pil: If True, return PIL Image; otherwise numpy array
        
        Returns:
            Watermarked image (PIL Image or numpy array)
        
        Example:
            >>> embedder = VideoSeal00EmbedderTFLite("videoseal00_embedder_256.tflite")
            >>> img = Image.open("original.jpg")
            >>> message = np.random.randint(0, 2, 96)
            >>> img_w = embedder.embed(img, message)
            >>> img_w.save("watermarked.jpg")
        """
        # Preprocess inputs
        img_tensor = self.preprocess_image(image)
        msg_tensor = self.preprocess_message(message)
        
        # Set input tensors
        self.interpreter.set_tensor(self.input_details[0]['index'], img_tensor)
        self.interpreter.set_tensor(self.input_details[1]['index'], msg_tensor)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Postprocess
        img_w = self.postprocess_image(output)
        
        # Convert to PIL if requested
        if return_pil:
            img_w = Image.fromarray(img_w)
        
        return img_w
    
    def embed_batch(
        self,
        images: list,
        messages: Union[np.ndarray, list],
        return_pil: bool = True
    ) -> list:
        """
        Embed watermarks into multiple images.
        
        Args:
            images: List of images
            messages: Array of messages [N, 96] or list of messages
            return_pil: If True, return PIL Images; otherwise numpy arrays
        
        Returns:
            List of watermarked images
        
        Example:
            >>> embedder = VideoSeal00EmbedderTFLite("videoseal00_embedder_256.tflite")
            >>> images = [Image.open(f"img{i}.jpg") for i in range(5)]
            >>> messages = np.random.randint(0, 2, (5, 96))
            >>> imgs_w = embedder.embed_batch(images, messages)
        """
        if isinstance(messages, list):
            messages = np.array(messages)
        
        if len(images) != len(messages):
            raise ValueError(
                f"Number of images ({len(images)}) must match "
                f"number of messages ({len(messages)})"
            )
        
        results = []
        for img, msg in zip(images, messages):
            img_w = self.embed(img, msg, return_pil=return_pil)
            results.append(img_w)
        
        return results
    
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
            'message_shape': tuple(self.input_details[1]['shape']),
            'output_shape': tuple(self.output_details[0]['shape']),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_dtype': str(self.output_details[0]['dtype'])
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VideoSeal00EmbedderTFLite(\n"
            f"  model={self.model_path.name},\n"
            f"  quantization={self.quantization},\n"
            f"  image_size={self.image_size},\n"
            f"  nbits={self.nbits}\n"
            f")"
        )


def load_embedder00(
    model_path: Optional[Union[str, Path]] = None,
    image_size: int = 256,
    quantization: Optional[str] = None,
    models_dir: Optional[Union[str, Path]] = None
) -> VideoSeal00EmbedderTFLite:
    """
    Load a VideoSeal 0.0 TFLite embedder model.
    
    VideoSeal 0.0 is a legacy baseline model with 96-bit message capacity.
    It's smaller and faster than VideoSeal 1.0 (256-bit).
    
    Args:
        model_path: Direct path to model file (overrides other args)
        image_size: Image size the model was trained on (default: 256)
        quantization: Quantization type (None for FLOAT32, 'int8', 'fp16')
                     Note: INT8 not supported for embedder
        models_dir: Directory containing TFLite models
                   (default: ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/
                             generative/examples/videoseal0.0/videoseal00_tflite/)
    
    Returns:
        VideoSeal00EmbedderTFLite instance
    
    Example:
        >>> # Load FLOAT32 model from default location
        >>> embedder = load_embedder00()
        
        >>> # Load from custom path
        >>> embedder = load_embedder00(
        ...     model_path='/path/to/videoseal00_embedder_256.tflite'
        ... )
        
        >>> # Embed watermark
        >>> message = np.random.randint(0, 2, 96)
        >>> img_w = embedder.embed("original.jpg", message)
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
            raise ValueError(
                "INT8 quantization is not supported for VideoSeal 0.0 embedder.\n"
                "Use FLOAT32 instead (63.81 MB, PSNR 46.56 dB).\n\n"
                "For mobile deployment, consider:\n"
                "1. Use FLOAT32 embedder (63.81 MB)\n"
                "2. Use FP16 if available (~32 MB)\n"
                "3. Use hybrid architecture (server embed + mobile detect)"
            )
        
        # Build filename
        quant_suffix = f"_{quantization}" if quantization else ""
        model_filename = f"videoseal00_embedder_{image_size}{quant_suffix}.tflite"
        model_path = models_dir / model_filename
        
        # If not found, try without quantization suffix
        if not model_path.exists() and quantization:
            print(f"Warning: {quantization.upper()} model not found, trying FLOAT32...")
            model_filename = f"videoseal00_embedder_{image_size}.tflite"
            model_path = models_dir / model_filename
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n\n"
            f"Expected location: {models_dir}\n"
            f"Expected filename: videoseal00_embedder_256.tflite\n\n"
            f"To generate the model, run:\n"
            f"  cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0\n"
            f"  python convert_embedder_to_tflite.py --output_dir ./videoseal00_tflite"
        )
    
    return VideoSeal00EmbedderTFLite(model_path, image_size)
