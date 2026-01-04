#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VideoSeal TFLite Embedder for watermark embedding.

⚠️  **CURRENT STATUS**: NOT FUNCTIONAL

The embedder converts successfully to TFLite (90.42 MB FLOAT32) but fails
during loading due to TFLite BROADCAST_TO operation limitations.

**Recommended Alternatives**:
1. Use PyTorch embedder: videoseal.load('videoseal_1.0')
2. Hybrid architecture: Server PyTorch embed + Mobile TFLite detect

See: tflite/EMBEDDER_STATUS.md for details

This module is provided for reference and future use when TFLite resolves
the BROADCAST_TO issue.
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


class VideoSealEmbedderTFLite:
    """
    TFLite-based VideoSeal Embedder for watermark embedding.
    
    This embedder can:
    - Embed a 256-bit message into an image
    - Run efficiently on mobile and edge devices
    - Maintain visual quality of the watermarked image
    
    Args:
        model_path: Path to the TFLite model file
        image_size: Expected input image size (default: 256)
    
    Example:
        >>> embedder = VideoSealEmbedderTFLite("videoseal_embedder_tflite_256.tflite")
        >>> img = Image.open("original.jpg")
        >>> message = np.random.randint(0, 2, 256)
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
        self.nbits = 256  # VideoSeal uses 256-bit messages
        
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
        # Input 0: Image [1, 3, 256, 256]
        # Input 1: Message [1, 256]
        expected_img_shape = (1, 3, image_size, image_size)
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
        
        print(f"Loaded TFLite embedder: {self.model_path.name}")
        print(f"  Quantization: {self.quantization}")
        print(f"  Model size: {model_size_mb:.2f} MB")
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
            Preprocessed image as numpy array [1, 3, H, W] in range [0, 1]
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
        
        # Convert from HWC to CHW format
        image = np.transpose(image, (2, 0, 1))
        
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
            Preprocessed message as numpy array [1, 256] with float32 values
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
            output: Model output [1, 3, H, W] in range [0, 1]
        
        Returns:
            Image as numpy array [H, W, 3] in range [0, 255]
        """
        # Remove batch dimension
        output = output[0]
        
        # Convert from CHW to HWC
        output = np.transpose(output, (1, 2, 0))
        
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
            message: Binary message (256 bits)
            return_pil: If True, return PIL Image; otherwise numpy array
        
        Returns:
            Watermarked image (PIL Image or numpy array)
        
        Example:
            >>> embedder = VideoSealEmbedderTFLite("videoseal_embedder_tflite_256.tflite")
            >>> img = Image.open("original.jpg")
            >>> message = np.random.randint(0, 2, 256)
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
            messages: Array of messages [N, 256] or list of messages
            return_pil: If True, return PIL Images; otherwise numpy arrays
        
        Returns:
            List of watermarked images
        
        Example:
            >>> embedder = VideoSealEmbedderTFLite("videoseal_embedder_tflite_256.tflite")
            >>> images = [Image.open(f"img{i}.jpg") for i in range(5)]
            >>> messages = np.random.randint(0, 2, (5, 256))
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
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VideoSealEmbedderTFLite(\n"
            f"  model={self.model_path.name},\n"
            f"  quantization={self.quantization},\n"
            f"  image_size={self.image_size},\n"
            f"  nbits={self.nbits}\n"
            f")"
        )


def load_embedder(
    model_name: str = "videoseal",
    quantization: str = "float32",
    model_dir: Optional[Union[str, Path]] = None
) -> VideoSealEmbedderTFLite:
    """
    Load a VideoSeal TFLite embedder model.
    
    ⚠️  **IMPORTANT**: VideoSeal TFLite embedder is currently NOT FUNCTIONAL
    due to TFLite BROADCAST_TO operation limitations. The model converts
    successfully but fails during tensor allocation.
    
    **Recommended Alternatives**:
    1. Use PyTorch embedder (videoseal.load()) - Full functionality
    2. Hybrid architecture: Server-side PyTorch embed + Mobile TFLite detect
    3. Wait for TFLite BROADCAST_TO fix or model architecture changes
    
    Args:
        model_name: Model name (default: "videoseal")
        quantization: Quantization type - "float32", "int8", or "fp16"
                     Note: ALL quantization types are affected
        model_dir: Directory containing TFLite models
                  (default: ~/work/models/videoseal_tflite)
    
    Returns:
        VideoSealEmbedderTFLite instance (will fail on initialization)
    
    Raises:
        RuntimeError: BROADCAST_TO operation not supported
    
    See Also:
        - Documentation: .cursor/memory-banks/fixed_msg_embedder/int8-limitation.md
        - Workarounds: .cursor/memory-banks/fixed_msg_embedder/workarounds.md
    """
    # Default model directory
    if model_dir is None:
        model_dir = Path.home() / "work" / "models" / "videoseal_tflite"
    else:
        model_dir = Path(model_dir)
    
    # Validate quantization
    quantization = quantization.lower()
    if quantization not in ["float32", "int8", "fp16"]:
        raise ValueError(
            f"Invalid quantization: {quantization}. "
            f"Must be 'float32', 'int8', or 'fp16'"
        )
    
    # Warn about INT8
    if quantization == "int8":
        raise ValueError(
            "INT8 quantization is not supported for VideoSeal embedder due to "
            "TFLite BROADCAST_TO operation limitations. Use 'float32' instead.\n\n"
            "For mobile deployment, consider:\n"
            "1. Use FLOAT32 embedder (90.42 MB)\n"
            "2. Use FP16 if available (~45 MB)\n"
            "3. Use hybrid architecture (server embed + mobile detect)\n\n"
            "See documentation: .cursor/memory-banks/fixed_msg_embedder/int8-limitation.md"
        )
    
    # Construct model filename
    if quantization == "float32":
        model_filename = f"{model_name}_embedder_tflite_256.tflite"
    else:
        model_filename = f"{model_name}_embedder_tflite_256_{quantization}.tflite"
    
    model_path = model_dir / model_filename
    
    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n\n"
            f"Available models in {model_dir}:\n" +
            "\n".join(f"  - {f.name}" for f in model_dir.glob("*.tflite"))
            if model_dir.exists() else f"Directory does not exist: {model_dir}"
        )
    
    return VideoSealEmbedderTFLite(model_path)

