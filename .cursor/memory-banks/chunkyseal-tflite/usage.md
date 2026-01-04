# ChunkySeal TFLite Usage Guide

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Loading Models](#loading-models)
- [Detection](#detection)
- [Message Extraction](#message-extraction)
- [Batch Processing](#batch-processing)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

## Installation

### Requirements

```bash
# Core dependencies
pip install tensorflow  # or tensorflow-lite-runtime
pip install numpy
pip install pillow
```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 2 GB | 4 GB |
| Storage | 1 GB | 2 GB |
| Python | 3.9+ | 3.11+ |

## Quick Start

### 1. Basic Detection

```python
from videoseal.chunky_tflite import load_detector

# Load INT8 model (recommended)
detector = load_detector(quantization='int8')

# Detect watermark
result = detector.detect("watermarked.jpg")

# Display results
print(f"Confidence: {result['confidence']:.3f}")
print(f"Message (first 32 bits): {result['message'][:32]}")
print(f"Total 1 bits: {result['message'].sum()}/1024")
```

### 2. Verify Watermark

```python
from videoseal.chunky_tflite import load_detector

detector = load_detector(quantization='int8')

# Check if watermarked
is_watermarked, confidence, _ = detector.verify_watermark(
    "image.jpg",
    confidence_threshold=0.5
)

if is_watermarked:
    print(f"✓ Watermark detected (confidence: {confidence:.3f})")
else:
    print(f"✗ No watermark detected")
```

## Loading Models

### Method 1: Auto-Detection

```python
from videoseal.chunky_tflite import load_detector

# Auto-detect FLOAT32 model
detector = load_detector()

# Auto-detect INT8 model
detector = load_detector(quantization='int8')

# Auto-detect FP16 model
detector = load_detector(quantization='fp16')
```

**Search Locations**:
1. `~/work/models/chunkyseal_tflite/`
2. `~/work/ai_edge_torch/.../chunkyseal/chunkyseal_tflite/`
3. `./chunkyseal_tflite/`

### Method 2: Explicit Path

```python
from videoseal.chunky_tflite import load_detector

# Load from specific path
detector = load_detector(
    model_path="/path/to/chunkyseal_detector_chunkyseal_256_int8.tflite"
)
```

### Method 3: Custom Directory

```python
from videoseal.chunky_tflite import load_detector

# Specify models directory
detector = load_detector(
    models_dir="/custom/models/dir",
    quantization='int8'
)
```

### Direct Instantiation

```python
from videoseal.chunky_tflite import ChunkySealDetectorTFLite

# Create detector directly
detector = ChunkySealDetectorTFLite(
    model_path="/path/to/model.tflite",
    image_size=256
)
```

## Detection

### Basic Detection

```python
# Detect from file path
result = detector.detect("watermarked.jpg")

# Detect from PIL Image
from PIL import Image
img = Image.open("watermarked.jpg")
result = detector.detect(img)

# Detect from numpy array
import numpy as np
img_array = np.array(img)
result = detector.detect(img_array)
```

### Detection Result

```python
result = detector.detect("image.jpg")

# Result dictionary contains:
{
    'confidence': float,           # Detection confidence
    'message': np.ndarray,         # Binary message (1024,)
    'message_logits': np.ndarray,  # Raw logits (1024,)
    'predictions': np.ndarray      # Full predictions (1025,)
}
```

### Accessing Results

```python
result = detector.detect("image.jpg")

# Get confidence
confidence = result['confidence']

# Get binary message
message = result['message']  # Shape: (1024,), values: 0 or 1

# Get raw logits
logits = result['message_logits']  # Shape: (1024,), float values

# Get all predictions
predictions = result['predictions']  # Shape: (1025,)
# predictions[0] = confidence
# predictions[1:] = message logits
```

### Custom Threshold

```python
# Use custom threshold for message extraction
result = detector.detect("image.jpg", threshold=0.5)

# Default threshold is 0.0
# Positive logits → 1, negative logits → 0
```

## Message Extraction

### Binary Format (Default)

```python
# Returns numpy array of 0s and 1s
message = detector.extract_message("image.jpg", format='binary')
# Shape: (1024,), dtype: int32
# Example: [1, 0, 1, 1, 0, ...]
```

### Hexadecimal Format

```python
# Returns hex string
message_hex = detector.extract_message("image.jpg", format='hex')
# Example: "0x3a5f7c9e..."
# Length: 256 hex characters (1024 bits / 4 bits per hex)
```

### Integer Format

```python
# Returns integer value
message_int = detector.extract_message("image.jpg", format='int')
# Example: 12345678901234567890...
# Very large integer (1024-bit)
```

### Bit String Format

```python
# Returns string of '0' and '1'
message_bits = detector.extract_message("image.jpg", format='bits')
# Example: "10110101..."
# Length: 1024 characters
```

### Format Comparison

```python
# Extract in all formats
binary = detector.extract_message("image.jpg", format='binary')
hex_str = detector.extract_message("image.jpg", format='hex')
integer = detector.extract_message("image.jpg", format='int')
bits = detector.extract_message("image.jpg", format='bits')

print(f"Binary: {binary[:32]}")
print(f"Hex: {hex_str[:34]}...")
print(f"Int: {str(integer)[:50]}...")
print(f"Bits: {bits[:64]}...")
```

## Batch Processing

### Process Multiple Images

```python
from pathlib import Path

# Get image paths
image_paths = list(Path("images/").glob("*.jpg"))

# Process batch
results = detector.detect_batch(image_paths)

# Iterate results
for path, result in zip(image_paths, results):
    print(f"{path.name}:")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Message: {result['message'][:16]}...")
```

### Batch with Different Inputs

```python
# Mix of paths, PIL Images, and numpy arrays
images = [
    "image1.jpg",
    Image.open("image2.jpg"),
    np.array(Image.open("image3.jpg"))
]

results = detector.detect_batch(images)
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def process_image(image_path):
    return detector.detect(image_path)

# Process in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, image_paths))
```

## Advanced Usage

### Watermark Verification with Expected Message

```python
import numpy as np

# Expected message (1024 bits)
expected_message = np.random.randint(0, 2, size=1024)

# Verify with expected message
is_watermarked, confidence, bit_accuracy = detector.verify_watermark(
    "image.jpg",
    expected_message=expected_message,
    confidence_threshold=0.5
)

print(f"Watermarked: {is_watermarked}")
print(f"Confidence: {confidence:.3f}")
print(f"Bit accuracy: {bit_accuracy*100:.2f}%")
```

### Get Model Information

```python
# Get detailed model info
info = detector.get_model_info()

print(f"Model: {info['model_name']}")
print(f"Path: {info['model_path']}")
print(f"Quantization: {info['quantization']}")
print(f"Size: {info['model_size_mb']:.2f} MB")
print(f"Capacity: {info['message_length']} bits")
print(f"Input shape: {info['input_shape']}")
print(f"Output shape: {info['output_shape']}")
print(f"Input dtype: {info['input_dtype']}")
print(f"Output dtype: {info['output_dtype']}")
```

### Custom Image Preprocessing

```python
from PIL import Image
import numpy as np

# Load and preprocess manually
img = Image.open("image.jpg").convert("RGB")
img = img.resize((256, 256), Image.Resampling.BILINEAR)
img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

# Run inference directly
detector.interpreter.set_tensor(
    detector.input_details[0]['index'],
    img_array
)
detector.interpreter.invoke()
predictions = detector.interpreter.get_tensor(
    detector.output_details[0]['index']
)
```

### Timing Inference

```python
import time

# Warm-up
_ = detector.detect("image.jpg")

# Time multiple runs
times = []
for _ in range(10):
    start = time.time()
    _ = detector.detect("image.jpg")
    times.append(time.time() - start)

avg_time = np.mean(times)
print(f"Average inference time: {avg_time*1000:.2f} ms")
```

## Best Practices

### 1. Use INT8 for Production

```python
# Recommended for deployment
detector = load_detector(quantization='int8')

# Benefits:
# - 67.5% smaller (960 MB vs 2.95 GB)
# - Faster inference (~1.2-1.5× speedup)
# - Lower memory usage (~50% less RAM)
# - Minimal accuracy loss (~97-98% bit accuracy)
```

### 2. Reuse Detector Instance

```python
# ✅ Good: Create once, use many times
detector = load_detector(quantization='int8')
for image_path in image_paths:
    result = detector.detect(image_path)

# ❌ Bad: Create for each image
for image_path in image_paths:
    detector = load_detector(quantization='int8')  # Slow!
    result = detector.detect(image_path)
```

### 3. Batch Processing for Multiple Images

```python
# ✅ Good: Use batch processing
results = detector.detect_batch(image_paths)

# ❌ Less efficient: Individual calls
results = [detector.detect(path) for path in image_paths]
```

### 4. Handle Errors Gracefully

```python
from pathlib import Path

def safe_detect(detector, image_path):
    try:
        result = detector.detect(image_path)
        return result
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Use with batch
results = [safe_detect(detector, path) for path in image_paths]
results = [r for r in results if r is not None]  # Filter None
```

### 5. Choose Appropriate Threshold

```python
# For high precision (fewer false positives)
result = detector.detect("image.jpg", threshold=0.5)

# For high recall (fewer false negatives)
result = detector.detect("image.jpg", threshold=-0.5)

# Default (balanced)
result = detector.detect("image.jpg", threshold=0.0)
```

### 6. Monitor Memory Usage

```python
import psutil
import os

process = psutil.Process(os.getpid())

# Before loading
mem_before = process.memory_info().rss / 1024 / 1024
print(f"Memory before: {mem_before:.2f} MB")

# Load detector
detector = load_detector(quantization='int8')

# After loading
mem_after = process.memory_info().rss / 1024 / 1024
print(f"Memory after: {mem_after:.2f} MB")
print(f"Memory used: {mem_after - mem_before:.2f} MB")
```

## Common Patterns

### Pattern 1: Watermark Detection Pipeline

```python
from videoseal.chunky_tflite import load_detector
from pathlib import Path

def detect_watermarks_in_directory(directory, confidence_threshold=0.5):
    """Detect watermarks in all images in a directory."""
    detector = load_detector(quantization='int8')
    
    results = []
    for image_path in Path(directory).glob("*.jpg"):
        result = detector.detect(image_path)
        
        if result['confidence'] > confidence_threshold:
            results.append({
                'path': str(image_path),
                'confidence': result['confidence'],
                'message': result['message']
            })
    
    return results

# Usage
watermarked_images = detect_watermarks_in_directory("images/", 0.5)
print(f"Found {len(watermarked_images)} watermarked images")
```

### Pattern 2: Message Comparison

```python
def compare_messages(detector, image1, image2):
    """Compare messages from two images."""
    msg1 = detector.extract_message(image1, format='binary')
    msg2 = detector.extract_message(image2, format='binary')
    
    # Calculate similarity
    similarity = np.mean(msg1 == msg2)
    
    return {
        'similarity': similarity,
        'matching_bits': int(similarity * 1024),
        'different_bits': int((1 - similarity) * 1024)
    }

# Usage
comparison = compare_messages(detector, "image1.jpg", "image2.jpg")
print(f"Similarity: {comparison['similarity']*100:.2f}%")
print(f"Matching bits: {comparison['matching_bits']}/1024")
```

### Pattern 3: Confidence Histogram

```python
import matplotlib.pyplot as plt

def plot_confidence_distribution(detector, image_paths):
    """Plot distribution of detection confidences."""
    confidences = []
    
    for image_path in image_paths:
        result = detector.detect(image_path)
        confidences.append(result['confidence'])
    
    plt.hist(confidences, bins=50)
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Watermark Detection Confidence Distribution')
    plt.show()

# Usage
plot_confidence_distribution(detector, image_paths)
```

## See Also

- [Implementation Details](./implementation.md)
- [Architecture](./architecture.md)
- [Quantization Guide](./quantization.md)
- [Troubleshooting](./troubleshooting.md)
- [ChunkySeal TFLite README](../../chunky_tflite/README.md)

