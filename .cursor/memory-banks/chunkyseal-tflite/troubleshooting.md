# ChunkySeal TFLite Troubleshooting Guide

## Table of Contents
- [Common Issues](#common-issues)
- [Model Loading Issues](#model-loading-issues)
- [Inference Issues](#inference-issues)
- [Memory Issues](#memory-issues)
- [Accuracy Issues](#accuracy-issues)
- [Performance Issues](#performance-issues)
- [Conversion Issues](#conversion-issues)

## Common Issues

### Model Not Found

**Symptom**:
```python
FileNotFoundError: Model not found: /path/to/model.tflite
```

**Causes**:
1. Model file doesn't exist
2. Incorrect path
3. Model not converted yet

**Solutions**:

```python
# Solution 1: Verify file exists
from pathlib import Path
model_path = Path("/path/to/model.tflite")
print(f"Exists: {model_path.exists()}")
print(f"Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

# Solution 2: Use explicit path
from videoseal.chunky_tflite import load_detector
detector = load_detector(
    model_path="/full/path/to/chunkyseal_detector_chunkyseal_256_int8.tflite"
)

# Solution 3: Check common locations
possible_locations = [
    "~/work/ai_edge_torch/.../chunkyseal/chunkyseal_tflite/",
    "~/work/models/chunkyseal_tflite/",
    "./chunkyseal_tflite/"
]
```

### TensorFlow Not Installed

**Symptom**:
```python
ImportError: TensorFlow is required for TFLite inference.
Install it with: pip install tensorflow
```

**Solution**:
```bash
# Install TensorFlow
pip install tensorflow

# Or TensorFlow Lite Runtime (smaller)
pip install tflite-runtime

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Wrong Input Shape

**Symptom**:
```python
ValueError: Model expects input shape (1, 3, 256, 256), but got (1, 256, 256, 3)
```

**Cause**: Image format mismatch (HWC vs CHW)

**Solution**:
```python
# The detector handles this automatically
# But if preprocessing manually:
import numpy as np

# Correct: CHW format
img_chw = np.transpose(img_hwc, (2, 0, 1))  # (H, W, C) → (C, H, W)
img_batch = np.expand_dims(img_chw, axis=0)  # Add batch dim

# Use detector's preprocessing
result = detector.detect(img)  # Handles format automatically
```

## Model Loading Issues

### Out of Memory During Loading

**Symptom**:
```
Killed (exit code 137)
# Or
MemoryError: Cannot allocate memory
```

**Cause**: Insufficient RAM to load model

**Solutions**:

```bash
# Solution 1: Check available RAM
free -h

# Solution 2: Use INT8 model (smaller)
# INT8: 960 MB vs FLOAT32: 2.95 GB
detector = load_detector(quantization='int8')

# Solution 3: Close other applications
# Free up RAM before loading

# Solution 4: Increase VM RAM
# If using VM, allocate more RAM in settings
```

### Model Corrupted

**Symptom**:
```
RuntimeError: Could not load model
# Or
ValueError: Invalid TFLite model
```

**Cause**: Incomplete download or corrupted file

**Solutions**:

```bash
# Solution 1: Verify file size
ls -lh chunkyseal_detector_chunkyseal_256_int8.tflite
# Should be: 960 MB (INT8) or 2.95 GB (FLOAT32)

# Solution 2: Check file integrity
file chunkyseal_detector_chunkyseal_256_int8.tflite
# Should show: data

# Solution 3: Re-convert model
cd ~/work/ai_edge_torch/.../chunkyseal
python convert_detector_to_tflite.py --quantize int8 --output_dir ./chunkyseal_tflite
```

### Quantization Mismatch

**Symptom**:
Model loads but gives unexpected results

**Cause**: Using wrong quantization type

**Solution**:

```python
# Check model quantization
detector = load_detector(model_path="model.tflite")
print(f"Detected quantization: {detector.quantization}")

# Verify matches filename
# INT8 model should have "_int8" in filename
# FLOAT32 model should NOT have quantization suffix
```

## Inference Issues

### Slow Inference

**Symptom**: Inference takes >10 seconds per image

**Causes & Solutions**:

```python
# Cause 1: Using FLOAT32 instead of INT8
# Solution: Use INT8 model
detector = load_detector(quantization='int8')

# Cause 2: Large image size
# Solution: Resize to 256×256 (done automatically)
result = detector.detect(large_image)  # Auto-resizes

# Cause 3: Cold start
# Solution: Warm-up inference
_ = detector.detect(dummy_image)  # Warm-up
result = detector.detect(real_image)  # Faster

# Cause 4: CPU-only inference
# Solution: Use GPU if available (TFLite GPU delegate)
# Note: Requires additional setup
```

### Inconsistent Results

**Symptom**: Different results on same image

**Causes & Solutions**:

```python
# Cause 1: Different preprocessing
# Solution: Use detector's preprocessing
result = detector.detect(image)  # Consistent preprocessing

# Cause 2: Floating-point precision
# Solution: This is expected for small differences
# Check if differences are significant:
result1 = detector.detect(image)
result2 = detector.detect(image)
diff = np.abs(result1['confidence'] - result2['confidence'])
print(f"Difference: {diff}")  # Should be ~0

# Cause 3: Random augmentation (shouldn't happen in inference)
# Solution: Ensure model is in eval mode (done automatically)
```

### Wrong Output Shape

**Symptom**:
```python
AssertionError: Expected output shape (1, 1025), got (1, 1024)
```

**Cause**: Wrong model loaded

**Solution**:

```python
# Verify model is ChunkySeal (1024-bit)
info = detector.get_model_info()
print(f"Output shape: {info['output_shape']}")  # Should be (1, 1025)
print(f"Message length: {info['message_length']}")  # Should be 1024

# If wrong model:
# - Check you're loading ChunkySeal, not VideoSeal
# - VideoSeal has 257 output channels (1 + 256)
# - ChunkySeal has 1025 output channels (1 + 1024)
```

## Memory Issues

### Out of Memory During Inference

**Symptom**:
```
MemoryError: Cannot allocate memory
# Or process killed
```

**Solutions**:

```python
# Solution 1: Use INT8 model
detector = load_detector(quantization='int8')  # Uses ~50% less RAM

# Solution 2: Process images one at a time
for image_path in image_paths:
    result = detector.detect(image_path)
    # Process result immediately
    # Don't accumulate results in memory

# Solution 3: Clear memory between batches
import gc
for i, image_path in enumerate(image_paths):
    result = detector.detect(image_path)
    if i % 100 == 0:
        gc.collect()  # Force garbage collection
```

### Memory Leak

**Symptom**: Memory usage grows over time

**Causes & Solutions**:

```python
# Cause 1: Accumulating results
# Solution: Process results immediately

# BAD: Accumulates in memory
results = []
for image in images:
    results.append(detector.detect(image))  # Memory grows

# GOOD: Process immediately
for image in images:
    result = detector.detect(image)
    process_result(result)  # Use and discard

# Cause 2: Not releasing PIL images
# Solution: Explicitly close images
from PIL import Image
img = Image.open("image.jpg")
result = detector.detect(img)
img.close()  # Release memory

# Or use context manager
with Image.open("image.jpg") as img:
    result = detector.detect(img)
```

## Accuracy Issues

### Low Detection Confidence

**Symptom**: Confidence is always near 0 or negative

**Causes & Solutions**:

```python
# Cause 1: Image not watermarked
# Solution: Verify image is actually watermarked
# Low confidence on non-watermarked images is expected

# Cause 2: Wrong model
# Solution: Ensure using ChunkySeal model for ChunkySeal watermarks

# Cause 3: Image degradation
# Solution: Check image quality
from PIL import Image
img = Image.open("image.jpg")
print(f"Size: {img.size}")
print(f"Mode: {img.mode}")  # Should be RGB
print(f"Format: {img.format}")

# Cause 4: Preprocessing issues
# Solution: Let detector handle preprocessing
result = detector.detect(image)  # Don't preprocess manually
```

### Incorrect Message Extraction

**Symptom**: Extracted message doesn't match embedded message

**Causes & Solutions**:

```python
# Cause 1: Wrong threshold
# Solution: Try different thresholds
result_0 = detector.detect(image, threshold=0.0)
result_pos = detector.detect(image, threshold=0.5)
result_neg = detector.detect(image, threshold=-0.5)

# Cause 2: Quantization artifacts (INT8)
# Solution: Use FLOAT32 for reference
detector_f32 = load_detector(quantization=None)
detector_int8 = load_detector(quantization='int8')

result_f32 = detector_f32.detect(image)
result_int8 = detector_int8.detect(image)

# Compare
agreement = np.mean(result_f32['message'] == result_int8['message'])
print(f"Agreement: {agreement*100:.2f}%")  # Should be >95%

# Cause 3: Image corruption
# Solution: Check image integrity
# Watermarks can be destroyed by heavy compression, resizing, etc.
```

### Bit Accuracy Lower Than Expected

**Symptom**: Bit accuracy <90%

**Causes & Solutions**:

```python
# Cause 1: Using INT8 model
# Expected: ~97-98% bit accuracy
# Solution: This is normal for INT8

# Cause 2: Image degradation
# Solution: Check if image was modified after watermarking
# - Compression
# - Resizing
# - Color space conversion
# - Filters/effects

# Cause 3: Wrong expected message
# Solution: Verify expected message is correct
expected_message = ...  # From embedding
detected_message = detector.extract_message(image, format='binary')
accuracy = np.mean(expected_message == detected_message)
print(f"Accuracy: {accuracy*100:.2f}%")
```

## Performance Issues

### Slow Batch Processing

**Symptom**: Processing 1000 images takes too long

**Solutions**:

```python
# Solution 1: Use INT8 model
detector = load_detector(quantization='int8')

# Solution 2: Parallel processing
from concurrent.futures import ThreadPoolExecutor

def process_image(image_path):
    return detector.detect(image_path)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, image_paths))

# Solution 3: Optimize I/O
# Load images in parallel with processing
from queue import Queue
from threading import Thread

def image_loader(paths, queue):
    for path in paths:
        img = Image.open(path)
        queue.put((path, img))
    queue.put(None)  # Sentinel

def image_processor(queue, results):
    while True:
        item = queue.get()
        if item is None:
            break
        path, img = item
        result = detector.detect(img)
        results.append((path, result))

# Use producer-consumer pattern
```

### High Memory Usage

**Symptom**: Process uses >10 GB RAM

**Solutions**:

```python
# Solution 1: Use INT8 model
detector = load_detector(quantization='int8')  # ~2-3 GB vs ~4-6 GB

# Solution 2: Don't load all images at once
# BAD:
images = [Image.open(path) for path in image_paths]  # Loads all
results = detector.detect_batch(images)

# GOOD:
results = detector.detect_batch(image_paths)  # Loads one at a time

# Solution 3: Monitor memory
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

## Conversion Issues

See [Conversion Guide](./conversion.md#troubleshooting-conversion) for detailed conversion troubleshooting.

### Quick Conversion Fixes

```bash
# Issue: Out of memory during conversion
# Solution: Increase VM RAM to 20+ GB

# Issue: Module not found
# Solution: Set PYTHONPATH
export PYTHONPATH=/home/madhuhegde/work/videoseal/videoseal:$PYTHONPATH

# Issue: Checkpoint not found
# Solution: Use detector-only checkpoint
# Location: /mnt/shared/shared/ChunkySeal/chunkyseal_detector_only.pth

# Issue: Conversion hangs
# Solution: Be patient, takes 5-10 minutes
# Monitor: htop or top
```

## Getting Help

### Diagnostic Information

When reporting issues, provide:

```python
# 1. Model information
info = detector.get_model_info()
print("Model Info:")
for key, value in info.items():
    print(f"  {key}: {value}")

# 2. System information
import platform
import sys
print("\nSystem Info:")
print(f"  OS: {platform.system()} {platform.release()}")
print(f"  Python: {sys.version}")
print(f"  TensorFlow: {tf.__version__}")

# 3. Error traceback
# Include full error message and traceback

# 4. Minimal reproduction
# Provide code that reproduces the issue
```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with simple image
import numpy as np
from PIL import Image

# Create test image
test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
result = detector.detect(test_img)
print(f"Test result: {result}")
```

## See Also

- [Usage Guide](./usage.md)
- [Implementation Details](./implementation.md)
- [Conversion Guide](./conversion.md)
- [Quantization Guide](./quantization.md)
- [ChunkySeal TFLite README](../../chunky_tflite/README.md)

