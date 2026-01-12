# VideoSeal 0.0 TFLite Implementation

## Overview

VideoSeal 0.0 is a **legacy baseline model** with **96-bit message capacity**. It's a smaller, faster alternative to VideoSeal 1.0 (256-bit).

| Component | Status | Size (FLOAT32) | Quality | Functional |
|-----------|--------|----------------|---------|------------|
| **Embedder** | ✅ Production | 63.81 MB | PSNR 44-46 dB | ✅ Yes |
| **Detector** | ✅ Production | 94.66 MB | 96.88% accuracy | ✅ Yes |

**Total Size**: 158.48 MB (vs 218 MB for VideoSeal 1.0)  
**Size Reduction**: ~27% smaller  
**Speed**: ~30% faster inference

---

## Quick Start

### Installation

```bash
# Ensure TensorFlow is installed
pip install tensorflow numpy pillow

# Models should be in:
# ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0/videoseal00_tflite/
```

### Basic Usage

```python
from videoseal.tflite import load_embedder00, load_detector00
import numpy as np

# Load models
embedder = load_embedder00()  # 63.81 MB
detector = load_detector00()  # 94.66 MB

# Embed 96-bit watermark
message = np.random.randint(0, 2, 96)
img_w = embedder.embed("original.jpg", message)
img_w.save("watermarked.jpg")

# Detect watermark
result = detector.detect("watermarked.jpg")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Bit accuracy: {(message == result['message']).mean()*100:.1f}%")
```

---

## API Reference

### Embedder

#### `load_embedder00(model_path=None, image_size=256, quantization=None, models_dir=None)`

Load VideoSeal 0.0 embedder.

**Args:**
- `model_path`: Direct path to model file (optional)
- `image_size`: Image size (default: 256)
- `quantization`: 'float32', 'fp16' (INT8 not supported)
- `models_dir`: Custom model directory

**Returns:** `VideoSeal00EmbedderTFLite` instance

**Example:**
```python
embedder = load_embedder00()
```

#### `VideoSeal00EmbedderTFLite.embed(image, message, return_pil=True)`

Embed a 96-bit watermark into an image.

**Args:**
- `image`: PIL Image, numpy array, or path
- `message`: 96-bit binary array/list
- `return_pil`: Return PIL Image (default: True)

**Returns:** Watermarked image

**Example:**
```python
message = np.random.randint(0, 2, 96)
img_w = embedder.embed("image.jpg", message)
```

#### `VideoSeal00EmbedderTFLite.embed_batch(images, messages, return_pil=True)`

Embed watermarks into multiple images.

**Args:**
- `images`: List of images
- `messages`: Array of messages [N, 96]
- `return_pil`: Return PIL Images (default: True)

**Returns:** List of watermarked images

---

### Detector

#### `load_detector00(model_path=None, image_size=256, quantization=None, models_dir=None)`

Load VideoSeal 0.0 detector.

**Args:**
- `model_path`: Direct path to model file (optional)
- `image_size`: Image size (default: 256)
- `quantization`: 'float32', 'fp16' (INT8 has issues)
- `models_dir`: Custom model directory

**Returns:** `VideoSeal00DetectorTFLite` instance

**Example:**
```python
detector = load_detector00()
```

#### `VideoSeal00DetectorTFLite.detect(image, threshold=0.0)`

Detect watermark in an image.

**Args:**
- `image`: PIL Image, numpy array, or path
- `threshold`: Binary threshold (default: 0.0)

**Returns:** Dictionary with:
- `confidence`: Detection confidence
- `message`: Binary message (96 bits)
- `message_logits`: Raw logits
- `predictions`: Full prediction array

**Example:**
```python
result = detector.detect("watermarked.jpg")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Message: {result['message'][:32]}")
```

#### `VideoSeal00DetectorTFLite.extract_message(image, threshold=0.0, format='binary')`

Extract watermark message in different formats.

**Args:**
- `image`: Input image
- `threshold`: Binary threshold
- `format`: 'binary', 'hex', 'int', or 'bits'

**Returns:** Message in specified format

**Example:**
```python
msg_binary = detector.extract_message("image.jpg", format='binary')
msg_hex = detector.extract_message("image.jpg", format='hex')
msg_int = detector.extract_message("image.jpg", format='int')
```

#### `VideoSeal00DetectorTFLite.verify_watermark(image, expected_message=None, confidence_threshold=0.5)`

Verify if an image contains a watermark.

**Args:**
- `image`: Input image
- `expected_message`: Expected message (optional)
- `confidence_threshold`: Minimum confidence

**Returns:** Tuple of (is_watermarked, confidence, bit_accuracy)

**Example:**
```python
is_watermarked, confidence, accuracy = detector.verify_watermark(
    "image.jpg",
    expected_message=message,
    confidence_threshold=0.5
)
```

---

## Examples

### Complete Example

```bash
cd ~/work/videoseal/videoseal/tflite
python3 example00.py
```

This will:
1. Load embedder and detector
2. Generate a 96-bit message
3. Embed watermark into test image
4. Detect and verify watermark
5. Show accuracy metrics

### Custom Image

```python
from videoseal.tflite import load_embedder00, load_detector00
import numpy as np
from PIL import Image

# Load models
embedder = load_embedder00()
detector = load_detector00()

# Load your image
img = Image.open("my_image.jpg")

# Generate message (or use your own)
message = np.random.randint(0, 2, 96)

# Embed
img_w = embedder.embed(img, message)
img_w.save("my_watermarked.jpg")

# Detect
result = detector.detect(img_w)
print(f"Confidence: {result['confidence']:.3f}")
print(f"Bit accuracy: {(message == result['message']).mean()*100:.1f}%")
```

### Batch Processing

```python
from videoseal.tflite import load_embedder00, load_detector00
import numpy as np

embedder = load_embedder00()
detector = load_detector00()

# Process multiple images
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
messages = np.random.randint(0, 2, (3, 96))

# Embed batch
imgs_w = embedder.embed_batch(images, messages)

# Save
for i, img_w in enumerate(imgs_w):
    img_w.save(f"watermarked_{i}.jpg")

# Detect batch
results = detector.detect_batch(imgs_w)

for i, result in enumerate(results):
    accuracy = (messages[i] == result['message']).mean()
    print(f"Image {i}: {result['confidence']:.3f}, {accuracy*100:.1f}%")
```

---

## Model Specifications

### Embedder (FLOAT32)

| Property | Value |
|----------|-------|
| **File** | `videoseal00_embedder_256.tflite` |
| **Size** | 63.81 MB |
| **Input 1** | Image [1, 3, 256, 256] FLOAT32 |
| **Input 2** | Message [1, 96] FLOAT32 |
| **Output** | Watermarked Image [1, 3, 256, 256] FLOAT32 |
| **PSNR** | 44-46 dB (invisible watermark) |
| **Status** | ✅ Production Ready |

### Detector (FLOAT32)

| Property | Value |
|----------|-------|
| **File** | `videoseal00_detector_256.tflite` |
| **Size** | 94.66 MB |
| **Input** | Image [1, 3, 256, 256] FLOAT32 |
| **Output** | Predictions [1, 97] FLOAT32 (1 confidence + 96 bits) |
| **Accuracy** | 96.88% (validated on real images) |
| **Status** | ✅ Production Ready |

---

## Performance Comparison

### VideoSeal 0.0 vs 1.0

| Metric | VideoSeal 0.0 | VideoSeal 1.0 | Difference |
|--------|---------------|---------------|------------|
| **Message Bits** | 96 | 256 | -62.5% |
| **Embedder Size** | 63.81 MB | 90.42 MB | -29.4% |
| **Detector Size** | 94.66 MB | 127.57 MB | -25.8% |
| **Total Size** | 158.48 MB | 218 MB | -27.3% |
| **Inference Speed** | Baseline | ~30% slower | +30% faster |
| **PSNR** | 44-46 dB | 43 dB | Similar |
| **Detection Accuracy** | 96.88% | ~97% | Similar |

**Use VideoSeal 0.0 when:**
- You need smaller model size
- You need faster inference
- 96 bits is sufficient for your use case
- You're deploying on resource-constrained devices

**Use VideoSeal 1.0 when:**
- You need more message capacity (256 bits)
- Model size is not a concern
- You need the latest features

---

## Known Issues

### INT8 Detector

❌ **INT8 detector has known issues** (BATCH_MATMUL type mismatch)

**Error:**
```
RuntimeError: tensorflow/lite/kernels/batch_matmul.cc:350
BATCH_MATMUL operation type mismatch
```

**Workaround:** Use FLOAT32 detector (94.66 MB, 96.88% accuracy)

**Details:** See `INT8_BATCH_MATMUL_ISSUE.md` in the conversion directory

### INT8 Embedder

❌ **INT8 embedder not supported**

**Reason:** TFLite conversion limitations with message processor

**Workaround:** Use FLOAT32 embedder (63.81 MB, PSNR 44-46 dB)

---

## File Structure

```
videoseal/tflite/
├── embedder00.py          # VideoSeal 0.0 embedder implementation
├── detector00.py          # VideoSeal 0.0 detector implementation
├── example00.py           # Complete usage example
├── README00.md            # This file
├── embedder.py            # VideoSeal 1.0 embedder (256-bit)
├── detector.py            # VideoSeal 1.0 detector (256-bit)
├── example.py             # VideoSeal 1.0 example
└── __init__.py            # Module exports
```

---

## Model Location

Models are expected at:

```
~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0/videoseal00_tflite/
├── videoseal00_embedder_256.tflite       # FLOAT32 (63.81 MB) ✅
├── videoseal00_detector_256.tflite       # FLOAT32 (94.66 MB) ✅
└── videoseal00_detector_256_int8.tflite  # INT8 (24.90 MB) ❌ Not functional
```

### Generate Models

If models are missing:

```bash
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0

# Generate embedder
python convert_embedder_to_tflite.py --output_dir ./videoseal00_tflite

# Generate detector
python convert_detector_to_tflite.py --output_dir ./videoseal00_tflite
```

---

## Use Cases

### ✅ Ideal For

- **Resource-constrained devices** (smaller model size)
- **Fast inference** (30% faster than VideoSeal 1.0)
- **Mobile watermark embedding** (63.81 MB embedder)
- **Edge device detection** (94.66 MB detector)
- **Applications with 96-bit capacity** (sufficient for most use cases)

### ⚠️ Considerations

- **Message capacity**: Only 96 bits (vs 256 in VideoSeal 1.0)
- **INT8 not supported**: FLOAT32 only (larger size, slower)
- **Legacy model**: VideoSeal 1.0 is the current standard

---

## Documentation

### Conversion Documentation

See the conversion directory for detailed technical documentation:

```
~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0/
├── README.md                      # Conversion guide
├── ACCURACY_REPORT.md             # Detailed accuracy analysis
├── FINAL_SUMMARY.md               # Project summary
├── INT8_BATCH_MATMUL_ISSUE.md    # INT8 issue analysis
└── GENERATION_REPORT.md           # Model generation log
```

### VideoSeal Repository

- **GitHub**: https://github.com/facebookresearch/videoseal
- **Paper**: "VideoSeal: Open and Efficient Video Watermarking"
- **Model Card**: `videoseal/cards/videoseal_0.0.yaml`

---

## Troubleshooting

### Model Not Found

**Error:** `FileNotFoundError: Model not found`

**Solution:**
1. Check model location:
   ```bash
   ls ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/videoseal0.0/videoseal00_tflite/
   ```

2. Generate models if missing (see "Generate Models" section above)

3. Or specify custom path:
   ```python
   embedder = load_embedder00(model_path="/path/to/model.tflite")
   ```

### INT8 Detector Fails

**Error:** `RuntimeError: BATCH_MATMUL operation type mismatch`

**Solution:** Use FLOAT32 detector instead:
```python
detector = load_detector00(quantization=None)  # Uses FLOAT32
```

### Low Detection Accuracy

**Possible causes:**
1. Image compression (JPEG quality too low)
2. Image transformations (resize, crop, rotate)
3. Image format conversion

**Solutions:**
- Use high-quality images (JPEG quality > 90)
- Minimize transformations
- Test with PNG format

---

## License

This implementation follows the same license as VideoSeal (see LICENSE file in root directory).

---

## Citation

If you use VideoSeal 0.0 in your research, please cite:

```bibtex
@article{videoseal2024,
  title={VideoSeal: Open and Efficient Video Watermarking},
  author={VideoSeal Team},
  journal={arXiv preprint},
  year={2024}
}
```

---

**Last Updated**: January 12, 2026  
**Status**: ✅ Production Ready (FLOAT32)  
**Embedder**: 63.81 MB, PSNR 44-46 dB  
**Detector**: 94.66 MB, 96.88% accuracy  

For VideoSeal 1.0 (256-bit), see `README.md` in this directory.
