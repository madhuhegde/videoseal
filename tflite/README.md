# VideoSeal TFLite Implementation

## Overview

This directory contains TFLite implementations for VideoSeal watermarking:

| Component | Status | Size (FLOAT32) | Size (INT8) | Quality | Functional |
|-----------|--------|----------------|-------------|---------|------------|
| **Detector** | ✅ Production | 127.57 MB | 32.90 MB | Excellent | ✅ Yes |
| **Embedder** | ✅ Production | 90.42 MB | N/A | PSNR 43 dB | ✅ Yes |

**Latest Update** (Jan 4, 2026): Embedder PSNR issue resolved! Now production-ready with 43.29 dB quality and 97.7% detection accuracy.

## Quick Start

### Detection (Recommended - Fully Functional)

```python
from videoseal.tflite import load_detector

# Load INT8 detector (recommended)
detector = load_detector(quantization='int8')

# Detect watermark
result = detector.detect("watermarked.jpg")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Message: {result['message'][:32]}")
```

### Embedding (✅ Now Functional!)

```python
from videoseal.tflite import load_embedder, load_detector
import numpy as np

# Load models
embedder = load_embedder(quantization='float32')  # 90.42 MB
detector = load_detector(quantization='int8')      # 32.90 MB

# Embed watermark
message = np.random.randint(0, 2, 256)
img_w = embedder.embed("original.jpg", message)

# Verify
result = detector.detect(img_w)
print(f"Detection accuracy: {(message == result['message']).mean()*100:.1f}%")
```

## Files

### Working Components

| File | Description | Status |
|------|-------------|--------|
| `detector.py` | TFLite detector implementation | ✅ Working |
| `example.py` | Detector usage examples | ✅ Working |
| `compare_pytorch_tflite.py` | Detector benchmarking | ✅ Working |

### Embedder Components

| File | Description | Status |
|------|-------------|--------|
| `embedder.py` | TFLite embedder implementation | ✅ Working |
| `example_embedder.py` | Embedder usage examples | ✅ Working |
| `compare_embedder.py` | PyTorch vs TFLite comparison | ✅ Working |
| `test_embedder_accuracy.py` | Accuracy test suite | ✅ Working |

## Detector Usage

### Basic Detection

```python
from videoseal.tflite import load_detector

# Load model
detector = load_detector(quantization='int8')

# Detect watermark
result = detector.detect("image.jpg")

# Access results
confidence = result['confidence']
message = result['message']  # 256-bit binary array
message_logits = result['message_logits']  # Raw logits
```

### Batch Processing

```python
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = detector.detect_batch(images)

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['confidence']:.3f}")
```

### Message Extraction

```python
# Extract as binary array
message = detector.extract_message("image.jpg", format='binary')

# Extract as binary string
message_str = detector.extract_message("image.jpg", format='string')

# Extract as hex string
message_hex = detector.extract_message("image.jpg", format='hex')
```

### Watermark Verification

```python
expected_msg = np.random.randint(0, 2, 256)

is_watermarked, confidence, accuracy = detector.verify_watermark(
    "image.jpg",
    expected_message=expected_msg,
    confidence_threshold=0.5
)

print(f"Watermarked: {is_watermarked}")
print(f"Confidence: {confidence:.3f}")
print(f"Bit accuracy: {accuracy:.2f}%")
```

## Model Specifications

### Detector

**FLOAT32**:
- Size: 127.57 MB
- Inference: ~100-200 ms (CPU)
- Bit accuracy: 100% (reference)

**INT8** (Recommended):
- Size: 32.90 MB (74.2% reduction)
- Inference: ~23 ms (CPU, 4.31× faster)
- Bit accuracy: 97.66%

### Embedder

**FLOAT32**:
- Size: 90.42 MB
- Status: ✅ Production Ready
- Quality: PSNR 43.29 dB
- Detection Accuracy: 97.7%

**INT8**:
- Not supported (requires workarounds)
- See memory banks for details

## Performance Comparison

### Detector: PyTorch vs TFLite

| Metric | PyTorch | TFLite FLOAT32 | TFLite INT8 |
|--------|---------|----------------|-------------|
| **Size** | ~500 MB | 127.57 MB | 32.90 MB |
| **Speed (CPU)** | ~100 ms | ~100 ms | ~23 ms |
| **Bit Accuracy** | 100% | 100% | 97.66% |
| **Mobile Ready** | ❌ | ✅ | ✅ |

## Installation

### Requirements

```bash
pip install tensorflow  # or tensorflow-lite
pip install numpy pillow
```

### Model Files

Models are located at:
```
~/work/models/videoseal_tflite/
├── videoseal_detector_videoseal_256.tflite       # FLOAT32 (127.57 MB)
├── videoseal_detector_videoseal_256_int8.tflite  # INT8 (32.90 MB) ✅
└── videoseal_embedder_tflite_256.tflite          # FLOAT32 (90.42 MB) ✅
```

## Examples

### Run Detector Example

```bash
cd ~/work/videoseal/videoseal/tflite
python3 example.py
```

### Compare PyTorch vs TFLite

```bash
python3 compare_pytorch_tflite.py --quantization int8
```

## Documentation

### Memory Banks (Detailed Documentation)
- `.cursor/memory-banks/fixed_msg_embedder/` - Complete embedder documentation
- `.cursor/memory-banks/videoseal-tflite/` - TFLite implementation overview

### Root Documentation
- `../IMAGE_WATERMARK_USAGE.md` - PyTorch CLI usage
- `../EMBEDDER_PSNR_FIX_COMPLETE.md` - PSNR fix summary
- `../EMBEDDER_TFLITE_SUCCESS.md` - Success summary

## Use Cases

### ✅ Ideal For

- **Mobile watermark detection** (INT8, 32.90 MB)
- **Edge device deployment**
- **Real-time detection** (~23 ms)
- **Batch processing**

### ⚠️ Considerations

- **Embedder size**: 90.42 MB (FLOAT32 only, INT8 not supported)
- **Total size**: 123 MB for both embedder + detector
- **Hybrid option**: Server PyTorch embed + mobile TFLite detect (smaller footprint)

## Troubleshooting

### Detector Issues

See `compare_pytorch_tflite.py` for debugging.

### Embedder Issues

**Problem**: Need more details on embedder implementation

**Solution**: See comprehensive documentation in `.cursor/memory-banks/fixed_msg_embedder/`

## Detailed Documentation

For comprehensive documentation, see the memory banks:

**`.cursor/memory-banks/fixed_msg_embedder/`**:
- `overview.md` - High-level summary
- `psnr-fix.md` - PSNR fix details (fixed attenuation)
- `broadcast-to-solution.md` - BROADCAST_TO fix (explicit concatenation)
- `implementation.md` - Implementation guide
- `usage.md` - Usage examples
- `troubleshooting.md` - Common issues
- `int8-limitation.md` - INT8 quantization limitations
- `workarounds.md` - Practical workarounds

## Future Work

1. ✅ Detector: Fully functional (FLOAT32 + INT8)
2. ✅ Embedder: Production-ready (FLOAT32, PSNR 43 dB)
3. 🔄 FP16 quantization: Potential 50% size reduction
4. 🔄 Dynamic shapes: Support multiple image sizes
5. 🔄 INT8 embedder: Requires workarounds

## Contributing

When adding features:
1. Update this README
2. Add examples to `example.py`
3. Update memory bank documentation
4. Run verification tests

## License

This implementation follows the same license as VideoSeal (see LICENSE file in root directory).

---

**Last Updated**: January 4, 2026  
**Detector Status**: ✅ Production Ready (FLOAT32 + INT8)  
**Embedder Status**: ✅ Production Ready (FLOAT32, PSNR 43.29 dB, 97.7% accuracy)

For detailed technical documentation, see `.cursor/memory-banks/fixed_msg_embedder/`

