# VideoSeal TFLite Memory Bank

## Overview

This memory bank provides documentation for the **VideoSeal v1.0 TFLite** implementation - a standard-capacity (256-bit) watermark detector optimized for mobile and edge devices.

**Key Specs**: 256-bit capacity • 32.90 MB (INT8) • ~25 ms inference • 97.66% accuracy

## Primary Documentation

The implementation is located in the project's `tflite/` directory:

### Implementation Files

1. **[tflite/detector.py](../../../tflite/detector.py)** (374 lines)
   - `VideoSealDetectorTFLite` class implementation
   - Complete API with preprocessing and message extraction
   - INT8 quantization support

2. **[tflite/example.py](../../../tflite/example.py)** (92 lines)
   - Simple usage examples
   - Detection and verification workflows

3. **[tflite/compare_pytorch_tflite.py](../../../tflite/compare_pytorch_tflite.py)** (430 lines)
   - Benchmarking script for PyTorch vs TFLite comparison
   - Accuracy and speed comparisons

### Memory Bank Files

- **[overview.md](./overview.md)** - High-level introduction with architecture and design decisions

## Quick Start

```python
from videoseal.tflite import load_detector

# Load INT8 model (recommended for mobile)
detector = load_detector(model_name='videoseal', quantization='int8')

# Detect watermark
result = detector.detect("watermarked.jpg")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Message: {result['message'][:32]}")  # First 32 of 256 bits
```

## Key Specifications

| Specification | Value |
|--------------|-------|
| **Capacity** | 256 bits (32 bytes) |
| **Architecture** | ConvNeXt-Tiny (Pure CNN, ~28M params) |
| **FLOAT32** | 127.57 MB, 99.6%+ accuracy |
| **INT8** | 32.90 MB (74.2% reduction), 97.66% accuracy |
| **Speed** | 4.31× faster than PyTorch (INT8) |
| **Input** | 256×256 YUV (Y channel only) |

## VideoSeal vs ChunkySeal

| Feature | VideoSeal v1.0 | ChunkySeal |
|---------|---------------|------------|
| **Capacity** | 256 bits | 1024 bits (4×) |
| **Model Size (INT8)** | 32.90 MB | 960 MB (29×) |
| **Inference Time** | ~25 ms | ~4 seconds (160×) |
| **Use Case** | Standard watermarking | High-capacity |

**Choose VideoSeal** for: Mobile apps, real-time processing, standard watermarking  
**Choose ChunkySeal** for: High-capacity requirements (>256 bits)

## API Overview

```python
class VideoSealDetectorTFLite:
    def detect(image, threshold=0.0) -> dict
    def detect_batch(images, threshold=0.0) -> list
    def extract_message(image, format='binary')
    def verify_watermark(image, expected_message=None)
    def get_model_info() -> dict

def load_detector(
    model_path=None,
    model_name='videoseal',
    quantization=None,  # 'int8', 'fp16', or None
    models_dir=None
) -> VideoSealDetectorTFLite
```

## Performance Summary

From actual benchmarks:

| Model | Size | Bit Accuracy | Speed | Speedup |
|-------|------|--------------|-------|---------|
| **PyTorch** | ~110 MB | 100% (ref) | ~100 ms | 1.0× |
| **TFLite FLOAT32** | 127.57 MB | 99.6% | ~75 ms | 1.3× |
| **TFLite INT8** | 32.90 MB | 97.66% | ~23 ms | 4.31× |

## Related Documentation

### This Memory Bank
- [overview.md](./overview.md) - Architecture and design decisions

### ChunkySeal TFLite
- [ChunkySeal Memory Bank](../chunkyseal-tflite/README.md) - High-capacity variant (1024-bit)

### PyTorch Implementation (Complementary)
- [IMAGE_WATERMARK_USAGE.md](../../../IMAGE_WATERMARK_USAGE.md) - PyTorch CLI for **embedding + detection**
- [image_watermark.py](../../../image_watermark.py) - PyTorch CLI implementation

**Note**: The PyTorch CLI supports both embedding and detection, while TFLite only supports detection.

### Project Documentation
- [VideoSeal Model Card](../../../videoseal/cards/videoseal_1.0.yaml)
- [ChunkySeal vs VideoSeal Comparison](../chunkyseal-tflite/architecture.md)

### Conversion Scripts
- Located in `ai_edge_torch/generative/examples/videoseal/` (separate repository)

## Key Achievements

- ✅ 74.2% size reduction with INT8 quantization
- ✅ 97.66% bit accuracy maintained
- ✅ 4.31× speedup vs PyTorch
- ✅ Production-ready with complete API

## Common Operations

### Detection
```python
result = detector.detect("image.jpg")
# Returns: {'confidence': float, 'message': ndarray, 'mask': ndarray}
```

### Verification
```python
is_watermarked, confidence, accuracy = detector.verify_watermark(
    "image.jpg",
    expected_message=expected_msg,
    confidence_threshold=0.5
)
```

### Message Formats
```python
msg_binary = detector.extract_message("image.jpg", format='binary')  # ndarray
msg_hex = detector.extract_message("image.jpg", format='hex')        # string
msg_int = detector.extract_message("image.jpg", format='int')        # int
msg_bits = detector.extract_message("image.jpg", format='bits')      # string
```

## Troubleshooting

**Model not found**: Use `load_detector()` with auto-detection  
**Slow inference**: Switch to INT8 model  
**Low accuracy**: Verify image contains watermark  
**Memory issues**: INT8 uses 50% less RAM

See [overview.md](./overview.md) for detailed architecture and design information.

---

**Last Updated**: January 4, 2025  
**Status**: Production-ready  
**Documentation**: Complete
