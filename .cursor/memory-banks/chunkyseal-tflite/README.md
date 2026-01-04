# ChunkySeal TFLite Memory Bank

## Overview

This memory bank contains comprehensive documentation for the ChunkySeal TFLite implementation - a high-capacity (1024-bit) watermark detector optimized for mobile and edge devices.

## Documentation Structure

| File | Description | Lines |
|------|-------------|-------|
| **[overview.md](./overview.md)** | High-level introduction and key features | ~250 |
| **[architecture.md](./architecture.md)** | Detailed architecture analysis | ~400 |
| **[yaml-analysis.md](./yaml-analysis.md)** | YAML configuration analysis for all models | ~400 |
| **[implementation.md](./implementation.md)** | Implementation details and API design | ~350 |
| **[usage.md](./usage.md)** | Usage guide with examples | ~450 |
| **[conversion.md](./conversion.md)** | TFLite conversion process | ~300 |
| **[quantization.md](./quantization.md)** | Quantization options and performance | ~350 |
| **[troubleshooting.md](./troubleshooting.md)** | Common issues and solutions | ~400 |

**Total**: ~2,500 lines of documentation

## Quick Navigation

### Getting Started
1. Start with [overview.md](./overview.md) for introduction
2. Read [usage.md](./usage.md) for practical examples
3. Check [troubleshooting.md](./troubleshooting.md) if you encounter issues

### Technical Deep Dive
1. [architecture.md](./architecture.md) - Understand the model
2. [implementation.md](./implementation.md) - Code structure and API
3. [conversion.md](./conversion.md) - How models are converted

### Optimization
1. [quantization.md](./quantization.md) - Choose quantization type
2. [usage.md](./usage.md) - Best practices

## Key Topics Covered

### Architecture
- ConvNeXt-Chunky encoder (36 blocks)
- No attention mechanisms (pure CNN)
- Proportional scaling for 1024-bit capacity
- Comparison with VideoSeal variants

### Implementation
- `ChunkySealDetectorTFLite` class
- API design and method signatures
- Image preprocessing
- Message extraction formats
- Batch processing

### Quantization
- FLOAT32 (2.95 GB, reference)
- INT8 (960 MB, 67.5% reduction, recommended)
- FP16 (~1.48 GB, balanced)
- Performance comparisons

### Conversion
- PyTorch to TFLite using AI Edge Torch
- Checkpoint preparation
- Quantization recipes
- Verification process

### Usage
- Loading models
- Detection and verification
- Message extraction
- Batch processing
- Best practices

### Troubleshooting
- Model loading issues
- Inference problems
- Memory management
- Accuracy concerns
- Performance optimization

## Related Documentation

### In Project
- [ChunkySeal TFLite Package](../../../chunky_tflite/)
  - `detector.py` - ChunkySeal detector implementation
  - `example.py` - Usage examples
  - `compare_pytorch_tflite.py` - Benchmarking script
  - `test_int8.py` - INT8 model testing

### Conversion Scripts
- [ChunkySeal Conversion](../../../ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/chunkyseal/)
  - `convert_detector_to_tflite.py`
  - `chunkyseal_models.py`
  - `verify_detector_tflite.py`

### Model Cards
- [ChunkySeal YAML](../../../videoseal/cards/chunkyseal.yaml)
- [VideoSeal v1.0 YAML](../../../videoseal/cards/videoseal_1.0.yaml)

## Quick Reference

### Load Model
```python
from videoseal.chunky_tflite import load_detector
detector = load_detector(quantization='int8')
```

### Detect Watermark
```python
result = detector.detect("watermarked.jpg")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Message: {result['message'][:32]}")
```

### Extract Message
```python
message = detector.extract_message("image.jpg", format='binary')
```

### Verify Watermark
```python
is_watermarked, confidence, accuracy = detector.verify_watermark(
    "image.jpg",
    expected_message=expected_msg
)
```

## Model Specifications

| Specification | Value |
|--------------|-------|
| **Architecture** | ConvNeXt-Chunky (Pure CNN) |
| **Capacity** | 1024 bits (128 bytes) |
| **Parameters** | ~200M |
| **Input Size** | 256×256 RGB |
| **Output** | 1025 channels (1 mask + 1024 bits) |
| **FLOAT32 Size** | 2.95 GB |
| **INT8 Size** | 960 MB (67.5% reduction) |
| **Inference Time** | ~4 seconds (CPU, INT8) |

## Performance Summary

### Size Comparison
- FLOAT32: 2.95 GB
- INT8: 960 MB (67.5% smaller) ✅ Recommended
- FP16: ~1.48 GB (50% smaller)

### Speed Comparison (CPU)
- FLOAT32: ~4-5 seconds
- INT8: ~4 seconds (1.2-1.5× faster)
- FP16: ~3.5-4 seconds (1.1-1.3× faster)

### Accuracy (Estimated)
- FLOAT32: 100% (reference)
- INT8: ~97-98% bit accuracy
- FP16: ~99.5% bit accuracy

## Use Cases

### ✅ Ideal For
- High-capacity watermarking (1024 bits)
- Edge servers with adequate resources
- Batch processing applications
- Applications requiring 4× more data than VideoSeal

### ❌ Not Ideal For
- Real-time mobile applications (<100ms latency)
- Low-end mobile devices (limited RAM/storage)
- IoT devices with <1GB RAM
- Applications where 256 bits is sufficient

## Key Achievements

1. ✅ Successfully converted ChunkySeal to TFLite
2. ✅ 67.5% size reduction with INT8 quantization
3. ✅ Maintained 1024-bit capacity
4. ✅ Complete Python API
5. ✅ Production-ready implementation
6. ✅ Comprehensive documentation

## Contributing

When adding to this memory bank:
1. Follow the existing structure
2. Use markdown format
3. Include code examples
4. Cross-reference related docs
5. Keep content up-to-date

## Version History

- **January 3, 2025**: Initial creation
  - Complete documentation set
  - All 7 core documents
  - ~2,500 lines of documentation

## License

This documentation follows the same license as VideoSeal (see LICENSE file in the root directory).

## Contact

For issues or questions:
1. Check [troubleshooting.md](./troubleshooting.md)
2. Review [usage.md](./usage.md) for examples
3. See [ChunkySeal TFLite Package](../../../chunky_tflite/)

---

**Last Updated**: January 3, 2025  
**Documentation Version**: 1.0  
**ChunkySeal TFLite Version**: 1.0

