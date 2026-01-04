# ChunkySeal TFLite - Overview

## What is ChunkySeal TFLite?

ChunkySeal TFLite is a TensorFlow Lite implementation of the ChunkySeal watermark detector, enabling high-capacity (1024-bit) watermark detection on edge devices.

**Key Specs**: 1024-bit capacity • 960 MB (INT8) • ~4s inference • ConvNeXt-Chunky

## Architecture

### ConvNeXt-Chunky Structure

```
Input Image (RGB, 256×256)
    ↓
ConvNeXt-Chunky Encoder (Pure CNN, 36 blocks)
├── Stem: Conv2d (stride=2)
├── Stage 1: 3 × ConvNeXt Block (362 channels)
├── Stage 2: 3 × ConvNeXt Block (724 channels)
├── Stage 3: 27 × ConvNeXt Block (1448 channels)
└── Stage 4: 3 × ConvNeXt Block (2896 channels)
    ↓
Pixel Decoder (Upsampling + Conv)
    ↓
Output: 1025 channels (1 mask + 1024 message bits)
```

**Key Characteristics**:
- ✅ Pure CNN (no attention mechanisms)
- ✅ RGB color space (all 3 channels)
- ✅ 36 ConvNeXt blocks (2× VideoSeal's 18 blocks)
- ✅ Proportionally scaled channels for 1024-bit capacity

## Key Design Decisions

### 1. Proportional Scaling for High Capacity

**Method**: Dimensions scaled by `sqrt(nbits / 128)`

For 1024 bits: `sqrt(1024/128) = 2.83×` larger channels

**Scaling**:
```
Base dimensions:     [128,  256,  512,  1024]
Scaled dimensions:   [362,  724,  1448, 2896]
```

**Why this works**: Proportional scaling maintains the model's capacity to encode more bits while keeping the architecture consistent.

### 2. ConvNeXt vs Vision Transformer

Same as VideoSeal v1.0 - pure CNN for better TFLite compatibility.

**Benefits for ChunkySeal**:
- ✅ 67.5% size reduction with INT8 (2.95 GB → 960 MB)
- ✅ Stable quantization despite large model size
- ✅ No attention operations that would complicate conversion

### 3. RGB vs YUV Processing

**ChunkySeal**: RGB (all 3 channels)  
**VideoSeal v1.0**: YUV (Y channel only)

**Why RGB for ChunkySeal?**
- Higher capacity needs more information
- Simpler processing (no color space conversion)
- Better feature extraction for 1024 bits

**Trade-off**: Larger model but better capacity utilization.

## Architecture Comparison

| Model | Blocks | Channels (Stage 4) | Parameters | Capacity | Model Size (INT8) |
|-------|--------|-------------------|------------|----------|-------------------|
| **VideoSeal v1.0** | 18 | 768 | ~28M | 256 bits | 32.90 MB |
| **ChunkySeal** | 36 | 2896 | ~200M | 1024 bits | 960 MB |
| **Scaling Factor** | 2× | 3.77× | 7.14× | 4× | 29× |

**Key Insight**: ChunkySeal is not just 4× larger for 4× capacity - it uses proportional scaling which results in ~7× more parameters and ~29× larger model size.

## ConvNeXt Block Details

Each ConvNeXt block (same structure in both VideoSeal and ChunkySeal):
```
Input
  ↓
Depthwise Conv 7×7
  ↓
LayerNorm
  ↓
Linear (4× expansion)
  ↓
GELU activation
  ↓
Linear (projection)
  ↓
Residual connection
  ↓
Output
```

**ChunkySeal difference**: More blocks (36 vs 18) and wider channels (2896 vs 768 in final stage).

## Performance Characteristics

### Model Sizes
- **FLOAT32**: 2.95 GB (baseline)
- **INT8**: 960 MB (67.5% reduction) ✅ Recommended
- **FP16**: ~1.48 GB (50% reduction)

### Inference Speed (CPU, 256×256 image)
- **FLOAT32**: ~4-5 seconds
- **INT8**: ~4 seconds (1.2-1.5× faster)
- **FP16**: ~3.5-4 seconds (1.1-1.3× faster)

### Memory Requirements
- **Model loading**: ~1-2 GB (INT8)
- **Inference**: ~2-3 GB total RAM
- **Minimum recommended**: 4 GB RAM

## Comparison: ChunkySeal vs VideoSeal

| Aspect | ChunkySeal | VideoSeal v1.0 | Ratio |
|--------|-----------|----------------|-------|
| **Capacity** | 1024 bits | 256 bits | 4× |
| **Parameters** | ~200M | ~28M | 7.14× |
| **Model Size (INT8)** | 960 MB | 32.90 MB | 29× |
| **Inference Time** | ~4 seconds | ~25 ms | 160× |
| **Memory Usage** | ~2-3 GB | ~100 MB | 20-30× |
| **Input Channels** | 3 (RGB) | 1 (Y) | 3× |
| **Output Channels** | 1025 | 257 | 4× |

**When to use ChunkySeal**:
- ✅ Need >256 bits capacity
- ✅ Have adequate resources (4+ GB RAM)
- ✅ Latency >1 second acceptable
- ✅ Edge server or high-end mobile

**When to use VideoSeal**:
- ✅ 256 bits sufficient
- ✅ Limited resources (<1 GB RAM)
- ✅ Real-time processing required (<100ms)
- ✅ Mobile or IoT devices

## Use Cases

### ✅ Ideal For ChunkySeal
- High-capacity watermarking (1024 bits needed)
- Edge servers with 4+ GB RAM
- Batch processing (latency not critical)
- Applications needing 4× more data than VideoSeal
- Forensic tracking with detailed metadata

### ❌ Not Ideal For ChunkySeal
- Real-time mobile apps (<100ms latency)
- Low-end mobile devices (<2 GB RAM)
- IoT devices with limited resources
- Applications where 256 bits is sufficient
- Battery-constrained devices

## Implementation Highlights

### Automatic Features
- ✅ Quantization type auto-detection from filename
- ✅ Automatic image preprocessing (resize, normalize)
- ✅ Multiple input formats (PIL Image, numpy array, file path)
- ✅ Multiple message formats (binary, hex, int, bits)
- ✅ Batch processing support

### Python API
```python
from videoseal.chunky_tflite import load_detector

# Load INT8 model (recommended)
detector = load_detector(quantization='int8')

# Detect watermark
result = detector.detect("image.jpg")
# Returns: {'confidence': 0.987, 'message': array([...1024 bits...]), 'mask': array([...])}

# Verify with expected message
is_watermarked, conf, acc = detector.verify_watermark(
    "image.jpg",
    expected_message=expected_bits,  # 1024 bits
    confidence_threshold=0.5
)

# Batch processing
results = detector.detect_batch(["img1.jpg", "img2.jpg"])
```

## Technical Details

### Input Processing
1. Load image (any format supported by PIL)
2. Resize to 256×256 (if needed)
3. Keep RGB format (all 3 channels)
4. Normalize to [0, 1] range
5. Add batch dimension

### Output Processing
1. Model outputs 1025 channels
2. Channel 0: Watermark mask (confidence map)
3. Channels 1-1024: Message bits (logits)
4. Apply sigmoid to get probabilities
5. Threshold at 0.5 to get binary message
6. Extract confidence from mask

### Quantization Details
- **Type**: Dynamic INT8
- **Granularity**: Channelwise (per-channel scales)
- **Weights**: 8-bit integers with per-channel scales
- **Activations**: Dynamically quantized at runtime
- **Input/Output**: FLOAT32 (for compatibility)

**Challenge**: Large model size (960 MB even with INT8) due to proportional scaling for 1024-bit capacity.

## Related Documentation

### This Memory Bank
- **[README.md](./README.md)** - Memory bank index
- **[architecture.md](./architecture.md)** - Detailed architecture analysis
- **[implementation.md](./implementation.md)** - API and code structure
- **[usage.md](./usage.md)** - Usage examples and best practices
- **[conversion.md](./conversion.md)** - TFLite conversion process
- **[quantization.md](./quantization.md)** - Quantization options
- **[troubleshooting.md](./troubleshooting.md)** - Common issues

### Project Implementation
- **[chunky_tflite/detector.py](../../../chunky_tflite/detector.py)** - Detector implementation
- **[chunky_tflite/example.py](../../../chunky_tflite/example.py)** - Usage examples
- **[chunky_tflite/test_int8.py](../../../chunky_tflite/test_int8.py)** - INT8 testing

### Related Memory Banks
- **[VideoSeal TFLite](../videoseal-tflite/README.md)** - Standard-capacity variant (256-bit)

## See Also

- [ChunkySeal Model Card](../../../videoseal/cards/chunkyseal.yaml)
- [VideoSeal vs ChunkySeal Architecture Comparison](./architecture.md)
- [Memory Bank Index](../MEMORY_BANK_INDEX.md)

---

**Architecture**: ConvNeXt-Chunky (Pure CNN)  
**Status**: Production-ready  
**Last Updated**: January 4, 2025
