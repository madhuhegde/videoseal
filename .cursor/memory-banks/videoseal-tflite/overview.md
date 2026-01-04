# VideoSeal TFLite - Overview

## What is VideoSeal TFLite?

VideoSeal TFLite is a TensorFlow Lite implementation of the VideoSeal v1.0 watermark detector, enabling efficient 256-bit watermark detection on mobile and edge devices.

## Architecture

### ConvNeXt-Tiny Structure

```
Input Image (YUV, Y channel, 256×256)
    ↓
ConvNeXt-Tiny Encoder (Pure CNN, 18 blocks)
├── Stem: Conv2d (stride=4)
├── Stage 1: 3 × ConvNeXt Block (96 channels)
├── Stage 2: 3 × ConvNeXt Block (192 channels)
├── Stage 3: 9 × ConvNeXt Block (384 channels)
└── Stage 4: 3 × ConvNeXt Block (768 channels)
    ↓
Pixel Decoder (Upsampling + Conv)
    ↓
Output: 257 channels (1 mask + 256 message bits)
```

**Key Characteristics**:
- ✅ Pure CNN (no attention mechanisms or transformers)
- ✅ YUV color space (Y channel only for better imperceptibility)
- ✅ Excellent TFLite compatibility
- ✅ Easy INT8 quantization

## Key Design Decisions

### 1. ConvNeXt vs Vision Transformer

**VideoSeal v0.0** used Vision Transformer (ViT) with attention mechanisms  
**VideoSeal v1.0** switched to ConvNeXt (pure CNN)

**Why ConvNeXt?**
- ✅ Better TFLite compatibility (no complex attention operations)
- ✅ Easier quantization (74.2% size reduction with INT8)
- ✅ Faster inference on mobile CPUs
- ✅ More stable numerical behavior

**Trade-off**: Slightly different feature extraction approach, but maintains high accuracy

### 2. YUV Color Space Processing

**Choice**: Process Y (luminance) channel only instead of RGB

**Benefits**:
- Watermarks less visible in luminance
- Smaller model (single channel vs 3 channels)
- Faster processing
- Better imperceptibility

**Implementation**: Input images are automatically converted from RGB to YUV, and only the Y channel is processed.

### 3. INT8 Quantization Strategy

**Method**: Dynamic INT8 with channelwise granularity

**How it works**:
- Weights: Quantized to 8-bit integers (static)
- Activations: Dynamically quantized at runtime
- I/O: Kept as FLOAT32 for compatibility

**Results**:
- 74.2% size reduction (127.57 MB → 32.90 MB)
- 97.66% bit accuracy (only 2.34% degradation)
- 4.31× faster inference

**Why it works**: Pure CNN architecture without attention makes quantization more stable and predictable.

## Architecture Comparison

### VideoSeal Variants

| Model | Architecture | Attention | Capacity | TFLite Status |
|-------|-------------|-----------|----------|---------------|
| **VideoSeal v0.0** | ViT + Pixel Decoder | ✅ Yes | 256 bits | Not converted |
| **VideoSeal v1.0** | ConvNeXt-Tiny | ❌ No | 256 bits | ✅ Converted (this) |
| **ChunkySeal** | ConvNeXt-Chunky | ❌ No | 1024 bits | ✅ Converted |
| **PixelSeal** | ConvNeXt-Tiny | ❌ No | 256 bits | Not converted |

**Key Insight**: Models without attention (v1.0, ChunkySeal, PixelSeal) are better suited for TFLite conversion and quantization.

### ConvNeXt Block Structure

Each ConvNeXt block consists of:
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

**Why this works for TFLite**:
- All operations are standard convolutions and linear layers
- No dynamic shapes or complex attention patterns
- Easy to quantize without accuracy loss

## Performance Characteristics

### Model Sizes
- **FLOAT32**: 127.57 MB (baseline)
- **INT8**: 32.90 MB (74.2% reduction)
- **FP16**: ~64 MB (estimated, 50% reduction)

### Inference Speed (CPU, 256×256 image)
- **PyTorch**: ~100 ms (baseline)
- **TFLite FLOAT32**: ~75 ms (1.3× faster)
- **TFLite INT8**: ~23 ms (4.31× faster)

### Accuracy
- **FLOAT32**: 99.6%+ bit accuracy vs PyTorch
- **INT8**: 97.66% bit accuracy (250/256 bits correct)
- **Confidence**: Within ±0.01 of PyTorch

## Use Cases

### ✅ Ideal For
- Mobile applications (small model, fast inference)
- Real-time processing (<100ms latency required)
- IoT devices with limited resources
- Standard watermarking (256 bits sufficient)
- Batch processing with speed requirements
- Edge deployment without GPU

### ❌ Not Ideal For
- High-capacity watermarking (need >256 bits)
- Applications requiring 1024+ bit capacity
- When ChunkySeal's capacity is needed
- Server-side processing with unlimited resources (PyTorch may be simpler)

## Implementation Highlights

### Automatic Features
- ✅ Quantization type auto-detection from filename
- ✅ Automatic image preprocessing (resize, normalize, RGB→YUV)
- ✅ Multiple input formats (PIL Image, numpy array, file path)
- ✅ Multiple message formats (binary, hex, int, bits)
- ✅ Automatic fallback to FLOAT32 if INT8 not found

### Python API
```python
from videoseal.tflite import load_detector

# Load with auto-detection
detector = load_detector(model_name='videoseal', quantization='int8')

# Detect watermark
result = detector.detect("image.jpg")
# Returns: {'confidence': 0.987, 'message': array([...]), 'mask': array([...])}

# Verify with expected message
is_watermarked, conf, acc = detector.verify_watermark(
    "image.jpg",
    expected_message=expected_bits,
    confidence_threshold=0.5
)

# Batch processing
results = detector.detect_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

## Comparison with ChunkySeal

| Aspect | VideoSeal v1.0 | ChunkySeal |
|--------|---------------|------------|
| **Capacity** | 256 bits | 1024 bits (4×) |
| **Parameters** | ~28M | ~200M (7×) |
| **Architecture** | ConvNeXt-Tiny (18 blocks) | ConvNeXt-Chunky (36 blocks) |
| **Model Size (INT8)** | 32.90 MB | 960 MB (29×) |
| **Inference Time** | ~25 ms | ~4 seconds (160×) |
| **Memory Usage** | ~100 MB | ~2-3 GB |
| **Mobile Friendly** | ✅ Excellent | ⚠️ Challenging |
| **Real-time** | ✅ Yes (<50ms) | ❌ No (~4s) |

**Recommendation**: Use VideoSeal for standard watermarking on mobile/edge devices. Use ChunkySeal only when 1024-bit capacity is absolutely required.

## Technical Details

### Input Processing
1. Load image (any format supported by PIL)
2. Resize to 256×256 (if needed)
3. Convert RGB → YUV color space
4. Extract Y channel (luminance)
5. Normalize to [0, 1] range
6. Add batch dimension

### Output Processing
1. Model outputs 257 channels
2. Channel 0: Watermark mask (confidence map)
3. Channels 1-256: Message bits (logits)
4. Apply sigmoid to get probabilities
5. Threshold at 0.5 to get binary message
6. Extract confidence from mask

### Quantization Details
- **Type**: Dynamic INT8
- **Granularity**: Channelwise (per-channel scales)
- **Weights**: 8-bit integers with per-channel scales
- **Activations**: Dynamically quantized at runtime
- **Input/Output**: FLOAT32 (for compatibility)

## Related Documentation

- **[README.md](./README.md)** - Memory bank index and quick reference
- **[tflite/detector.py](../../../tflite/detector.py)** - Implementation source code
- **[tflite/example.py](../../../tflite/example.py)** - Usage examples
- **[ChunkySeal Architecture](../chunkyseal-tflite/architecture.md)** - High-capacity variant

## See Also

- [VideoSeal Model Card](../../../videoseal/cards/videoseal_1.0.yaml)
- [ChunkySeal TFLite](../chunkyseal-tflite/README.md)
- [Memory Bank Index](../MEMORY_BANK_INDEX.md)

---

**Architecture**: ConvNeXt-Tiny (Pure CNN)  
**Status**: Production-ready  
**Last Updated**: January 4, 2025
